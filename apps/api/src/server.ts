import { createServer as createHttpServer } from 'node:http';

import { Router } from 'express';

import { API_PREFIX, createApp } from './app.js';
import { loadEnv, usesMemoryMongo, type Env } from './config/env.js';
import { connectMongo, disconnectMongo, mongoReady, syncIndexes } from './db/connect.js';
import './db/models.js';
import { backfillPredictions } from './demo/backfill.js';
import { seedDemoData } from './demo/seed.js';
import { InferenceClient } from './lib/inferenceClient.js';
import { createLogger } from './lib/logger.js';
import { loadKeys } from './lib/tokens.js';
import { alertRoutes } from './modules/alerts/routes.js';
import { authRoutes } from './modules/auth/routes.js';
import { clinicalRoutes } from './modules/clinical/routes.js';
import { dashboardRoutes } from './modules/dashboard/routes.js';
import { predictionRoutes } from './modules/predictions/routes.js';
import { simulationRoutes } from './modules/simulation/routes.js';
import { SimulationService } from './modules/simulation/service.js';
import { SocketBus } from './realtime/bus.js';
import { createSocketServer } from './realtime/socket.js';
import { InProcessPredictionService } from './services/predictionService.js';
import { requireAuth } from './middleware/auth.js';

/**
 * Process entrypoint. Config is validated before anything else starts, so a
 * misconfigured deployment fails immediately and loudly rather than at the
 * first request that happens to need the missing value.
 */
async function main(): Promise<void> {
  const env = loadEnv();
  const logger = createLogger({
    level: env.LOG_LEVEL,
    isProduction: env.NODE_ENV === 'production',
  });

  // ── Database ────────────────────────────────────────────────────────────
  let stopMemoryServer: (() => Promise<void>) | undefined;
  let mongoUri = env.MONGO_URI;

  if (usesMemoryMongo(env)) {
    // Imported lazily so the dev dependency is never required in production.
    const { MongoMemoryServer } = await import('mongodb-memory-server');
    logger.info('starting in-process MongoDB (demo mode)…');
    const server = await MongoMemoryServer.create({
      binary: { version: '7.0.14' },
      instance: { dbName: env.MONGO_DB },
    });
    mongoUri = server.getUri(env.MONGO_DB);
    stopMemoryServer = async () => {
      await server.stop();
    };
  }

  await connectMongo(mongoUri, logger);
  await syncIndexes(logger);

  // Time-series collections must be created explicitly or Mongo makes an
  // ordinary collection on first insert.
  const { VitalsModel } = await import('./db/models.js');
  await VitalsModel.createCollection().catch(() => undefined);

  // ── HTTP + realtime ─────────────────────────────────────────────────────
  const keys = loadKeys(env);
  const inference = new InferenceClient(env, logger);

  const apiRouter = Router();
  const app = createApp({
    env,
    logger,
    apiRouter,
    readinessChecks: {
      mongo: () => Promise.resolve(mongoReady()),
      inference: () => inference.healthy(),
    },
  });

  const http = createHttpServer(app);
  const io = createSocketServer(http, env, keys, logger);
  const bus = new SocketBus(io);

  const imageUrlFor = (imageId: string) =>
    `${publicApiBase(env)}${API_PREFIX}/images/${imageId}/file`;

  const predictions = new InProcessPredictionService({
    env,
    logger,
    inference,
    bus,
    imageUrlFor,
  });
  const simulation = new SimulationService(bus, predictions, logger);

  // Seeding needs the prediction service, so it runs after wiring: the demo
  // database is populated and then scored through the real pipeline.
  if (env.DEMO_MODE) {
    await seedDemoData(env, logger);
    await backfillPredictions(predictions, logger);
  }

  // Auth routes are public by necessity; every route mounted after the
  // `requireAuth` guard needs a session.
  apiRouter.use('/auth', authRoutes(env, keys));
  apiRouter.use(requireAuth(keys, env));
  apiRouter.use(clinicalRoutes(env, imageUrlFor));
  apiRouter.use(predictionRoutes(predictions));
  apiRouter.use(alertRoutes(bus));
  apiRouter.use(dashboardRoutes());
  apiRouter.use(simulationRoutes(simulation));

  const server = http.listen(env.API_PORT, () => {
    logger.info({ port: env.API_PORT, env: env.NODE_ENV, demo: env.DEMO_MODE }, 'api listening');
    if (env.DEMO_MODE) {
      logger.info(`demo ready — open ${env.WEB_ORIGIN}`);
    }
  });

  // ── Shutdown ────────────────────────────────────────────────────────────
  let shuttingDown = false;
  const shutdown = (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    logger.info({ signal }, 'shutting down');

    simulation.stopAll();
    void io.close();
    server.close(() => {
      void (async () => {
        await disconnectMongo();
        await stopMemoryServer?.();
        process.exit(0);
      })();
    });

    // Don't hang forever on a stuck connection.
    setTimeout(() => process.exit(1), 10_000).unref();
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

/**
 * The base URL the *inference service* should use to fetch an image. In demo
 * both processes are on localhost; in the Docker stack this becomes the
 * internal service name, and later a presigned MinIO URL instead.
 */
function publicApiBase(env: Env): string {
  return `http://127.0.0.1:${env.API_PORT}`;
}

main().catch((err: unknown) => {
  // The logger may not exist yet, so this is the one place console is right.
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
