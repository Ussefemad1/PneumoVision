import { createServer as createHttpServer } from 'node:http';

import { Router } from 'express';

import { API_PREFIX, createApp } from './app.js';
import { loadEnv, usesMemoryMongo } from './config/env.js';
import { MONGO_BINARY_VERSION, MONGO_DOWNLOAD_DIR } from './config/mongoBinary.js';
import { connectMongo, disconnectMongo, mongoReady, syncIndexes } from './db/connect.js';
import './db/models.js';
import { backfillPredictions } from './demo/backfill.js';
import { seedDemoData } from './demo/seed.js';
import { InferenceClient } from './lib/inferenceClient.js';
import { createLogger } from './lib/logger.js';
import { loadKeys } from './lib/tokens.js';
import { alertRoutes } from './modules/alerts/routes.js';
import { analyzeRoutes } from './modules/analyze/routes.js';
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

async function main(): Promise<void> {
  const env = loadEnv();
  const logger = createLogger({
    level: env.LOG_LEVEL,
    isProduction: env.NODE_ENV === 'production',
  });

  let stopMemoryServer: (() => Promise<void>) | undefined;
  let mongoUri = env.MONGO_URI;

  if (usesMemoryMongo(env)) {
    const { MongoMemoryServer } = await import('mongodb-memory-server');
    const cacheDir = process.env.MONGOMS_DOWNLOAD_DIR;
    if (cacheDir)
      logger.info({ version: MONGO_BINARY_VERSION, cacheDir }, 'using pre-cached mongod');
    else
      logger.warn(
        { version: MONGO_BINARY_VERSION, expected: MONGO_DOWNLOAD_DIR },
        'MONGOMS_DOWNLOAD_DIR is unset — mongod may be downloaded now (~600 MB)',
      );
    logger.info('starting in-process MongoDB (demo mode)…');
    const server = await MongoMemoryServer.create({
      binary: { version: MONGO_BINARY_VERSION },
      instance: { dbName: env.MONGO_DB },
    });
    mongoUri = server.getUri(env.MONGO_DB);
    stopMemoryServer = async () => {
      await server.stop();
    };
  }

  await connectMongo(mongoUri, logger);
  await syncIndexes(logger);
  const { VitalsModel } = await import('./db/models.js');
  await VitalsModel.createCollection().catch(() => undefined);

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
  // Where the SPA fetches image pixels from (cookie-authenticated). The
  // inference service never uses this — it receives the bytes directly.
  const imageUrlFor = (imageId: string) => `${API_PREFIX}/images/${imageId}/file`;

  const predictions = new InProcessPredictionService({ env, logger, inference, bus });
  const simulation = new SimulationService(bus, predictions, logger);

  if (env.DEMO_MODE) {
    await seedDemoData(env, logger);
    await backfillPredictions(predictions, logger);
  }

  apiRouter.use('/auth', authRoutes(env, keys));
  apiRouter.use(requireAuth(keys, env));
  apiRouter.use(analyzeRoutes(env, predictions));
  apiRouter.use(clinicalRoutes(env, imageUrlFor));
  apiRouter.use(predictionRoutes(predictions));
  apiRouter.use(alertRoutes(bus));
  apiRouter.use(dashboardRoutes());
  apiRouter.use(simulationRoutes(simulation));

  const server = http.listen(env.API_PORT, () => {
    logger.info({ port: env.API_PORT, env: env.NODE_ENV, demo: env.DEMO_MODE }, 'api listening');
    if (env.DEMO_MODE) logger.info(`demo ready — open ${env.WEB_ORIGIN}`);
  });

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
    setTimeout(() => process.exit(1), 10_000).unref();
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
