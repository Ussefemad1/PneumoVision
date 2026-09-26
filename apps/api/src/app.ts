import cookieParser from 'cookie-parser';
import express, { type Express, type Router } from 'express';
import helmet from 'helmet';
import { pinoHttp } from 'pino-http';

import type { Env } from './config/env.js';
import type { Logger } from './lib/logger.js';
import { createErrorHandler, notFoundHandler } from './middleware/errorHandler.js';
import { requestId } from './middleware/requestId.js';

export interface AppDeps {
  env: Env;
  logger: Logger;
  /** Mounted under `/api/v1`. */
  apiRouter?: Router;
  /** Liveness/readiness probes for the backing services. */
  readinessChecks?: Record<string, () => Promise<boolean>>;
}

export const API_PREFIX = '/api/v1';

export function createApp({ env, logger, apiRouter, readinessChecks = {} }: AppDeps): Express {
  const app = express();

  // Nginx terminates TLS, so trust its X-Forwarded-* for client IPs and
  // Secure-cookie handling. One hop only.
  app.set('trust proxy', 1);
  app.disable('x-powered-by');

  app.use(requestId);
  app.use(
    pinoHttp({
      logger,
      genReqId: (req) => (req as express.Request).requestId,
      customProps: (req) => ({ requestId: (req as express.Request).requestId }),
      // Health probes every few seconds would otherwise dominate the log.
      autoLogging: { ignore: (req) => req.url === '/health' || req.url === '/ready' },
    }),
  );

  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"], // Tailwind injects inline styles
          imgSrc: ["'self'", 'data:', 'blob:'],
          connectSrc: ["'self'", env.WEB_ORIGIN],
          objectSrc: ["'none'"],
          frameAncestors: ["'none'"],
          baseUri: ["'self'"],
          formAction: ["'self'"],
        },
      },
      hsts: env.NODE_ENV === 'production' ? { maxAge: 31_536_000, includeSubDomains: true } : false,
      // The demo serves images cross-origin to the Vite dev server.
      crossOriginResourcePolicy: { policy: env.DEMO_MODE ? 'cross-origin' : 'same-site' },
      referrerPolicy: { policy: 'no-referrer' },
    }),
  );

  /**
   * CORS for the Vite dev server, which runs on a different port in DEMO_MODE.
   * Locked to the single configured origin and credentialed, so the auth
   * cookie is sent. In the Docker stack everything is same-origin behind
   * Nginx and this never applies.
   */
  if (env.DEMO_MODE) {
    app.use((req, res, next) => {
      const origin = req.headers.origin;
      if (origin === env.WEB_ORIGIN) {
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Access-Control-Allow-Credentials', 'true');
        res.setHeader('Access-Control-Allow-Headers', 'content-type,x-request-id');
        res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PATCH,DELETE,OPTIONS');
        res.setHeader('Vary', 'Origin');
      }
      if (req.method === 'OPTIONS') {
        res.sendStatus(204);
        return;
      }
      next();
    });
  }

  app.use(express.json({ limit: '1mb' }));
  app.use(cookieParser());

  // Liveness: the process is up. Never touches a dependency.
  app.get('/health', (_req, res) => {
    res.json({ data: { status: 'ok', uptimeSeconds: Math.floor(process.uptime()) } });
  });

  // Readiness: every backing service is reachable.
  app.get('/ready', (_req, res, next) => {
    void (async () => {
      try {
        const names = Object.keys(readinessChecks);
        const results = await Promise.all(
          names.map(async (name) => {
            try {
              return [name, await readinessChecks[name]!()] as const;
            } catch {
              return [name, false] as const;
            }
          }),
        );
        const checks = Object.fromEntries(results);
        const ready = results.every(([, ok]) => ok);
        res.status(ready ? 200 : 503).json({ data: { ready, checks } });
      } catch (err) {
        next(err);
      }
    })();
  });

  if (apiRouter) app.use(API_PREFIX, apiRouter);

  app.use(notFoundHandler);
  app.use(createErrorHandler(logger));

  return app;
}
