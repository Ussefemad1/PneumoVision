import express, { type Express } from 'express';
import helmet from 'helmet';
import { pinoHttp } from 'pino-http';

import type { Env } from './config/env.js';
import type { Logger } from './lib/logger.js';
import { createErrorHandler, notFoundHandler } from './middleware/errorHandler.js';
import { requestId } from './middleware/requestId.js';

export interface AppDeps {
  env: Env;
  logger: Logger;
  /** Liveness/readiness probes for the backing services. Phase 1 has none
   *  wired yet; each is added as its client lands in a later phase. */
  readinessChecks?: Record<string, () => Promise<boolean>>;
}

export const API_PREFIX = '/api/v1';

export function createApp({ env, logger, readinessChecks = {} }: AppDeps): Express {
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
      crossOriginResourcePolicy: { policy: 'same-site' },
      referrerPolicy: { policy: 'no-referrer' },
    }),
  );

  app.use(express.json({ limit: '1mb' }));

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

  app.use(notFoundHandler);
  app.use(createErrorHandler(logger));

  return app;
}
