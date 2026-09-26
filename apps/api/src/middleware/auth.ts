import type { CurrentUser, Role } from '@pneumovision/shared';
import type { NextFunction, Request, Response } from 'express';

import type { Env } from '../config/env.js';
import { ACCESS_COOKIE, verifyAccessToken, type TokenKeys } from '../lib/tokens.js';
import { ApiError } from './errorHandler.js';

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      /** Set by `requireAuth`. Absent on public routes. */
      user?: CurrentUser;
    }
  }
}

/**
 * Reads the access token from its httpOnly cookie and attaches the caller.
 *
 * TODO(phase-3): also check the user is still active and not locked out, and
 * emit an audit entry for every authenticated read of patient data.
 */
export function requireAuth(keys: TokenKeys, env: Env) {
  return function auth(req: Request, _res: Response, next: NextFunction): void {
    const cookies = req.cookies as Record<string, string | undefined> | undefined;
    const token = cookies?.[ACCESS_COOKIE];
    if (!token) {
      next(ApiError.unauthorized('Not signed in'));
      return;
    }

    verifyAccessToken(token, keys, env)
      .then((claims) => {
        req.user = {
          id: claims.sub,
          email: claims.email,
          name: claims.name,
          role: claims.role,
        };
        next();
      })
      .catch(() => next(ApiError.unauthorized('Session expired or invalid')));
  };
}

/**
 * Role gate. Deny by default: a route without `requireRole` is reachable by
 * any authenticated user, so every patient-data route names its roles.
 */
export function requireRole(...allowed: Role[]) {
  return function rbac(req: Request, _res: Response, next: NextFunction): void {
    if (!req.user) {
      next(ApiError.unauthorized());
      return;
    }
    if (!allowed.includes(req.user.role)) {
      next(ApiError.forbidden(`Requires one of: ${allowed.join(', ')}`));
      return;
    }
    next();
  };
}
