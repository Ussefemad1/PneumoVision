import { loginRequest } from '@pneumovision/shared';
import { Router } from 'express';

import type { Env } from '../../config/env.js';
import { UserModel } from '../../db/models.js';
import { asyncHandler, ok } from '../../lib/http.js';
import { verifyPassword } from '../../lib/password.js';
import {
  ACCESS_COOKIE,
  accessCookieOptions,
  signAccessToken,
  type TokenKeys,
} from '../../lib/tokens.js';
import { requireAuth } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';

/**
 * Demo authentication: email + password, access token in an httpOnly cookie.
 *
 * TODO(phase-3): TOTP MFA (mandatory for admin), rotating refresh tokens with
 * reuse detection, account lockout after 5 failures, and an audit entry for
 * every auth event. The shapes below are already the ones those will use.
 */
export function authRoutes(env: Env, keys: TokenKeys): Router {
  const router = Router();
  const auth = requireAuth(keys, env);

  router.post(
    '/login',
    asyncHandler(async (req, res) => {
      const { email, password } = loginRequest.parse(req.body);

      const user = await UserModel.findOne({ email: email.toLowerCase() }).select('+passwordHash');

      // Verify even when the user is missing, so a wrong address and a wrong
      // password take indistinguishable time.
      const stored =
        user?.passwordHash ??
        '$argon2id$v=19$m=19456,t=2,p=1$aaaaaaaaaaaaaaaa$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
      const valid = await verifyPassword(stored, password);

      if (!user || !valid || !user.isActive) {
        throw ApiError.unauthorized('Incorrect email or password');
      }

      const token = await signAccessToken(
        { sub: String(user._id), email: user.email, name: user.name, role: user.role },
        keys,
        env,
      );

      res.cookie(ACCESS_COOKIE, token, accessCookieOptions(env, 15 * 60 * 1000));
      ok(res, {
        id: String(user._id),
        email: user.email,
        name: user.name,
        role: user.role,
      });
    }),
  );

  router.post('/logout', (_req, res) => {
    res.clearCookie(ACCESS_COOKIE, { ...accessCookieOptions(env, 0), maxAge: undefined });
    ok(res, { ok: true });
  });

  router.get('/me', auth, (req, res) => {
    ok(res, req.user);
  });

  return router;
}
