import { createPrivateKey, createPublicKey, type KeyObject } from 'node:crypto';

import type { Role } from '@pneumovision/shared';
import { SignJWT, jwtVerify } from 'jose';

import type { Env } from '../config/env.js';

/**
 * Access tokens: EdDSA (Ed25519), short-lived, delivered as an httpOnly cookie.
 *
 * TODO(phase-3): rotating refresh tokens with reuse detection (revoke the
 * whole family), plus the `refreshTokens` collection. The demo issues a single
 * access token and asks the user to log in again when it expires.
 */

const ALG = 'EdDSA';

export interface AccessTokenClaims {
  sub: string;
  email: string;
  name: string;
  role: Role;
}

function decodeKey(b64: string): string {
  return Buffer.from(b64, 'base64').toString('utf8');
}

export interface TokenKeys {
  privateKey: KeyObject;
  publicKey: KeyObject;
}

export function loadKeys(env: Env): TokenKeys {
  try {
    return {
      privateKey: createPrivateKey(decodeKey(env.JWT_PRIVATE_KEY_B64)),
      publicKey: createPublicKey(decodeKey(env.JWT_PUBLIC_KEY_B64)),
    };
  } catch (cause) {
    throw new Error(
      'JWT_PRIVATE_KEY_B64 / JWT_PUBLIC_KEY_B64 are not a base64-encoded Ed25519 PEM pair. ' +
        'Generate them with: npm run gen:secrets',
      { cause },
    );
  }
}

export async function signAccessToken(
  claims: AccessTokenClaims,
  keys: TokenKeys,
  env: Env,
): Promise<string> {
  return new SignJWT({ email: claims.email, name: claims.name, role: claims.role })
    .setProtectedHeader({ alg: ALG })
    .setSubject(claims.sub)
    .setIssuer(env.JWT_ISSUER)
    .setAudience(env.JWT_AUDIENCE)
    .setIssuedAt()
    .setExpirationTime(env.ACCESS_TOKEN_TTL)
    .sign(keys.privateKey);
}

export async function verifyAccessToken(
  token: string,
  keys: TokenKeys,
  env: Env,
): Promise<AccessTokenClaims> {
  const { payload } = await jwtVerify(token, keys.publicKey, {
    issuer: env.JWT_ISSUER,
    audience: env.JWT_AUDIENCE,
    algorithms: [ALG],
  });

  if (!payload.sub || typeof payload.role !== 'string') {
    throw new Error('token is missing required claims');
  }

  return {
    sub: payload.sub,
    email: typeof payload.email === 'string' ? payload.email : '',
    name: typeof payload.name === 'string' ? payload.name : '',
    role: payload.role as Role,
  };
}

export const ACCESS_COOKIE = 'pv_access';

/**
 * Cookie flags. `Secure` is omitted in the demo because it runs over plain
 * http://localhost, where a Secure cookie would simply never be stored.
 * SameSite=Lax rather than Strict so the cookie survives a top-level
 * navigation back into the app.
 *
 * TODO(phase-10): Secure + SameSite=Strict + a double-submit CSRF token once
 * the stack is behind TLS at Nginx.
 */
export function accessCookieOptions(env: Env, maxAgeMs: number) {
  return {
    httpOnly: true,
    sameSite: 'lax' as const,
    secure: !env.DEMO_MODE,
    path: '/',
    maxAge: maxAgeMs,
  };
}
