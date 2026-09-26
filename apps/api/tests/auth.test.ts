import { generateKeyPairSync } from 'node:crypto';

import { Router } from 'express';
import request from 'supertest';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createApp } from '../src/app.js';
import { loadEnv } from '../src/config/env.js';
import { UserModel } from '../src/db/models.js';
import { hashPassword } from '../src/lib/password.js';
import { createLogger } from '../src/lib/logger.js';
import { loadKeys } from '../src/lib/tokens.js';
import { authRoutes } from '../src/modules/auth/routes.js';
import { dashboardRoutes } from '../src/modules/dashboard/routes.js';
import { requireAuth } from '../src/middleware/auth.js';
import { clearCollections, startMemoryMongo, stopMemoryMongo } from './helpers/mongo.js';

/** A throwaway Ed25519 keypair, generated once for the suite. */
function testKeys() {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  return {
    priv: Buffer.from(privateKey.export({ type: 'pkcs8', format: 'pem' }) as string).toString(
      'base64',
    ),
    pub: Buffer.from(publicKey.export({ type: 'spki', format: 'pem' }) as string).toString(
      'base64',
    ),
  };
}

const PASSWORD = 'correct-horse-battery';
let app: ReturnType<typeof createApp>;

beforeAll(async () => {
  await startMemoryMongo();

  const keyPair = testKeys();
  const env = loadEnv({
    DEMO_MODE: 'true',
    NODE_ENV: 'test',
    LOG_LEVEL: 'silent',
    WEB_ORIGIN: 'http://localhost:5173',
    JWT_PRIVATE_KEY_B64: keyPair.priv,
    JWT_PUBLIC_KEY_B64: keyPair.pub,
    INFERENCE_URL: 'http://127.0.0.1:8000',
    INFERENCE_HMAC_SECRET: 'c'.repeat(32),
    SEED_ADMIN_PASSWORD: PASSWORD,
    SEED_CLINICIAN_PASSWORD: PASSWORD,
  });

  const logger = createLogger({ level: 'silent', isProduction: true });
  const keys = loadKeys(env);

  const apiRouter = Router();
  apiRouter.use('/auth', authRoutes(env, keys));
  apiRouter.use(requireAuth(keys, env));
  apiRouter.use(dashboardRoutes());

  app = createApp({ env, logger, apiRouter });

  await UserModel.create({
    email: 'clinician@pneumovision.local',
    passwordHash: await hashPassword(PASSWORD),
    name: 'Dr Demo',
    role: 'clinician',
  });
}, 180_000);

afterAll(async () => {
  await clearCollections();
  await stopMemoryMongo();
});

const login = (email: string, password: string) =>
  request(app).post('/api/v1/auth/login').send({ email, password });

describe('POST /auth/login', () => {
  it('signs in a valid user and sets an httpOnly cookie', async () => {
    const res = await login('clinician@pneumovision.local', PASSWORD);

    expect(res.status).toBe(200);
    expect(res.body.data.role).toBe('clinician');

    const cookie = res.headers['set-cookie']?.[0] ?? '';
    expect(cookie).toContain('pv_access=');
    expect(cookie).toContain('HttpOnly');
  });

  it('gives a wrong password and an unknown email the same response', async () => {
    const wrongPassword = await login('clinician@pneumovision.local', 'not-the-password');
    const unknownEmail = await login('nobody@pneumovision.local', PASSWORD);

    expect(wrongPassword.status).toBe(401);
    expect(unknownEmail.status).toBe(401);
    // Identical wording, so the response cannot be used to enumerate accounts.
    expect(wrongPassword.body.error.message).toBe(unknownEmail.body.error.message);
    expect(wrongPassword.body.error.code).toBe(unknownEmail.body.error.code);
  });

  it('never returns the password hash', async () => {
    const res = await login('clinician@pneumovision.local', PASSWORD);
    expect(JSON.stringify(res.body)).not.toContain('argon2');
  });
});

describe('RBAC', () => {
  it('rejects /dashboard/ward without a cookie', async () => {
    const res = await request(app).get('/api/v1/dashboard/ward');
    expect(res.status).toBe(401);
    expect(res.body.error.code).toBe('UNAUTHENTICATED');
  });

  it('allows /dashboard/ward with a valid session cookie', async () => {
    const signedIn = await login('clinician@pneumovision.local', PASSWORD);
    const cookie = signedIn.headers['set-cookie'];

    const res = await request(app).get('/api/v1/dashboard/ward').set('Cookie', cookie);
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body.data.rows)).toBe(true);
  });

  it('rejects a forged cookie', async () => {
    const res = await request(app)
      .get('/api/v1/dashboard/ward')
      .set('Cookie', ['pv_access=not.a.real.token']);
    expect(res.status).toBe(401);
  });
});
