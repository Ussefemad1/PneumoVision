import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createUser, deactivateUser, resetPassword } from '../src/cli/users.js';
import { loadEnv } from '../src/config/env.js';
import { UserModel } from '../src/db/models.js';
import { seedDemoData } from '../src/demo/seed.js';
import { createLogger } from '../src/lib/logger.js';
import { verifyPassword } from '../src/lib/password.js';
import { clearCollections, startMemoryMongo, stopMemoryMongo } from './helpers/mongo.js';

/** The full-stack (DEMO_MODE=false) profile as infra/aws/bootstrap.sh writes it. */
const FULL_STACK = {
  DEMO_MODE: 'false',
  NODE_ENV: 'test',
  LOG_LEVEL: 'silent',
  WEB_ORIGIN: 'https://203-0-113-7.sslip.io',
  JWT_PRIVATE_KEY_B64: 'x',
  JWT_PUBLIC_KEY_B64: 'x',
  INFERENCE_URL: 'http://inference:8000',
  INFERENCE_HMAC_SECRET: 'c'.repeat(32),
  MONGO_URI: 'mongodb://u:p@mongo:27017/pneumovision?authSource=admin',
  REDIS_URL: 'redis://:p@redis:6379',
  S3_ENDPOINT: 'http://minio:9000',
  S3_BUCKET: 'pneumovision',
  S3_ACCESS_KEY: 'k',
  S3_SECRET_KEY: 's',
  FIELD_ENCRYPTION_KEY_HEX: 'ab'.repeat(32),
  CSRF_SECRET: 'd'.repeat(32),
};

describe('full-stack env', () => {
  it('needs an admin password to seed', () => {
    expect(() => loadEnv({ ...FULL_STACK, SEED_DEMO_DATA: 'true' })).toThrow(/SEED_ADMIN_PASSWORD/);
  });

  it('seeds without any clinician password', () => {
    const env = loadEnv({
      ...FULL_STACK,
      SEED_DEMO_DATA: 'true',
      SEED_ADMIN_PASSWORD: 'long-random-admin-pw',
    });
    expect(env.SEED_DEMO_DATA).toBe(true);
    expect(env.SEED_CLINICIAN_PASSWORD).toBe('');
  });

  it('does not seed by default', () => {
    expect(loadEnv(FULL_STACK).SEED_DEMO_DATA).toBe(false);
  });
});

describe('production seed and user CLI', () => {
  let dataDir: string;

  beforeAll(async () => {
    await startMemoryMongo();
    dataDir = mkdtempSync(join(tmpdir(), 'pv-seed-'));
  }, 180_000);

  afterAll(async () => {
    await clearCollections();
    await stopMemoryMongo();
    rmSync(dataDir, { recursive: true, force: true });
  });

  it('creates the bootstrap admin only, no account with a known password', async () => {
    const env = loadEnv({
      ...FULL_STACK,
      SEED_DEMO_DATA: 'true',
      SEED_ADMIN_PASSWORD: 'long-random-admin-pw',
      DEMO_DATA_DIR: relative(process.cwd(), dataDir),
    });
    const result = await seedDemoData(env, createLogger({ level: 'silent', isProduction: true }));
    expect(result.users).toBe(1);
    expect(result.stays).toBeGreaterThan(0);

    const users = await UserModel.find().select('+passwordHash').lean();
    expect(users.map((u) => u.role)).toEqual(['admin']);
    expect(await verifyPassword(users[0]!.passwordHash, 'long-random-admin-pw')).toBe(true);

    // Idempotent: a restart seeds nothing more.
    const again = await seedDemoData(env, createLogger({ level: 'silent', isProduction: true }));
    expect(again.users).toBe(0);
    expect(await UserModel.countDocuments()).toBe(1);
  }, 120_000);

  it('creates a user with a generated password, then resets and deactivates it', async () => {
    const password = await createUser({
      email: 'Dr.A@Example.org',
      name: 'Dr A',
      role: 'clinician',
    });
    expect(password).toMatch(/^[A-Za-z0-9_-]{24}$/);
    const user = await UserModel.findOne({ email: 'dr.a@example.org' }).select('+passwordHash');
    expect(user?.role).toBe('clinician');
    expect(await verifyPassword(user!.passwordHash, password)).toBe(true);

    await expect(
      createUser({ email: 'dr.a@example.org', name: 'Dup', role: 'clinician' }),
    ).rejects.toThrow(/already exists/);
    await expect(createUser({ email: 'b@example.org', name: 'B', role: 'root' })).rejects.toThrow(
      /role must be one of/,
    );

    const fresh = await resetPassword('dr.a@example.org');
    expect(fresh).not.toBe(password);
    const reset = await UserModel.findOne({ email: 'dr.a@example.org' }).select('+passwordHash');
    expect(await verifyPassword(reset!.passwordHash, fresh)).toBe(true);
    expect(await verifyPassword(reset!.passwordHash, password)).toBe(false);

    await deactivateUser('dr.a@example.org');
    expect((await UserModel.findOne({ email: 'dr.a@example.org' }))?.isActive).toBe(false);
    await expect(resetPassword('nobody@example.org')).rejects.toThrow(/no user/);
  });
});
