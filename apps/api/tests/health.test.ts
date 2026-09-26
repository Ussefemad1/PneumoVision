import request from 'supertest';
import { describe, expect, it } from 'vitest';

import { createApp } from '../src/app.js';
import { loadEnv } from '../src/config/env.js';
import { createLogger } from '../src/lib/logger.js';

/** A complete, syntactically valid env so config validation passes in tests. */
const TEST_ENV = {
  NODE_ENV: 'test',
  LOG_LEVEL: 'silent',
  WEB_ORIGIN: 'https://localhost',
  JWT_PRIVATE_KEY_B64: 'dGVzdA==',
  JWT_PUBLIC_KEY_B64: 'dGVzdA==',
  FIELD_ENCRYPTION_KEY_HEX: 'a'.repeat(64),
  CSRF_SECRET: 'b'.repeat(32),
  MONGO_URI: 'mongodb://localhost:27017/test',
  REDIS_URL: 'redis://localhost:6379',
  S3_ENDPOINT: 'http://localhost:9000',
  S3_BUCKET: 'test',
  S3_ACCESS_KEY: 'test',
  S3_SECRET_KEY: 'test',
  INFERENCE_URL: 'http://localhost:8000',
  INFERENCE_HMAC_SECRET: 'c'.repeat(32),
} satisfies NodeJS.ProcessEnv;

function makeApp(readinessChecks?: Record<string, () => Promise<boolean>>) {
  const env = loadEnv({ ...TEST_ENV, LOG_LEVEL: 'error' });
  const logger = createLogger({ level: 'error', isProduction: true });
  return createApp(readinessChecks ? { env, logger, readinessChecks } : { env, logger });
}

describe('config validation', () => {
  it('rejects a missing required secret', () => {
    const { MONGO_URI: _omitted, ...withoutMongo } = TEST_ENV;
    expect(() => loadEnv(withoutMongo)).toThrow(/MONGO_URI/);
  });

  it('rejects a field encryption key that is not 32 bytes of hex', () => {
    expect(() => loadEnv({ ...TEST_ENV, FIELD_ENCRYPTION_KEY_HEX: 'abc' })).toThrow(/32 bytes/);
  });

  it('caps presigned URL lifetime at 15 minutes', () => {
    expect(() => loadEnv({ ...TEST_ENV, S3_PRESIGN_TTL_SECONDS: '3600' })).toThrow();
    expect(loadEnv(TEST_ENV).S3_PRESIGN_TTL_SECONDS).toBe(300);
  });
});

describe('GET /health', () => {
  it('reports liveness without touching dependencies', async () => {
    const res = await request(makeApp()).get('/health');
    expect(res.status).toBe(200);
    expect(res.body.data.status).toBe('ok');
  });

  it('sets security headers and a request id', async () => {
    const res = await request(makeApp()).get('/health');
    expect(res.headers['x-request-id']).toMatch(/^[0-9a-f-]{36}$/);
    expect(res.headers['content-security-policy']).toContain("default-src 'self'");
    expect(res.headers['x-powered-by']).toBeUndefined();
  });
});

describe('GET /ready', () => {
  it('is 200 when every dependency check passes', async () => {
    const app = makeApp({ mongo: () => Promise.resolve(true) });
    const res = await request(app).get('/ready');
    expect(res.status).toBe(200);
    expect(res.body.data).toEqual({ ready: true, checks: { mongo: true } });
  });

  it('is 503 when a dependency is down or throws', async () => {
    const app = makeApp({
      mongo: () => Promise.resolve(true),
      redis: () => Promise.reject(new Error('ECONNREFUSED')),
    });
    const res = await request(app).get('/ready');
    expect(res.status).toBe(503);
    expect(res.body.data.ready).toBe(false);
    expect(res.body.data.checks.redis).toBe(false);
  });
});

describe('error envelope', () => {
  it('returns { error: { code, message, requestId } } for unknown routes', async () => {
    const res = await request(makeApp()).get('/api/v1/nope');
    expect(res.status).toBe(404);
    expect(res.body.error.code).toBe('NOT_FOUND');
    expect(res.body.error.requestId).toBe(res.headers['x-request-id']);
  });
});
