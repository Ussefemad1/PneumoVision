import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Router } from 'express';
import request from 'supertest';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { createApp } from '../src/app.js';
import { loadEnv } from '../src/config/env.js';
import { createLogger } from '../src/lib/logger.js';

/**
 * Single-service mode (Railway): the API also serves the built SPA.
 *
 * The property that matters is the history fallback *not* swallowing API
 * routes. If it did, a mistyped endpoint would return the app shell with a
 * 200 and the client would try to parse HTML as JSON — a failure that looks
 * like a frontend bug and is miserable to trace back to routing.
 */

let webRoot: string;
let app: ReturnType<typeof createApp>;

const baseEnv = {
  DEMO_MODE: 'true',
  NODE_ENV: 'test',
  LOG_LEVEL: 'silent',
  WEB_ORIGIN: 'http://localhost:5173',
  JWT_PRIVATE_KEY_B64: 'dGVzdA==',
  JWT_PUBLIC_KEY_B64: 'dGVzdA==',
  INFERENCE_URL: 'http://127.0.0.1:8000',
  INFERENCE_HMAC_SECRET: 'c'.repeat(32),
  SEED_ADMIN_PASSWORD: 'demo-password',
  SEED_CLINICIAN_PASSWORD: 'demo-password',
};

beforeAll(() => {
  webRoot = mkdtempSync(join(tmpdir(), 'pv-web-'));
  mkdirSync(join(webRoot, 'assets'));
  writeFileSync(join(webRoot, 'index.html'), '<!doctype html><title>PneumoVision</title>');
  writeFileSync(join(webRoot, 'assets', 'index-abc123.js'), 'console.log(1);');

  const env = loadEnv({ ...baseEnv, WEB_DIST_DIR: webRoot });
  const logger = createLogger({ level: 'silent', isProduction: true });

  // An API router that answers one route, so we can prove the static layer
  // sits behind it rather than in front.
  const apiRouter = Router();
  apiRouter.get('/ping', (_req, res) => res.json({ data: 'pong' }));

  app = createApp({ env, logger, apiRouter });
});

afterAll(() => {
  // The temp dir is disposable; leaving it to the OS is fine.
});

describe('static SPA serving', () => {
  it('serves the app shell at the root', async () => {
    const res = await request(app).get('/');
    expect(res.status).toBe(200);
    expect(res.text).toContain('PneumoVision');
  });

  it('falls back to the shell for a client-side route', async () => {
    const res = await request(app).get('/stays/6ab8102532b36a8d77573b75');
    expect(res.status).toBe(200);
    expect(res.text).toContain('PneumoVision');
  });

  it('serves hashed assets as immutable', async () => {
    const res = await request(app).get('/assets/index-abc123.js');
    expect(res.status).toBe(200);
    expect(res.headers['cache-control']).toContain('immutable');
  });

  it('never caches the shell', async () => {
    const res = await request(app).get('/');
    expect(res.headers['cache-control']).toBe('no-store');
  });
});

describe('the fallback does not shadow the API', () => {
  it('still routes a real API endpoint', async () => {
    const res = await request(app).get('/api/v1/ping');
    expect(res.status).toBe(200);
    expect(res.body.data).toBe('pong');
  });

  it('returns the JSON error envelope for an unknown API route, not the shell', async () => {
    const res = await request(app).get('/api/v1/does-not-exist');
    expect(res.status).toBe(404);
    expect(res.body.error.code).toBe('NOT_FOUND');
    expect(res.text).not.toContain('<!doctype html>');
  });
});

describe('without WEB_DIST_DIR', () => {
  it('leaves non-API routes as a 404, so Nginx stays responsible for the SPA', async () => {
    const env = loadEnv(baseEnv);
    const logger = createLogger({ level: 'silent', isProduction: true });
    const bare = createApp({ env, logger, apiRouter: Router() });

    const res = await request(bare).get('/stays/abc');
    expect(res.status).toBe(404);
    expect(res.body.error.code).toBe('NOT_FOUND');
  });
});
