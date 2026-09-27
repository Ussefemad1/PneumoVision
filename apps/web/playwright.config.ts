import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig, devices } from '@playwright/test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

/**
 * The demo's seeded login lives in `.env.demo` (generated, never committed).
 * Read it so `npx playwright test` needs no extra environment; anything
 * already set in the environment wins.
 */
const envDemo = resolve(repoRoot, '.env.demo');
if (existsSync(envDemo)) {
  for (const line of readFileSync(envDemo, 'utf8').split(/\r?\n/)) {
    const match = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/.exec(line);
    if (match && process.env[match[1]!] === undefined) process.env[match[1]!] = match[2];
  }
}

const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'html' : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  /**
   * The real stack: mock inference + API (in-memory Mongo, seeded) + Vite.
   * Nothing of our own API is route-mocked. Ready once an API route answers
   * through the Vite proxy — 401 unauthenticated means both are up.
   */
  webServer: process.env.PLAYWRIGHT_BASE_URL
    ? undefined
    : {
        command: 'npm run dev:demo',
        cwd: repoRoot,
        url: `${baseURL}/api/v1/auth/me`,
        reuseExistingServer: !process.env.CI,
        timeout: 240_000,
        stdout: 'ignore',
        stderr: 'pipe',
      },
});
