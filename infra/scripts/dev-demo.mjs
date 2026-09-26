/**
 * One-command demo launcher.
 *
 * Starts, in order:
 *   1. the FastAPI inference service (uvicorn, MOCK_MODE) on the repo venv
 *   2. the API gateway (which boots its own in-process MongoDB and seeds it)
 *   3. the Vite dev server
 *
 * No Docker, no Redis, no MinIO. Output from all three is prefixed and
 * colourised; Ctrl-C shuts everything down.
 *
 * Usage: npm run dev:demo
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const isWindows = process.platform === 'win32';

const COLOURS = { inference: '\x1b[35m', api: '\x1b[36m', web: '\x1b[32m', reset: '\x1b[0m' };

function fail(message, hint) {
  console.error(`\n\x1b[31m✖ ${message}\x1b[0m`);
  if (hint) console.error(`\n${hint}\n`);
  process.exit(1);
}

// ── Environment ─────────────────────────────────────────────────────────────

const envFile = join(root, '.env.demo');
if (!existsSync(envFile)) {
  fail(
    '.env.demo not found',
    'Create it with:\n\n  npm run gen:secrets -- --write\n\n' +
      'That writes .env.demo with a fresh Ed25519 keypair, an HMAC secret and\n' +
      'the two demo account passwords, and prints the logins.',
  );
}

/** Minimal .env parser — no dependency, handles KEY=value and # comments. */
function parseEnvFile(path) {
  const out = {};
  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq === -1) continue;
    out[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
  }
  return out;
}

const fileEnv = parseEnvFile(envFile);
const env = { ...process.env, ...fileEnv, DEMO_MODE: 'true', FORCE_COLOR: '1' };

for (const required of ['JWT_PRIVATE_KEY_B64', 'INFERENCE_HMAC_SECRET', 'SEED_ADMIN_PASSWORD']) {
  if (!env[required] || env[required].startsWith('replace-me')) {
    fail(
      `${required} is still a placeholder in .env.demo`,
      'Generate real values with:\n\n  npm run gen:secrets -- --demo\n',
    );
  }
}

// ── Python ──────────────────────────────────────────────────────────────────

const venvPython = isWindows
  ? join(root, '.venv', 'Scripts', 'python.exe')
  : join(root, '.venv', 'bin', 'python');

if (!existsSync(venvPython)) {
  fail(
    `Python venv not found at ${venvPython}`,
    'The inference service runs on the repo venv. Create it with:\n\n' +
      (isWindows
        ? '  py -3.11 -m venv .venv\n  .\\.venv\\Scripts\\Activate.ps1\n'
        : '  python3.11 -m venv .venv\n  source .venv/bin/activate\n') +
      '  python -m pip install -r services/inference/requirements.txt\n',
  );
}

// ── Process plumbing ────────────────────────────────────────────────────────

const children = [];
let shuttingDown = false;

function start(name, command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd ?? root,
    env: { ...env, ...options.env },
    stdio: ['ignore', 'pipe', 'pipe'],
    // Node >= 20 refuses to spawn a .cmd shim without a shell (the fix for
    // CVE-2024-27980), and npm on Windows *is* a .cmd. Every argument below
    // is a literal defined in this file, so there is nothing to inject.
    shell: options.shell ?? false,
  });

  const colour = COLOURS[name] ?? '';
  const label = `${colour}[${name}]${COLOURS.reset}`;
  for (const stream of [child.stdout, child.stderr]) {
    createInterface({ input: stream }).on('line', (line) => {
      if (line.trim()) console.log(`${label} ${line}`);
    });
  }

  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    console.log(`${label} exited (code ${code ?? 'null'}, signal ${signal ?? 'none'})`);
    // If any leg dies the demo is broken; take the rest down rather than
    // leaving a half-running stack that looks fine in the terminal.
    shutdown(code ?? 1);
  });

  children.push({ name, child });
  return child;
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('\nshutting down…');
  for (const { child } of children) {
    if (child.exitCode === null) {
      // SIGTERM is not really a thing on Windows; taskkill the tree instead.
      if (isWindows) spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
      else child.kill('SIGTERM');
    }
  }
  setTimeout(() => process.exit(code), 1500).unref();
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

/** Polls a URL until it answers or the deadline passes. */
async function waitFor(url, label, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (shuttingDown) return false;
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  fail(`${label} did not become ready within ${Math.round(timeoutMs / 1000)}s`);
  return false;
}

// ── Boot ────────────────────────────────────────────────────────────────────

const inferencePort = new URL(env.INFERENCE_URL ?? 'http://127.0.0.1:8000').port || '8000';

console.log('\n\x1b[1mPneumoVision demo\x1b[0m — starting inference, api and web…\n');

start('inference', venvPython, [
  '-m',
  'uvicorn',
  'app.main:app',
  '--host',
  '127.0.0.1',
  '--port',
  inferencePort,
  '--log-level',
  'warning',
], {
  cwd: join(root, 'services', 'inference'),
  env: { MOCK_MODE: 'true', PYTHONUNBUFFERED: '1' },
});

await waitFor(`http://127.0.0.1:${inferencePort}/v1/health`, 'inference service');
console.log(`${COLOURS.inference}[inference]${COLOURS.reset} ready (MOCK_MODE)\n`);

const npmCmd = isWindows ? 'npm.cmd' : 'npm';
const npmOptions = { shell: isWindows };

// The API boots mongodb-memory-server and seeds on first run, which takes a
// few seconds; the web dev server does not depend on it being ready.
start('api', npmCmd, ['run', 'dev', '--workspace', '@pneumovision/api'], npmOptions);
start('web', npmCmd, ['run', 'dev', '--workspace', '@pneumovision/web'], npmOptions);

const webOrigin = env.WEB_ORIGIN ?? 'http://localhost:5173';
await waitFor(`http://127.0.0.1:${env.API_PORT ?? 4000}/health`, 'api', 180_000);

console.log(
  [
    '',
    '\x1b[1m\x1b[32m  PneumoVision demo is up\x1b[0m',
    '',
    `  open      \x1b[4m${webOrigin}\x1b[0m`,
    `  sign in   ${env.SEED_CLINICIAN_EMAIL ?? 'clinician@pneumovision.local'}`,
    `            ${env.SEED_CLINICIAN_PASSWORD}`,
    '',
    '  Ctrl-C to stop everything.',
    '',
  ].join('\n'),
);
