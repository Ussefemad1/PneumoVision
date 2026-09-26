import { z } from 'zod';

/**
 * Environment validation. The process crashes at boot if anything required is
 * missing or malformed (Section 7: "all secrets from env, validated at boot").
 * Nothing in the app reads process.env directly — import `env` instead.
 *
 * Two profiles:
 *
 * - **DEMO_MODE=true** — the single-command `npm run dev:demo` slice. Needs
 *   only what the demo actually uses: HTTP config, JWT keys, the inference
 *   service, and the seed credentials. `MONGO_URI` may be empty, in which case
 *   an in-process mongodb-memory-server is started.
 * - **DEMO_MODE=false** — the full stack. Redis, S3/MinIO, field encryption
 *   and CSRF secrets all become mandatory, because the code paths that need
 *   them are live.
 */

const bool = (dflt: boolean) =>
  z
    .enum(['true', 'false'])
    .default(String(dflt) as 'true' | 'false')
    .transform((v) => v === 'true');

const hex = (bytes: number) =>
  z.string().regex(new RegExp(`^[0-9a-fA-F]{${bytes * 2}}$`), `must be ${bytes} bytes as hex`);

const baseSchema = z.object({
  // ── Always required ───────────────────────────────────────────────────────
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  // 'silent' is a real pino level and is what the test suite runs at.
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent']).default('info'),
  API_PORT: z.coerce.number().int().positive().default(4000),
  WEB_ORIGIN: z.string().url(),

  DEMO_MODE: bool(false),

  // Auth. EdDSA (Ed25519) in both profiles, so phase 3 changes nothing here.
  ACCESS_TOKEN_TTL: z.string().default('15m'),
  REFRESH_TOKEN_TTL: z.string().default('7d'),
  JWT_PRIVATE_KEY_B64: z.string().min(1),
  JWT_PUBLIC_KEY_B64: z.string().min(1),
  JWT_ISSUER: z.string().default('pneumovision'),
  JWT_AUDIENCE: z.string().default('pneumovision-web'),

  // Inference service.
  INFERENCE_URL: z.string().url(),
  INFERENCE_HMAC_SECRET: z.string().min(16),
  INFERENCE_HMAC_MAX_SKEW_SECONDS: z.coerce.number().int().positive().default(60),
  INFERENCE_TIMEOUT_MS: z.coerce.number().int().positive().default(120_000),
  MOCK_MODE: bool(true),
  DEFAULT_THETA: z.coerce.number().gt(0).lt(1).default(0.75),

  // ── Database ──────────────────────────────────────────────────────────────
  // Empty in demo means "start mongodb-memory-server".
  MONGO_URI: z.string().default(''),
  MONGO_DB: z.string().default('pneumovision'),

  // ── Seed credentials (demo + any seeded environment) ──────────────────────
  SEED_ADMIN_EMAIL: z.string().email().default('admin@pneumovision.local'),
  SEED_ADMIN_PASSWORD: z.string().min(8).or(z.literal('')).default(''),
  SEED_CLINICIAN_EMAIL: z.string().email().default('clinician@pneumovision.local'),
  SEED_CLINICIAN_PASSWORD: z.string().min(8).or(z.literal('')).default(''),

  /** Where demo artefacts (synthetic X-ray PNGs) are written. */
  DEMO_DATA_DIR: z.string().default('.demo-data'),

  /**
   * Absolute path to a built SPA to serve from this process.
   *
   * Unset in local development and in the Docker Compose stack, where Nginx
   * serves the static build and proxies the API. Set in single-service
   * deployments (Railway), where one container has to answer both.
   */
  WEB_DIST_DIR: z.string().optional(),

  // ── Required only when DEMO_MODE=false ────────────────────────────────────
  REDIS_URL: z.string().optional(),
  S3_ENDPOINT: z.string().url().optional(),
  S3_REGION: z.string().default('us-east-1'),
  S3_BUCKET: z.string().optional(),
  S3_ACCESS_KEY: z.string().optional(),
  S3_SECRET_KEY: z.string().optional(),
  S3_FORCE_PATH_STYLE: bool(true),
  S3_PRESIGN_TTL_SECONDS: z.coerce.number().int().positive().max(900).default(300),
  FIELD_ENCRYPTION_KEY_HEX: hex(32).optional(),
  FIELD_ENCRYPTION_KEY_VERSION: z.coerce.number().int().positive().default(1),
  CSRF_SECRET: z.string().min(16).optional(),
  CHECKPOINT_DIR: z.string().optional(),
});

/** Vars the full stack cannot run without, but the demo never touches. */
const FULL_STACK_REQUIRED = [
  'REDIS_URL',
  'S3_ENDPOINT',
  'S3_BUCKET',
  'S3_ACCESS_KEY',
  'S3_SECRET_KEY',
  'FIELD_ENCRYPTION_KEY_HEX',
  'CSRF_SECRET',
] as const;

const envSchema = baseSchema.superRefine((value, ctx) => {
  if (value.DEMO_MODE) {
    // The demo seeds two accounts, so their passwords must be real.
    for (const key of ['SEED_ADMIN_PASSWORD', 'SEED_CLINICIAN_PASSWORD'] as const) {
      if (!value[key]) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: [key],
          message: 'required in DEMO_MODE (the seeded login would have no password)',
        });
      }
    }
    return;
  }

  for (const key of FULL_STACK_REQUIRED) {
    if (!value[key]) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [key],
        message: 'required when DEMO_MODE=false',
      });
    }
  }

  if (!value.MONGO_URI) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['MONGO_URI'],
      message: 'required when DEMO_MODE=false (in-memory Mongo is demo-only)',
    });
  }
});

export type Env = z.infer<typeof envSchema>;

export function loadEnv(source: NodeJS.ProcessEnv = process.env): Env {
  const parsed = envSchema.safeParse(source);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `  ${i.path.join('.') || '(root)'}: ${i.message}`)
      .join('\n');
    // Deliberately not the logger: the logger itself depends on config.
    throw new Error(`Invalid environment configuration:\n${issues}`);
  }
  return parsed.data;
}

/** True when the API should start its own in-process MongoDB. */
export function usesMemoryMongo(env: Env): boolean {
  return env.DEMO_MODE && env.MONGO_URI === '';
}
