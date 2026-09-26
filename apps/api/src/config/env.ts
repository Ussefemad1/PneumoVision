import { z } from 'zod';

/**
 * Environment validation. The process crashes at boot if anything required is
 * missing or malformed (Section 7: "all secrets from env, validated at boot").
 * Nothing in the app reads process.env directly — import `env` instead.
 */

const hex = (bytes: number) =>
  z.string().regex(new RegExp(`^[0-9a-fA-F]{${bytes * 2}}$`), `must be ${bytes} bytes as hex`);

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  // 'silent' is a real pino level and is what the test suite runs at.
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent']).default('info'),
  API_PORT: z.coerce.number().int().positive().default(4000),

  WEB_ORIGIN: z.string().url(),

  // Auth
  ACCESS_TOKEN_TTL: z.string().default('15m'),
  REFRESH_TOKEN_TTL: z.string().default('7d'),
  JWT_PRIVATE_KEY_B64: z.string().min(1),
  JWT_PUBLIC_KEY_B64: z.string().min(1),
  JWT_ISSUER: z.string().default('pneumovision'),
  JWT_AUDIENCE: z.string().default('pneumovision-web'),

  // Application-level field encryption (AES-256-GCM)
  FIELD_ENCRYPTION_KEY_HEX: hex(32),
  FIELD_ENCRYPTION_KEY_VERSION: z.coerce.number().int().positive().default(1),
  CSRF_SECRET: z.string().min(16),

  // Datastores
  MONGO_URI: z.string().min(1),
  MONGO_DB: z.string().default('pneumovision'),
  REDIS_URL: z.string().min(1),

  // Object storage
  S3_ENDPOINT: z.string().url(),
  S3_REGION: z.string().default('us-east-1'),
  S3_BUCKET: z.string().min(1),
  S3_ACCESS_KEY: z.string().min(1),
  S3_SECRET_KEY: z.string().min(1),
  S3_FORCE_PATH_STYLE: z
    .enum(['true', 'false'])
    .default('true')
    .transform((v) => v === 'true'),
  S3_PRESIGN_TTL_SECONDS: z.coerce.number().int().positive().max(900).default(300),

  // Inference service (internal only)
  INFERENCE_URL: z.string().url(),
  INFERENCE_HMAC_SECRET: z.string().min(16),
  INFERENCE_HMAC_MAX_SKEW_SECONDS: z.coerce.number().int().positive().default(60),
  INFERENCE_TIMEOUT_MS: z.coerce.number().int().positive().default(120_000),
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
