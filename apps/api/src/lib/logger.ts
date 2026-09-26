import pino from 'pino';

/**
 * Structured logging with redaction (Section 7).
 *
 * Patient data must never reach the logs. The redaction list covers credentials
 * and the two encrypted fields — note text and demographics — plus anything
 * carrying a raw note body. When adding a route that accepts clinical content,
 * add its path here too.
 */
export const REDACTED_PATHS = [
  'req.headers.cookie',
  'req.headers.authorization',
  'req.headers["x-csrf-token"]',
  'req.headers["x-pv-signature"]',
  'res.headers["set-cookie"]',
  'password',
  'passwordHash',
  '*.password',
  '*.text',
  '*.textEnc',
  '*.demographicsEnc',
  '*.notes',
  '*.mfaSecret',
  '*.secretEnc',
  '*.token',
  '*.refreshToken',
];

export function createLogger(opts: { level: string; isProduction: boolean }) {
  return pino({
    level: opts.level,
    redact: { paths: REDACTED_PATHS, censor: '[redacted]' },
    // Pretty output is a dev-only convenience; production logs stay JSON.
    ...(opts.isProduction
      ? {}
      : { transport: { target: 'pino-pretty', options: { colorize: true } } }),
  });
}

export type Logger = ReturnType<typeof createLogger>;
