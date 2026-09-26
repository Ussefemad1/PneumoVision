import mongoose from 'mongoose';

import type { Logger } from '../lib/logger.js';

/**
 * Mongo connection handling.
 *
 * In DEMO_MODE the caller passes the URI of an in-process mongodb-memory-server
 * instance; in every other mode it is a real deployment. Nothing below cares
 * which, so the demo exercises the same Mongoose models and indexes as the
 * eventual Docker stack.
 */
export async function connectMongo(uri: string, logger: Logger): Promise<typeof mongoose> {
  mongoose.set('strictQuery', true);
  // NOTE: `sanitizeFilter` is deliberately NOT enabled globally. It wraps any
  // value containing $-keys in $eq, which breaks the $in / $lt operators this
  // code writes intentionally. Operator injection is prevented at the
  // boundary instead: every filter here is built from zod-validated,
  // type-narrowed values, never spread from a request body, and
  // `assertNoOperators` (lib/http.ts) rejects $-keys in any payload that
  // would reach a query.

  const conn = await mongoose.connect(uri, {
    serverSelectionTimeoutMS: 10_000,
  });
  logger.info({ db: conn.connection.name }, 'mongo connected');
  return conn;
}

export async function disconnectMongo(): Promise<void> {
  await mongoose.disconnect();
}

/** Readiness probe for `GET /ready`. */
export function mongoReady(): boolean {
  return mongoose.connection.readyState === mongoose.ConnectionStates.connected;
}

/**
 * Builds every declared index. Mongoose autoIndex does this lazily per model;
 * calling it explicitly at boot means a bad index definition fails loudly at
 * startup instead of on the first query that needs it.
 */
export async function syncIndexes(logger: Logger): Promise<void> {
  const names: string[] = [];
  for (const model of Object.values(mongoose.models)) {
    // Time-series collections manage their own clustered index and reject
    // syncIndexes(); skip anything flagged as one.
    if (model.schema.get('timeseries')) continue;
    await model.syncIndexes();
    names.push(model.modelName);
  }
  logger.debug({ models: names }, 'indexes synced');
}
