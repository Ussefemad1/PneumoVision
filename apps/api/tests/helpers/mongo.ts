import { MongoMemoryServer } from 'mongodb-memory-server';
import mongoose from 'mongoose';

import { MONGO_BINARY_VERSION } from '../../src/config/mongoBinary.js';

/**
 * A real mongod, in-process.
 *
 * Pinned to 7.0.x deliberately: the vitals collection is a native time-series
 * collection, which needs MongoDB >= 5.0, and 7.x is what the Docker stack
 * runs. The first run downloads the binary (cached under node_modules
 * afterwards), so the boot hook allows a long timeout.
 */
export const MONGO_VERSION = MONGO_BINARY_VERSION;

let server: MongoMemoryServer | undefined;

export async function startMemoryMongo(): Promise<string> {
  server = await MongoMemoryServer.create({ binary: { version: MONGO_VERSION } });
  const uri = server.getUri();
  mongoose.set('strictQuery', true);
  await mongoose.connect(uri);
  await createTimeSeriesCollections();
  return uri;
}

export async function stopMemoryMongo(): Promise<void> {
  await mongoose.disconnect();
  await server?.stop();
  server = undefined;
}

/** Models whose schema declares time-series options. */
function timeSeriesModels() {
  return Object.values(mongoose.models).filter((m) => m.schema.get('timeseries'));
}

/**
 * Time-series collections only get their options if Mongo creates them
 * explicitly — an implicit create on first insert yields an ordinary
 * collection, which would silently pass the wrong shape.
 */
async function createTimeSeriesCollections(): Promise<void> {
  for (const model of timeSeriesModels()) {
    await model.createCollection();
  }
}

/**
 * Empties every collection between tests without restarting mongod.
 *
 * Time-series collections need dropping rather than emptying, and the
 * `system.*` collections Mongo maintains alongside them (`system.views`,
 * `system.buckets.*`) must be left alone — dropping `system.views` while a
 * time-series collection exists is an error.
 */
export async function clearCollections(): Promise<void> {
  const { db } = mongoose.connection;
  if (!db) return;

  const infos = await db.listCollections().toArray();
  for (const info of infos) {
    if (info.name.startsWith('system.')) continue;
    if (info.type === 'timeseries') {
      await db.dropCollection(info.name);
    } else {
      await db.collection(info.name).deleteMany({});
    }
  }

  // Recreate what we dropped, with the time-series options intact.
  await createTimeSeriesCollections();
}
