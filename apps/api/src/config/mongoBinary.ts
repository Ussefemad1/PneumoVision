/**
 * The mongod build that `mongodb-memory-server` runs in DEMO_MODE.
 *
 * Single source of truth, deliberately. Three places need to agree on it:
 * the server that starts the in-process database, the test helper, and the
 * Railway image, which pre-downloads the binary at build time so the
 * container does not pull ~600 MB on every cold start. If any of them asked
 * for a different build, the download would silently happen at runtime
 * anyway and the pre-cache would be dead weight — so they all import this.
 *
 * 7.0.x specifically: the vitals collection is a native time-series
 * collection (MongoDB >= 5.0), and 7.x is what the Docker Compose stack runs.
 */
export const MONGO_BINARY_VERSION = '7.0.14';

/**
 * Where the pre-downloaded binary lives inside a built image.
 *
 * `mongodb-memory-server` reads MONGOMS_DOWNLOAD_DIR from the environment;
 * this is only the default the Railway image sets, exported here so the
 * build step and the runtime stage cannot disagree about the path.
 */
export const MONGO_DOWNLOAD_DIR = '/opt/mongodb-binaries';
