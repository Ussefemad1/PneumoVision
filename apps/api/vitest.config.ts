import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'tests/**/*.test.ts'],
    // Model and route tests boot a real mongod in-process; the first run also
    // downloads the pinned binary.
    testTimeout: 60_000,
    hookTimeout: 180_000,
    // A single mongod instance per file, and files must not share a database.
    fileParallelism: false,
  },
});
