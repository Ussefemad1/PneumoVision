/**
 * Regenerates the synthetic chest X-ray fixture used by the Analyze e2e tests.
 *
 * The image is drawn procedurally by the API's demo generator — it is not,
 * and is not derived from, a MIMIC-CXR image. Run from the repo root:
 *
 *   npx tsx apps/web/e2e/fixtures/generate.ts
 *
 * then update SYNTHETIC_CXR_SHA256 in analyze.spec.ts with the printed hash.
 */
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { generateSyntheticXray } from '../../../api/src/demo/syntheticXray.js';

const xray = generateSyntheticXray({ seed: 'e2e-analyze-fixture', size: 96, opacity: 0.6 });
writeFileSync(join(dirname(fileURLToPath(import.meta.url)), 'synthetic-cxr.png'), xray.png);
process.stdout.write(`${xray.width}x${xray.height} sha256=${xray.sha256}\n`);
