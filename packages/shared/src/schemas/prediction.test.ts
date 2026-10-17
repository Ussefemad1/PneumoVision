import { describe, expect, it } from 'vitest';

import { predictionResult } from './prediction.js';

const base = {
  probability: 0.41,
  unimodal: { ehr: 0.3, cxr: null, rr: 0.5, dn: null },
  joint: { high: 0.45, low: 0.35 },
  missingness: { vector: { ehr: true, cxr: false, rr: true, dn: false }, prediction: 0.2 },
  alphas: { high: 0.3, low: 0.2, miss: 0.1, ehr: 0.2, cxr: null, rr: 0.2, dn: null },
  confidence: { theta: 0.75, fractionAbove: { ehr: null, cxr: null, rr: null, dn: null } },
  received: { ehr: null, cxr: null, notes: [], mode: 'model' },
  sources: { fusion: 'trained' },
};

describe('predictionResult.fusionModel', () => {
  it('keeps the fusion model a fusion-mode result carries', () => {
    const fusionModel = {
      runId: 'r3-fusion-001',
      sha256: 'a241114adabb',
      head: 'late',
      heads: { high: 0.45, low: 0.35, late: 0.41, combined: 0.4, less_combined: 0.43 },
      bertModelName: 'dmis-lab/biobert-v1.1',
      theta: 0.75,
      modalities: ['EHR', 'CXR', 'RR'],
      calibrated: false,
      notes: ['uncalibrated'],
    };
    expect(predictionResult.parse({ ...base, fusionModel }).fusionModel).toEqual(fusionModel);
  });

  it('is null for mock and readers results', () => {
    expect(predictionResult.parse(base).fusionModel).toBeNull();
  });

  it('rejects an unknown head', () => {
    const bad = { runId: 'x', sha256: 'y', head: 'best', heads: {}, bertModelName: 'b', theta: 0.75,
      modalities: [], calibrated: false, notes: [] }; // prettier-ignore
    expect(() => predictionResult.parse({ ...base, fusionModel: bad })).toThrow();
  });
});
