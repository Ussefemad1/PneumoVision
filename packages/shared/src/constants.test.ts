import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import {
  PHENOTYPE_CLASSES,
  PNEUMONIA_CLASS_INDEX,
  NOTE_TYPES_BY_TASK,
  CXR_PATCH_GRID,
  CXR_TOKEN_COUNT,
} from './constants.js';
import {
  EHR_VARIABLES,
  EHR_DISCRETIZED_COLUMNS,
  EHR_DISCRETIZED_WIDTH,
  EHR_CONTINUOUS_COLUMN_INDICES,
} from './generated/ehr-variables.js';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const read = (p: string) => readFileSync(resolve(repoRoot, p), 'utf8');

describe('EHR contract matches the training discretizer', () => {
  it('declares exactly the 17 variables from discretizer_config.json', () => {
    const cfg = JSON.parse(read('medpatch/ehr_utils/resources/discretizer_config.json')) as {
      id_to_channel: string[];
    };
    expect([...EHR_VARIABLES]).toEqual(cfg.id_to_channel);
    expect(EHR_VARIABLES).toHaveLength(17);
  });

  it('expands to the 76-wide tensor the LSTM was built for', () => {
    // medpatch/models/ehr_models.py: class LSTM(..., input_dim=76)
    expect(EHR_DISCRETIZED_WIDTH).toBe(76);
    expect(EHR_DISCRETIZED_COLUMNS).toHaveLength(76);
    expect(EHR_CONTINUOUS_COLUMN_INDICES).toHaveLength(12);
    expect(EHR_DISCRETIZED_COLUMNS.filter((c) => c.startsWith('mask->'))).toHaveLength(17);
  });

  it('keeps the LSTM input width literal in sync with the training default', () => {
    const src = read('medpatch/models/ehr_models.py');
    const match = /def __init__\(self, args, input_dim=(\d+)/.exec(src);
    expect(match?.[1]).toBe(String(EHR_DISCRETIZED_WIDTH));
  });
});

describe('phenotype labels match DataFusion.py', () => {
  it('has the same 25 classes in the same order', () => {
    const src = read('medpatch/datasets/DataFusion.py');
    const block = /^CLASSES = \[([\s\S]*?)^\s*\]/m.exec(src);
    expect(block).not.toBeNull();
    const parsed = [...(block?.[1] ?? '').matchAll(/'([^']+)'/g)].map((m) => m[1]);
    expect(parsed).toEqual([...PHENOTYPE_CLASSES]);
    expect(parsed).toHaveLength(25);
  });

  it('points PNEUMONIA_CLASS_INDEX at the pneumonia phenotype', () => {
    expect(PHENOTYPE_CLASSES[PNEUMONIA_CLASS_INDEX]).toMatch(/^Pneumonia \(except/);
  });
});

describe('F8 leakage control', () => {
  it('never allows discharge notes into a mortality prediction', () => {
    expect(NOTE_TYPES_BY_TASK.mortality).not.toContain('discharge');
  });

  it('matches the modality sets in the MedPatch training scripts', () => {
    // mortality trains on EHR-CXR-RR; phenotyping adds DN.
    expect(read('medpatch/scripts/mortality/MedPatch/Confidence-Patching.sh')).toContain(
      '--modalities EHR-CXR-RR',
    );
    expect(read('medpatch/scripts/phenotyping/MedPatch/Confidence-Patching.sh')).toContain(
      '--modalities EHR-CXR-RR-DN',
    );
  });
});

describe('CXR geometry', () => {
  it('derives a 24x24 patch grid from vit_small_patch16_384', () => {
    expect(CXR_PATCH_GRID).toBe(384 / 16);
    expect(CXR_TOKEN_COUNT).toBe(577);
  });
});
