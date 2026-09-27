import { describe, expect, it } from 'vitest';

import { chunkCount, estimateTokens } from './text.js';

/**
 * Same vectors as services/inference/tests/test_service.py
 * `test_token_estimate_matches_the_typescript_vectors` — the Analyze page's
 * live counts must agree with what the service reports receiving.
 */
describe('estimateTokens / chunkCount', () => {
  it('matches the Python vectors', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('Bibasilar opacities, worse on the right.')).toBe(10);
    expect(estimateTokens('pneumothorax')).toBe(2);
    expect(estimateTokens('SpO2 88% on 4L')).toBe(5);
    expect(estimateTokens('word '.repeat(600))).toBe(600);
    expect(chunkCount(0)).toBe(0);
    expect(chunkCount(512)).toBe(1);
    expect(chunkCount(513)).toBe(2);
  });
});
