import { TEXT_CHUNK_TOKENS } from './constants.js';

/**
 * WordPiece-shaped token estimate for a clinical note.
 *
 * Bio_ClinicalBERT's tokenizer is not available in the browser (or in the
 * mock inference service), so both sides use the same cheap rule: every
 * punctuation character is one token, and every alphanumeric run is one token
 * per six characters, since long clinical words split into several pieces.
 *
 * Mirrored in `services/inference/app/text.py`; both are tested against the
 * same vectors, so change them together.
 */
const PIECE = /[A-Za-z0-9]+|[^\sA-Za-z0-9]/g;
const ALNUM_START = /^[A-Za-z0-9]/;

export function estimateTokens(text: string): number {
  let total = 0;
  for (const [piece] of text.matchAll(PIECE)) {
    total += ALNUM_START.test(piece) ? Math.max(1, Math.ceil(piece.length / 6)) : 1;
  }
  return total;
}

/** How many 512-token windows the text encoder will split a note into. */
export function chunkCount(tokens: number): number {
  return tokens > 0 ? Math.ceil(tokens / TEXT_CHUNK_TOKENS) : 0;
}
