import { createHash } from 'node:crypto';

import {
  EHR_VARIABLES,
  EHR_WINDOW_HOURS,
  NOTE_TYPES_BY_TASK,
  type Modality,
  type NoteType,
  type Task,
} from '@pneumovision/shared';
import type { Types } from 'mongoose';

import { ImageModel, NoteModel, VitalsModel } from '../db/models.js';
import type { EhrPayload, NotePayload } from '../lib/inferenceClient.js';

/**
 * Assembles exactly what may be sent to the model for one prediction.
 *
 * **F8 (leakage control)** is enforced here and nowhere else:
 *
 *  1. Only data timestamped **strictly before** `cutoffTime` is included.
 *  2. Discharge notes are dropped for `mortality` regardless of timing —
 *     they describe the outcome being predicted. This matches the upstream
 *     training setup, where mortality trains on EHR-CXR-RR only.
 *
 * Both rules are unit-tested. Any future caller that bypasses this module
 * bypasses the leakage guarantee, so route handlers must not query the
 * vitals/notes/images collections for prediction input directly.
 */

export interface GatheredInputs {
  ehr: EhrPayload | null;
  cxr: { imageId: string } | null;
  notes: NotePayload[];
  /** Which modalities actually carried data, for the reduced-input banner. */
  used: Modality[];
  /** Stable hash of the inputs, so a stored prediction stays reproducible. */
  inputHash: string;
}

/** Note types this task is permitted to see at all. */
export function allowedNoteTypes(task: Task): readonly NoteType[] {
  return NOTE_TYPES_BY_TASK[task];
}

/** Maps a note type to the encoder branch it feeds. */
export function noteModality(type: NoteType): 'rr' | 'dn' {
  return type === 'discharge' ? 'dn' : 'rr';
}

/**
 * Builds the 48 x 17 raw grid the inference service expects, ending at the
 * cutoff. Hours with nothing charted stay null — the discretizer imputes them
 * exactly as it does in training, so the API must not invent values.
 */
export function buildEhrWindow(
  rows: { ts: Date; values: Record<string, unknown> }[],
  cutoff: Date,
): EhrPayload | null {
  if (rows.length === 0) return null;

  const windowStart = new Date(cutoff.getTime() - EHR_WINDOW_HOURS * 3_600_000);
  const grid: (number | string | null)[][] = Array.from({ length: EHR_WINDOW_HOURS }, () =>
    EHR_VARIABLES.map(() => null),
  );

  let any = false;
  for (const row of rows) {
    const offsetMs = row.ts.getTime() - windowStart.getTime();
    const hour = Math.floor(offsetMs / 3_600_000);
    if (hour < 0 || hour >= EHR_WINDOW_HOURS) continue;

    EHR_VARIABLES.forEach((variable, col) => {
      const raw = row.values[variable];
      if (raw === null || raw === undefined) return;
      grid[hour]![col] = raw as number | string;
      any = true;
    });
  }

  return any ? { variables: [...EHR_VARIABLES], values: grid } : null;
}

export async function gatherInputs(
  stayId: Types.ObjectId,
  task: Task,
  cutoffTime: Date,
): Promise<GatheredInputs> {
  const permitted = allowedNoteTypes(task);

  const [vitalsRows, latestImage, notes] = await Promise.all([
    // Rule 1: strictly before the cutoff.
    VitalsModel.find({
      'meta.stayId': stayId,
      ts: { $gte: new Date(cutoffTime.getTime() - EHR_WINDOW_HOURS * 3_600_000), $lt: cutoffTime },
    })
      .sort({ ts: 1 })
      .lean(),

    ImageModel.findOne({ stayId, takenAt: { $lt: cutoffTime } })
      .sort({ takenAt: -1 })
      .lean(),

    // Rules 1 and 2 together: in-window, and of a permitted type.
    NoteModel.find({
      stayId,
      authoredAt: { $lt: cutoffTime },
      type: { $in: permitted },
    })
      .sort({ authoredAt: 1 })
      .lean(),
  ]);

  const ehr = buildEhrWindow(
    vitalsRows.map((r) => ({ ts: r.ts, values: r.values })),
    cutoffTime,
  );

  const notePayloads: NotePayload[] = notes.map((n) => ({
    id: String(n._id),
    text: n.text,
    type: n.type,
  }));

  const used: Modality[] = [];
  if (ehr) used.push('ehr');
  if (latestImage) used.push('cxr');
  if (notePayloads.some((n) => noteModality(n.type) === 'rr')) used.push('rr');
  if (notePayloads.some((n) => noteModality(n.type) === 'dn')) used.push('dn');

  // Hash identifiers and shapes, never note text — the hash ends up in logs
  // and API responses.
  const inputHash = createHash('sha256')
    .update(
      JSON.stringify({
        stayId: String(stayId),
        task,
        cutoff: cutoffTime.toISOString(),
        hours: vitalsRows.length,
        image: latestImage ? String(latestImage._id) : null,
        notes: notePayloads.map((n) => [n.id, n.text.length, n.type]),
      }),
    )
    .digest('hex');

  return {
    ehr,
    cxr: latestImage ? { imageId: String(latestImage._id) } : null,
    notes: notePayloads,
    used,
    inputHash,
  };
}
