import mongoose from 'mongoose';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { ImageModel, NoteModel, StayModel, VitalsModel } from '../src/db/models.js';
import {
  allowedNoteTypes,
  buildEhrWindow,
  gatherInputs,
  noteModality,
} from '../src/services/predictionInputs.js';
import { clearCollections, startMemoryMongo, stopMemoryMongo } from './helpers/mongo.js';

/**
 * F8 — leakage control.
 *
 * Two rules, both enforced in `predictionInputs.ts`:
 *   1. nothing timestamped at or after the cutoff may be sent;
 *   2. discharge notes are never sent for a mortality prediction.
 *
 * These are the tests that would catch a regression silently inflating the
 * model's apparent performance, so they check the actual gathered payload
 * rather than the helper functions alone.
 */

const T0 = new Date('2026-03-01T00:00:00.000Z');
const at = (hours: number) => new Date(T0.getTime() + hours * 3_600_000);

let stayId: mongoose.Types.ObjectId;

beforeAll(async () => {
  await startMemoryMongo();
}, 180_000);

afterAll(async () => {
  await stopMemoryMongo();
});

beforeEach(async () => {
  await clearCollections();
  const stay = await StayModel.create({
    patientId: new mongoose.Types.ObjectId(),
    ward: 'MICU',
    bedLabel: 'M-01',
    admittedAt: T0,
    availability: { ehr: true, cxr: true, notes: true },
  });
  stayId = stay._id;
});

async function seedTimeline() {
  // Vitals every hour for 30 hours.
  await VitalsModel.insertMany(
    Array.from({ length: 30 }, (_, hour) => ({
      ts: at(hour),
      meta: { stayId, source: 'test' },
      values: { 'Heart Rate': 80 + hour },
    })),
  );

  await ImageModel.create({
    stayId,
    filePath: 'before.png',
    takenAt: at(5),
    view: 'AP',
    width: 384,
    height: 384,
    sha256: 'a'.repeat(64),
  });
  await ImageModel.create({
    stayId,
    filePath: 'after.png',
    takenAt: at(25),
    view: 'AP',
    width: 384,
    height: 384,
    sha256: 'b'.repeat(64),
  });

  await NoteModel.create({
    stayId,
    type: 'radiology',
    authoredAt: at(4),
    text: 'Radiology before cutoff',
    tokenCount: 10,
  });
  await NoteModel.create({
    stayId,
    type: 'progress',
    authoredAt: at(22),
    text: 'Progress note after cutoff',
    tokenCount: 10,
  });
  await NoteModel.create({
    stayId,
    type: 'discharge',
    authoredAt: at(6),
    text: 'Discharge summary, well before the cutoff',
    tokenCount: 10,
  });
}

describe('rule 1 — nothing at or after the cutoff', () => {
  it('excludes vitals, images and notes recorded after the cutoff', async () => {
    await seedTimeline();
    const inputs = await gatherInputs(stayId, 'pneumonia', at(10));

    // The note authored at hour 22 is after the cutoff.
    const texts = inputs.notes.map((n) => n.text);
    expect(texts).toContain('Radiology before cutoff');
    expect(texts).not.toContain('Progress note after cutoff');

    // The hour-25 radiograph must not be the one chosen.
    const chosen = await ImageModel.findById(inputs.cxr!.imageId).lean();
    expect(chosen?.filePath).toBe('before.png');

    // Only the 10 hours before the cutoff carry data.
    const charted = inputs.ehr!.values.filter((row) => row.some((v) => v !== null));
    expect(charted).toHaveLength(10);
  });

  it('treats the cutoff as strictly exclusive', async () => {
    await VitalsModel.insertMany([
      { ts: at(0), meta: { stayId, source: 'test' }, values: { 'Heart Rate': 70 } },
      // Exactly on the cutoff — must be excluded.
      { ts: at(1), meta: { stayId, source: 'test' }, values: { 'Heart Rate': 999 } },
    ]);

    const inputs = await gatherInputs(stayId, 'mortality', at(1));
    const flat = inputs.ehr!.values.flat().filter((v) => v !== null);
    expect(flat).toContain(70);
    expect(flat).not.toContain(999);
  });

  it('reports no modalities when everything postdates the cutoff', async () => {
    await seedTimeline();
    const inputs = await gatherInputs(stayId, 'mortality', at(0));

    expect(inputs.ehr).toBeNull();
    expect(inputs.cxr).toBeNull();
    expect(inputs.notes).toHaveLength(0);
    expect(inputs.used).toEqual([]);
  });
});

describe('rule 2 — discharge notes never reach a mortality prediction', () => {
  it('drops the discharge note for mortality even though it precedes the cutoff', async () => {
    await seedTimeline();
    const inputs = await gatherInputs(stayId, 'mortality', at(29));

    expect(inputs.notes.some((n) => n.type === 'discharge')).toBe(false);
    // The other pre-cutoff notes still come through, so this is not a
    // blanket exclusion.
    expect(inputs.notes.map((n) => n.type).sort()).toEqual(['progress', 'radiology']);
    expect(inputs.used).not.toContain('dn');
  });

  it('allows the discharge note for the pneumonia phenotype task', async () => {
    await seedTimeline();
    const inputs = await gatherInputs(stayId, 'pneumonia', at(29));

    expect(inputs.notes.some((n) => n.type === 'discharge')).toBe(true);
    expect(inputs.used).toContain('dn');
  });

  it('declares the permitted note types per task', () => {
    expect(allowedNoteTypes('mortality')).not.toContain('discharge');
    expect(allowedNoteTypes('pneumonia')).toContain('discharge');
  });

  it('routes discharge notes to the DN branch and everything else to RR', () => {
    expect(noteModality('discharge')).toBe('dn');
    for (const type of ['radiology', 'progress', 'nursing'] as const) {
      expect(noteModality(type)).toBe('rr');
    }
  });
});

describe('EHR window construction', () => {
  it('lays rows out by hour offset and leaves gaps null', () => {
    const cutoff = at(48);
    const window = buildEhrWindow(
      [
        { ts: at(0), values: { 'Heart Rate': 60 } },
        { ts: at(47), values: { 'Heart Rate': 120 } },
      ],
      cutoff,
    );

    // 48 hourly rows x the 17 raw variables (the discretizer widens this to
    // 76 columns inside the inference service, not here).
    expect(window!.values).toHaveLength(48);
    expect(window!.values.every((row) => row.length === 17)).toBe(true);
    expect(window!.variables).toHaveLength(17);
    // Hour 0 of the window is 48h before the cutoff.
    expect(window!.values[0]!.some((v) => v === 60)).toBe(true);
    expect(window!.values[47]!.some((v) => v === 120)).toBe(true);
    // Everything between was never charted.
    expect(window!.values[20]!.every((v) => v === null)).toBe(true);
  });

  it('never invents a value for an unrecorded hour', () => {
    const window = buildEhrWindow([{ ts: at(10), values: { 'Heart Rate': 88 } }], at(48));
    const populated = window!.values.filter((row) => row.some((v) => v !== null));
    expect(populated).toHaveLength(1);
  });

  it('returns null when the window holds nothing at all', () => {
    expect(buildEhrWindow([], at(48))).toBeNull();
    expect(buildEhrWindow([{ ts: at(0), values: {} }], at(48))).toBeNull();
  });
});

describe('used modalities', () => {
  it('omits a modality that carried no data', async () => {
    // Vitals and a radiology note, but no radiograph.
    await VitalsModel.create({
      ts: at(1),
      meta: { stayId, source: 'test' },
      values: { 'Heart Rate': 90 },
    });
    await NoteModel.create({
      stayId,
      type: 'radiology',
      authoredAt: at(1),
      text: 'Report',
      tokenCount: 5,
    });

    const inputs = await gatherInputs(stayId, 'mortality', at(10));
    expect(inputs.used).toContain('ehr');
    expect(inputs.used).toContain('rr');
    expect(inputs.used).not.toContain('cxr');
    expect(inputs.cxr).toBeNull();
  });

  it('lists dn only when a discharge note was actually sent', async () => {
    await NoteModel.create({
      stayId,
      type: 'discharge',
      authoredAt: at(1),
      text: 'Summary',
      tokenCount: 5,
    });

    expect((await gatherInputs(stayId, 'pneumonia', at(10))).used).toContain('dn');
    expect((await gatherInputs(stayId, 'mortality', at(10))).used).not.toContain('dn');
  });
});

describe('input hash', () => {
  it('is stable for identical inputs and changes with the cutoff', async () => {
    await seedTimeline();
    const a = await gatherInputs(stayId, 'mortality', at(10));
    const b = await gatherInputs(stayId, 'mortality', at(10));
    const c = await gatherInputs(stayId, 'mortality', at(11));

    expect(a.inputHash).toBe(b.inputHash);
    expect(a.inputHash).not.toBe(c.inputHash);
  });

  it('does not change when note text changes but its length does not', async () => {
    // The hash covers identifiers and shapes, never the text itself: it ends
    // up in logs and API responses, so it must not be a carrier of clinical
    // content. Equal-length text must therefore hash identically.
    const note = await NoteModel.create({
      stayId,
      type: 'radiology',
      authoredAt: at(1),
      text: 'AAAAAAAAAA',
      tokenCount: 5,
    });
    const before = (await gatherInputs(stayId, 'mortality', at(10))).inputHash;

    await NoteModel.findByIdAndUpdate(note._id, { text: 'BBBBBBBBBB' });
    const after = (await gatherInputs(stayId, 'mortality', at(10))).inputHash;

    expect(after).toBe(before);
    expect(before).toMatch(/^[0-9a-f]{64}$/);
  });

  it('does change when the note length changes', async () => {
    const note = await NoteModel.create({
      stayId,
      type: 'radiology',
      authoredAt: at(1),
      text: 'AAAAAAAAAA',
      tokenCount: 5,
    });
    const before = (await gatherInputs(stayId, 'mortality', at(10))).inputHash;

    await NoteModel.findByIdAndUpdate(note._id, { text: 'AAAAAAAAAAA' });
    const after = (await gatherInputs(stayId, 'mortality', at(10))).inputHash;

    expect(after).not.toBe(before);
  });
});
