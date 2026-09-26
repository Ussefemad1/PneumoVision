import { randomBytes } from 'node:crypto';

import mongoose from 'mongoose';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import {
  AlertModel,
  ImageModel,
  NoteModel,
  PatientModel,
  PredictionModel,
  StayModel,
  UserModel,
  VitalsModel,
} from '../src/db/models.js';
import { clearCollections, startMemoryMongo, stopMemoryMongo } from './helpers/mongo.js';

const oid = () => new mongoose.Types.ObjectId();
const sha = () => randomBytes(32).toString('hex');

beforeAll(async () => {
  await startMemoryMongo();
}, 180_000);

afterAll(async () => {
  await stopMemoryMongo();
});

beforeEach(async () => {
  await clearCollections();
});

describe('UserModel', () => {
  it('stores a valid user and keeps the hash unselected', async () => {
    await UserModel.create({
      email: 'Clinician@Example.COM',
      passwordHash: '$argon2id$v=19$fake',
      name: 'Dr Demo',
      role: 'clinician',
    });

    const found = await UserModel.findOne({ email: 'clinician@example.com' });
    expect(found?.name).toBe('Dr Demo');
    // passwordHash has select:false, so a default read must not return it.
    expect((found as unknown as { passwordHash?: string }).passwordHash).toBeUndefined();

    const withHash = await UserModel.findOne({ email: 'clinician@example.com' }).select(
      '+passwordHash',
    );
    expect(withHash?.passwordHash).toContain('argon2id');
  });

  it('rejects an unknown role', async () => {
    await expect(
      UserModel.create({
        email: 'x@example.com',
        passwordHash: 'h',
        name: 'X',
        role: 'superuser',
      }),
    ).rejects.toThrow(/role/i);
  });

  it('enforces a unique email', async () => {
    await UserModel.syncIndexes();
    await UserModel.create({
      email: 'dup@example.com',
      passwordHash: 'h',
      name: 'A',
      role: 'admin',
    });
    await expect(
      UserModel.create({ email: 'dup@example.com', passwordHash: 'h', name: 'B', role: 'admin' }),
    ).rejects.toThrow(/duplicate key/i);
  });
});

describe('PatientModel', () => {
  it('stores a pseudonymous patient', async () => {
    const patient = await PatientModel.create({
      pseudoId: 'PV-000123',
      demographics: { age: 71, sex: 'M' },
    });
    expect(patient.pseudoId).toBe('PV-000123');
  });

  it('rejects a pseudoId that is not in PV-000000 form', async () => {
    await expect(
      PatientModel.create({ pseudoId: '12345', demographics: { age: 60, sex: 'F' } }),
    ).rejects.toThrow(/pseudoId/);
  });

  it('rejects an out-of-range age', async () => {
    await expect(
      PatientModel.create({ pseudoId: 'PV-000999', demographics: { age: 999, sex: 'F' } }),
    ).rejects.toThrow(/age/);
  });
});

describe('StayModel', () => {
  it('stores a stay with modality availability', async () => {
    const stay = await StayModel.create({
      patientId: oid(),
      ward: 'MICU',
      bedLabel: 'B-04',
      admittedAt: new Date(),
      availability: { ehr: true, cxr: false, notes: true },
    });
    expect(stay.status).toBe('active');
    expect(stay.availability.cxr).toBe(false);
  });

  it('rejects an unknown status', async () => {
    await expect(
      StayModel.create({
        patientId: oid(),
        ward: 'MICU',
        bedLabel: 'B-01',
        admittedAt: new Date(),
        status: 'transferred',
      }),
    ).rejects.toThrow(/status/);
  });

  it('rejects fields outside the schema', async () => {
    await expect(
      StayModel.create({
        patientId: oid(),
        ward: 'MICU',
        bedLabel: 'B-01',
        admittedAt: new Date(),
        mrn: '12345678',
      }),
    ).rejects.toThrow(/mrn/);
  });
});

describe('VitalsModel', () => {
  it('is created as a native time-series collection', async () => {
    await VitalsModel.createCollection();
    const infos = await mongoose.connection.db!.listCollections({ name: 'vitals' }).toArray();
    expect(infos[0]?.type).toBe('timeseries');
    expect(infos[0]?.options?.timeseries).toMatchObject({
      timeField: 'ts',
      metaField: 'meta',
      granularity: 'hours',
    });
  });

  it('stores an hourly bin with partial values', async () => {
    const stayId = oid();
    await VitalsModel.create({
      ts: new Date('2026-01-01T00:00:00Z'),
      meta: { stayId, source: 'seed' },
      values: { 'Heart Rate': 92, 'Oxygen saturation': 95, 'Glascow coma scale total': '15' },
    });

    const found = await VitalsModel.findOne({ 'meta.stayId': stayId });
    expect(found?.values['Heart Rate']).toBe(92);
    // Uncharted variables stay null rather than being invented.
    expect(found?.values['Glucose']).toBeNull();
  });

  it('rejects a categorical value outside the discretizer vocabulary', async () => {
    await expect(
      VitalsModel.create({
        ts: new Date(),
        meta: { stayId: oid() },
        values: { 'Glascow coma scale total': '42' },
      }),
    ).rejects.toThrow(/Glascow coma scale total/);
  });

  it('rejects a variable that is not one of the 17', async () => {
    await expect(
      VitalsModel.create({
        ts: new Date(),
        meta: { stayId: oid() },
        values: { 'Blood alcohol': 0.1 },
      }),
    ).rejects.toThrow(/Blood alcohol/);
  });
});

describe('ImageModel', () => {
  it('stores a radiograph pointer, never pixels', async () => {
    const img = await ImageModel.create({
      stayId: oid(),
      filePath: 'stay-1/cxr-0.png',
      takenAt: new Date(),
      view: 'AP',
      width: 384,
      height: 384,
      sha256: sha(),
    });
    expect(img.objectKey).toBeNull();
    expect(img.view).toBe('AP');
  });

  it('rejects a malformed sha256', async () => {
    await expect(
      ImageModel.create({
        stayId: oid(),
        filePath: 'x.png',
        takenAt: new Date(),
        view: 'AP',
        width: 384,
        height: 384,
        sha256: 'not-a-hash',
      }),
    ).rejects.toThrow(/sha256/);
  });

  it('rejects an unknown view', async () => {
    await expect(
      ImageModel.create({
        stayId: oid(),
        filePath: 'x.png',
        takenAt: new Date(),
        view: 'OBLIQUE',
        width: 384,
        height: 384,
        sha256: sha(),
      }),
    ).rejects.toThrow(/view/);
  });
});

describe('NoteModel', () => {
  it('stores a note with its token count', async () => {
    const note = await NoteModel.create({
      stayId: oid(),
      type: 'radiology',
      authoredAt: new Date(),
      text: 'Synthetic report. Patchy opacity at the right base.',
      tokenCount: 12,
    });
    expect(note.type).toBe('radiology');
  });

  it('rejects an unknown note type', async () => {
    await expect(
      NoteModel.create({
        stayId: oid(),
        type: 'consult',
        authoredAt: new Date(),
        text: 'x',
        tokenCount: 1,
      }),
    ).rejects.toThrow(/type/);
  });
});

describe('PredictionModel', () => {
  const validResult = {
    probability: 0.62,
    unimodal: { ehr: 0.58, cxr: 0.7, rr: 0.6, dn: null },
    joint: { high: 0.66, low: 0.4 },
    missingness: { vector: { ehr: true, cxr: true, rr: true, dn: false }, prediction: 0.5 },
    alphas: { high: 0.3, low: 0.1, miss: 0.1, ehr: 0.2, cxr: 0.2, rr: 0.1, dn: null },
    confidence: {
      theta: 0.75,
      fractionAbove: { ehr: 0.5, cxr: 0.6, rr: 0.4, dn: null },
      ehrTimesteps: Array.from({ length: 48 }, () => 0.8),
      cxrPatchGrid: [
        [0.8, 0.9],
        [0.7, 0.6],
      ],
      noteSpans: [{ noteId: 'n1', start: 0, end: 20, confidence: 0.81 }],
    },
  };

  it('stores the complete F1-F6 result', async () => {
    const pred = await PredictionModel.create({
      stayId: oid(),
      task: 'mortality',
      cutoffTime: new Date(),
      status: 'done',
      modelVersion: 'mock-0.1.0',
      inputHash: 'abc',
      latencyMs: 42,
      inputsUsed: ['ehr', 'cxr', 'rr'],
      result: validResult,
    });

    expect(pred.result?.probability).toBe(0.62);
    expect(pred.result?.alphas.dn).toBeNull();
    expect(pred.result?.confidence.ehrTimesteps).toHaveLength(48);
    expect(pred.result?.missingness.vector.dn).toBe(false);
  });

  it('rejects a probability outside [0, 1]', async () => {
    await expect(
      PredictionModel.create({
        stayId: oid(),
        task: 'mortality',
        cutoffTime: new Date(),
        result: { ...validResult, probability: 1.4 },
      }),
    ).rejects.toThrow(/probability/);
  });

  it('rejects an unknown task', async () => {
    await expect(
      PredictionModel.create({ stayId: oid(), task: 'sepsis', cutoffTime: new Date() }),
    ).rejects.toThrow(/task/);
  });

  it('declares the stay/task/cutoff index', async () => {
    await PredictionModel.syncIndexes();
    const indexes = await PredictionModel.collection.indexes();
    expect(
      indexes.some((i) => i.key.stayId === 1 && i.key.task === 1 && i.key.cutoffTime === -1),
    ).toBe(true);
  });
});

describe('AlertModel', () => {
  it('stores an open alert', async () => {
    const alert = await AlertModel.create({
      stayId: oid(),
      predictionId: oid(),
      task: 'mortality',
      severity: 'critical',
      rule: 'Mortality risk above 70%',
      value: 0.82,
    });
    expect(alert.status).toBe('open');
    expect(alert.acknowledgedAt).toBeNull();
  });

  it('rejects an unknown severity', async () => {
    await expect(
      AlertModel.create({
        stayId: oid(),
        task: 'mortality',
        severity: 'catastrophic',
        rule: 'x',
        value: 1,
      }),
    ).rejects.toThrow(/severity/);
  });

  it('declares the status/severity index', async () => {
    await AlertModel.syncIndexes();
    const indexes = await AlertModel.collection.indexes();
    expect(indexes.some((i) => i.key.status === 1 && i.key.severity === 1)).toBe(true);
  });
});
