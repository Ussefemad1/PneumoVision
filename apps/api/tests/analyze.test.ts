import { createHash, generateKeyPairSync } from 'node:crypto';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Router } from 'express';
import mongoose from 'mongoose';
import request from 'supertest';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { createApp } from '../src/app.js';
import { loadEnv, type Env } from '../src/config/env.js';
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
import { encodeGreyscalePng } from '../src/demo/syntheticXray.js';
import { InferenceClient } from '../src/lib/inferenceClient.js';
import { createLogger } from '../src/lib/logger.js';
import { hashPassword } from '../src/lib/password.js';
import { loadKeys } from '../src/lib/tokens.js';
import { requireAuth } from '../src/middleware/auth.js';
import { alertRoutes } from '../src/modules/alerts/routes.js';
import { analyzeRoutes, createAdhocPatient } from '../src/modules/analyze/routes.js';
import { authRoutes } from '../src/modules/auth/routes.js';
import { dashboardRoutes } from '../src/modules/dashboard/routes.js';
import { predictionRoutes } from '../src/modules/predictions/routes.js';
import { NullBus, SocketBus, userRoom } from '../src/realtime/bus.js';
import { InProcessPredictionService } from '../src/services/predictionService.js';
import { clearCollections, startMemoryMongo, stopMemoryMongo } from './helpers/mongo.js';

/**
 * The ad-hoc /analyze endpoint, end to end through the API.
 *
 * The inference service is replaced by a real HTTP server that records every
 * body it receives, so these tests assert what actually leaves the API — not
 * what a mocked client method was called with.
 */

const PASSWORD = 'correct-horse-battery';
const sha256 = (buf: Buffer) => createHash('sha256').update(buf).digest('hex');

/** A small, valid, synthetic greyscale PNG. Never a real radiograph. */
function syntheticPng(size = 16, seed = 1): Buffer {
  const pixels = new Uint8Array(size * size);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i * 7 + seed * 31) % 256;
  return encodeGreyscalePng(pixels, size, size);
}

// ── Inference stub ────────────────────────────────────────────────────────────

interface StubBody {
  task: string;
  ehr: { variables: string[]; values: (number | string | null)[][] } | null;
  cxr: {
    imageId?: string;
    contentType?: string;
    sha256?: string;
    dataB64?: string;
    presignedUrl?: string;
  } | null;
  notes: { id: string; type: string; text: string }[] | null;
}

let received: StubBody[] = [];
let stub: Server;

/** Mirrors the inference service's mock: `received` is filled from the request. */
function stubResult(body: StubBody) {
  const notes = body.notes ?? [];
  const vector = {
    ehr: body.ehr !== null,
    cxr: body.cxr !== null,
    rr: notes.some((n) => n.type !== 'discharge'),
    dn: notes.some((n) => n.type === 'discharge'),
  };
  const p = (on: boolean) => (on ? 0.4 : null);
  const bytes = body.cxr?.dataB64 ? Buffer.from(body.cxr.dataB64, 'base64') : null;
  return {
    probability: 0.4,
    unimodal: { ehr: p(vector.ehr), cxr: p(vector.cxr), rr: p(vector.rr), dn: p(vector.dn) },
    joint: { high: 0.4, low: 0.4 },
    missingness: { vector, prediction: 0.4 },
    alphas: { high: 0.3, low: 0.3, miss: 0.4 },
    confidence: { theta: 0.75, fractionAbove: {} },
    received: {
      ehr: body.ehr
        ? {
            hours: body.ehr.values.filter((row) => row.some((v) => v !== null)).length,
            variablesPresent: body.ehr.variables.filter((_, col) =>
              body.ehr!.values.some((row) => row[col] !== null),
            ),
          }
        : null,
      cxr: bytes ? { sha256: sha256(bytes), width: 16, height: 16 } : null,
      notes: notes.map((n) => ({ id: n.id, type: n.type, tokens: 3, chunks: 1 })),
      mode: 'mock',
    },
  };
}

function startStub(): Promise<string> {
  stub = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => {
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as StubBody;
      received.push(body);
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          result: stubResult(body),
          modelVersion: 'stub',
          latencyMs: 1,
          mock: true,
        }),
      );
    });
  });
  return new Promise((resolve) => {
    stub.listen(0, '127.0.0.1', () => {
      resolve(`http://127.0.0.1:${(stub.address() as AddressInfo).port}`);
    });
  });
}

// ── App ───────────────────────────────────────────────────────────────────────

let env: Env;
let dataDir: string;
let buildApp: (rateLimitPerMinute?: number) => ReturnType<typeof createApp>;
let app: ReturnType<typeof createApp>;
let cookie: string;

function testKeys() {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  return {
    priv: Buffer.from(privateKey.export({ type: 'pkcs8', format: 'pem' }) as string).toString(
      'base64',
    ),
    pub: Buffer.from(publicKey.export({ type: 'spki', format: 'pem' }) as string).toString(
      'base64',
    ),
  };
}

async function signIn(target: ReturnType<typeof createApp>): Promise<string> {
  const res = await request(target)
    .post('/api/v1/auth/login')
    .send({ email: 'clinician@pneumovision.local', password: PASSWORD });
  expect(res.status).toBe(200);
  return (res.headers['set-cookie'] as unknown as string[]).map((c) => c.split(';')[0]).join('; ');
}

beforeAll(async () => {
  await startMemoryMongo();
  const inferenceUrl = await startStub();
  dataDir = await mkdtemp(join(tmpdir(), 'pv-analyze-'));

  const keyPair = testKeys();
  env = loadEnv({
    DEMO_MODE: 'true',
    NODE_ENV: 'test',
    LOG_LEVEL: 'silent',
    WEB_ORIGIN: 'http://localhost:5173',
    JWT_PRIVATE_KEY_B64: keyPair.priv,
    JWT_PUBLIC_KEY_B64: keyPair.pub,
    INFERENCE_URL: inferenceUrl,
    INFERENCE_HMAC_SECRET: 'c'.repeat(32),
    SEED_ADMIN_PASSWORD: PASSWORD,
    SEED_CLINICIAN_PASSWORD: PASSWORD,
    DEMO_DATA_DIR: dataDir,
  });

  const logger = createLogger({ level: 'silent', isProduction: true });
  const keys = loadKeys(env);
  const inference = new InferenceClient(env, logger);
  const bus = new NullBus();
  const predictions = new InProcessPredictionService({ env, logger, inference, bus });

  buildApp = (rateLimitPerMinute = 1000) => {
    const apiRouter = Router();
    apiRouter.use('/auth', authRoutes(env, keys));
    apiRouter.use(requireAuth(keys, env));
    apiRouter.use(analyzeRoutes(env, predictions, { rateLimitPerMinute }));
    apiRouter.use(predictionRoutes(predictions));
    apiRouter.use(dashboardRoutes());
    apiRouter.use(alertRoutes(bus));
    return createApp({ env, logger, apiRouter });
  };
  app = buildApp();
}, 180_000);

afterAll(async () => {
  await stopMemoryMongo();
  await new Promise((resolve) => stub.close(resolve));
  await rm(dataDir, { recursive: true, force: true });
});

beforeEach(async () => {
  await clearCollections();
  await PatientModel.init();
  received = [];
  await UserModel.create({
    email: 'clinician@pneumovision.local',
    passwordHash: await hashPassword(PASSWORD),
    name: 'Dr Demo',
    role: 'clinician',
  });
  cookie = await signIn(app);
});

const analyze = (target = app) => request(target).post('/api/v1/analyze').set('Cookie', cookie);

const RADIOLOGY = 'Synthetic radiology note: patchy right basal opacity, no effusion.';
const DISCHARGE = 'Synthetic discharge summary: patient discharged home in stable condition.';

async function predictionFor(res: request.Response) {
  expect(res.status, JSON.stringify(res.body)).toBe(202);
  const doc = await PredictionModel.findById(res.body.data.predictionId).lean();
  expect(doc).not.toBeNull();
  return doc!;
}

// ── Bug A + B + C: uploaded inputs actually reach the model ──────────────────

describe('POST /analyze — modalities reach the model', () => {
  it('CXR only: the image bytes are sent signed, and the model reports the same hash', async () => {
    const png = syntheticPng();
    const res = await analyze()
      .field('task', 'pneumonia')
      .attach('cxr', png, { filename: 'cxr.png', contentType: 'image/png' });

    const prediction = await predictionFor(res);
    expect(prediction.status).toBe('done');
    expect(prediction.inputsUsed).toEqual(['cxr']);

    expect(received).toHaveLength(1);
    const cxr = received[0]!.cxr!;
    expect(cxr.presignedUrl).toBeUndefined();
    expect(cxr.contentType).toBe('image/png');
    expect(cxr.sha256).toBe(sha256(png));
    expect(sha256(Buffer.from(cxr.dataB64!, 'base64'))).toBe(sha256(png));

    const report = await request(app)
      .get(`/api/v1/predictions/${res.body.data.predictionId}`)
      .set('Cookie', cookie);
    expect(report.body.data.result.received.cxr.sha256).toBe(sha256(png));
    expect(report.body.data.result.received.mode).toBe('mock');
    expect(report.body.data.inputsUsed).toEqual(['cxr']);
  });

  it('notes only: a radiology note is used', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('notesJson', JSON.stringify([{ type: 'radiology', text: RADIOLOGY }]));

    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['rr']);
    expect(received[0]!.notes).toHaveLength(1);
    expect(received[0]!.notes![0]!.text).toBe(RADIOLOGY);
  });

  it('notes as files: each file takes its type from the aligned noteTypes array', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('noteTypes', JSON.stringify(['nursing', 'discharge']))
      .attach('notes', Buffer.from('Synthetic nursing note.'), 'a.txt')
      .attach('notes', Buffer.from(DISCHARGE), 'b.txt');

    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['rr', 'dn']);
    expect(received[0]!.notes!.map((n) => n.type).sort()).toEqual(['discharge', 'nursing']);
  });

  it('EHR only (JSON rows by hour offset): the vitals are used', async () => {
    const rows = [-2, -1, 0].map((hour) => ({
      hour,
      values: { 'Heart Rate': 100 + hour, 'Glascow coma scale total': '15' },
    }));
    const res = await analyze()
      .field('task', 'mortality')
      .field('ehrJson', JSON.stringify({ rows }));

    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['ehr']);
    const ehr = received[0]!.ehr!;
    // The last three hourly bins of the 48-hour window carry the rows.
    expect(ehr.values.slice(-3).map((r) => r[ehr.variables.indexOf('Heart Rate')])).toEqual([
      98, 99, 100,
    ]);
  });

  it('EHR stamped "now" is not dropped as future data', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .field(
        'ehrJson',
        JSON.stringify({ rows: [{ ts: new Date().toISOString(), values: { 'Heart Rate': 90 } }] }),
      );
    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['ehr']);
  });

  it('EHR only (CSV): the vitals are used', async () => {
    const csv = ['hour,Heart Rate,Oxygen saturation', '-1,96,94', '0,101,92'].join('\r\n');
    const res = await analyze()
      .field('task', 'mortality')
      .attach('ehr', Buffer.from(csv), { filename: 'ehr.csv', contentType: 'text/csv' });

    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['ehr']);
    expect(await VitalsModel.countDocuments()).toBe(2);
  });

  it('pneumonia with a discharge note sends it to the DN branch', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('notesJson', JSON.stringify([{ type: 'discharge', text: DISCHARGE }]));

    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['dn']);
    expect(res.body.data.excluded).toEqual([]);
  });
});

// ── F8 on the ad-hoc path ─────────────────────────────────────────────────────

describe('POST /analyze — F8 leakage control', () => {
  it('mortality with all three + a discharge note: DN is neither stored nor sent', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .field('ehrJson', JSON.stringify({ rows: [{ hour: 0, values: { 'Heart Rate': 88 } }] }))
      .field(
        'notesJson',
        JSON.stringify([
          { type: 'radiology', text: RADIOLOGY },
          { type: 'discharge', text: DISCHARGE },
        ]),
      )
      .attach('cxr', syntheticPng(), { filename: 'cxr.png', contentType: 'image/png' });

    const prediction = await predictionFor(res);
    expect(prediction.inputsUsed).toEqual(['ehr', 'cxr', 'rr']);
    expect(received[0]!.notes!.map((n) => n.type)).toEqual(['radiology']);
    expect(await NoteModel.countDocuments({ type: 'discharge' })).toBe(0);
    expect(res.body.data.excluded).toEqual([
      { source: 'notesJson', index: 1, type: 'discharge', reason: 'outcome-leakage' },
    ]);
    expect(prediction.excludedInputs).toHaveLength(1);
  });

  it('mortality with only a discharge note has no usable modality', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .field('notesJson', JSON.stringify([{ type: 'discharge', text: DISCHARGE }]));
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('NO_MODALITY');
    expect(await NoteModel.countDocuments()).toBe(0);
  });
});

// ── Bug D: ad-hoc stays stay out of the ward ──────────────────────────────────

describe('POST /analyze — ad-hoc stays', () => {
  it('creates an adhoc stay with no demographics, absent from the ward dashboard', async () => {
    await StayModel.create({
      patientId: new mongoose.Types.ObjectId(),
      ward: 'MICU',
      bedLabel: 'M-01',
      admittedAt: new Date(),
      availability: { ehr: false, cxr: false, notes: false },
    });

    const res = await analyze()
      .field('task', 'pneumonia')
      .field('notesJson', JSON.stringify([{ type: 'radiology', text: RADIOLOGY }]));
    await predictionFor(res);

    const stay = await StayModel.findById(res.body.data.stayId).lean();
    expect(stay!.status).toBe('adhoc');
    expect(stay!.ward).toBe('ADHOC');
    const patient = await PatientModel.findById(stay!.patientId).lean();
    expect(patient!.demographics ?? null).toBeNull();

    const ward = await request(app).get('/api/v1/dashboard/ward').set('Cookie', cookie);
    expect(ward.status).toBe(200);
    expect(ward.body.data.rows.map((r: { stayId: string }) => r.stayId)).not.toContain(
      res.body.data.stayId,
    );
    expect(ward.body.data.wards).not.toContain('ADHOC');
  });

  it('never raises alerts for an adhoc stay', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .field('ehrJson', JSON.stringify({ rows: [{ hour: 0, values: { 'Heart Rate': 150 } }] }));
    await predictionFor(res);
    expect(await AlertModel.countDocuments({ stayId: res.body.data.stayId })).toBe(0);
  });

  it('retries the pseudoId on a unique-index collision', async () => {
    await PatientModel.create({ pseudoId: 'PV-500000' });
    const ids = ['PV-500000', 'PV-500001'];
    const patient = await createAdhocPatient(null, () => ids.shift()!);
    expect(patient.pseudoId).toBe('PV-500001');
  });

  it('attaches to an existing stay when stayId is given', async () => {
    const stay = await StayModel.create({
      patientId: new mongoose.Types.ObjectId(),
      ward: 'MICU',
      bedLabel: 'M-02',
      admittedAt: new Date(Date.now() - 86_400_000),
      availability: { ehr: false, cxr: false, notes: false },
    });
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('stayId', String(stay._id))
      .field('notesJson', JSON.stringify([{ type: 'radiology', text: RADIOLOGY }]));
    await predictionFor(res);
    expect(res.body.data.stayId).toBe(String(stay._id));
    expect((await StayModel.findById(stay._id).lean())!.availability.notes).toBe(true);
  });

  it('404s for an unknown stayId', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('stayId', String(new mongoose.Types.ObjectId()))
      .field('notesJson', JSON.stringify([{ type: 'radiology', text: RADIOLOGY }]));
    expect(res.status).toBe(404);
  });
});

// ── Validation and limits (Bug E) ─────────────────────────────────────────────

describe('POST /analyze — validation', () => {
  it('rejects bytes that are not a PNG or JPEG', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .attach('cxr', Buffer.from('definitely not an image'), 'cxr.png');
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('INVALID_IMAGE');
  });

  it('rejects a truncated PNG whose magic bytes are valid', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .attach('cxr', syntheticPng().subarray(0, 12), 'cxr.png');
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('INVALID_IMAGE');
  });

  it('rejects an oversize X-ray with 413', async () => {
    const big = Buffer.alloc(20 * 1024 * 1024 + 1);
    syntheticPng().copy(big);
    const res = await analyze().field('task', 'pneumonia').attach('cxr', big, 'cxr.png');
    expect(res.status).toBe(413);
    expect(res.body.error.code).toBe('CXR_TOO_LARGE');
  });

  it('rejects an oversize note file with 413', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('noteTypes', JSON.stringify(['radiology']))
      .attach('notes', Buffer.alloc(200 * 1024 + 1, 'a'), 'n.txt');
    expect(res.status).toBe(413);
    expect(res.body.error.code).toBe('NOTE_FILE_TOO_LARGE');
  });

  it('rejects more than 32 note files with 413', async () => {
    let req = analyze()
      .field('task', 'pneumonia')
      .field('noteTypes', JSON.stringify(Array.from({ length: 33 }, () => 'radiology')));
    for (let i = 0; i < 33; i++) req = req.attach('notes', Buffer.from(`note ${i}`), `${i}.txt`);
    const res = await req;
    expect(res.status).toBe(413);
    expect(res.body.error.code).toBe('TOO_MANY_NOTES');
  });

  it('rejects note files whose noteTypes do not line up', async () => {
    const res = await analyze()
      .field('task', 'pneumonia')
      .field('noteTypes', JSON.stringify(['radiology', 'nursing']))
      .attach('notes', Buffer.from(RADIOLOGY), 'a.txt');
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('NOTE_TYPES_MISMATCH');
  });

  it('rejects an unknown EHR variable instead of ignoring it', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .field('ehrJson', JSON.stringify({ rows: [{ hour: 0, values: { 'Heart rate': 90 } }] }));
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('EHR_UNKNOWN_VARIABLE');
  });

  it('rejects an unknown EHR CSV column', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .attach('ehr', Buffer.from('hour,Pulse\n0,90\n'), 'ehr.csv');
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('EHR_UNKNOWN_VARIABLE');
  });

  it('rejects a categorical value outside the allowed list', async () => {
    const res = await analyze()
      .field('task', 'mortality')
      .field(
        'ehrJson',
        JSON.stringify({ rows: [{ hour: 0, values: { 'Glascow coma scale total': '99' } }] }),
      );
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('EHR_INVALID_VALUE');
  });

  it('rejects an EHR row in the future', async () => {
    const future = new Date(Date.now() + 3_600_000).toISOString();
    const res = await analyze()
      .field('task', 'mortality')
      .field('ehrJson', JSON.stringify({ rows: [{ ts: future, values: { 'Heart Rate': 90 } }] }));
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('EHR_FUTURE_TIMESTAMP');
    expect(await VitalsModel.countDocuments()).toBe(0);
  });

  it('rejects an EHR CSV with more than 48 rows', async () => {
    const lines = ['hour,Heart Rate', ...Array.from({ length: 49 }, (_, i) => `${-i},80`)];
    const res = await analyze()
      .field('task', 'mortality')
      .attach('ehr', Buffer.from(lines.join('\n')), 'ehr.csv');
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('EHR_TOO_MANY_ROWS');
  });

  it('rejects a request with no modality', async () => {
    const res = await analyze().field('task', 'pneumonia');
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('NO_MODALITY');
  });

  it('rejects a non-multipart body', async () => {
    const res = await analyze().send({ task: 'pneumonia' });
    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('INVALID_CONTENT_TYPE');
  });

  it('rate-limits per user', async () => {
    const limited = buildApp(2);
    const send = () =>
      request(limited).post('/api/v1/analyze').set('Cookie', cookie).field('task', 'pneumonia');
    expect((await send()).status).toBe(400);
    expect((await send()).status).toBe(400);
    const third = await send();
    expect(third.status).toBe(429);
    expect(third.body.error.code).toBe('RATE_LIMITED');
  });
});

// ── Bug B regression on the stay path ─────────────────────────────────────────

describe('POST /stays/:id/predictions', () => {
  it('sends the stored image bytes, not a cookie-protected URL', async () => {
    const png = syntheticPng(24, 3);
    await mkdir(join(dataDir, 'images'), { recursive: true });
    await writeFile(join(dataDir, 'images', 'stay-test.png'), png);

    const stay = await StayModel.create({
      patientId: new mongoose.Types.ObjectId(),
      ward: 'MICU',
      bedLabel: 'M-03',
      admittedAt: new Date(Date.now() - 86_400_000),
      availability: { ehr: false, cxr: true, notes: false },
    });
    await ImageModel.create({
      stayId: stay._id,
      filePath: 'stay-test.png',
      takenAt: new Date(Date.now() - 3_600_000),
      view: 'AP',
      width: 24,
      height: 24,
      sha256: sha256(png),
    });

    const res = await request(app)
      .post(`/api/v1/stays/${String(stay._id)}/predictions`)
      .set('Cookie', cookie)
      .send({ task: 'pneumonia' });

    expect(res.status).toBe(201);
    expect(res.body.data.status).toBe('done');
    const cxr = received[0]!.cxr!;
    expect(cxr.presignedUrl).toBeUndefined();
    expect(cxr.sha256).toBe(sha256(png));
    expect(Buffer.from(cxr.dataB64!, 'base64').equals(png)).toBe(true);
  });
});

// ── Realtime ──────────────────────────────────────────────────────────────────

describe('SocketBus', () => {
  it('emits prediction status to the stay room and the requesting user room', () => {
    const rooms: string[] = [];
    const io = { to: (room: string) => ({ emit: () => rooms.push(room) }) };
    const bus = new SocketBus(io as never);
    const userId = new mongoose.Types.ObjectId();
    const stayId = new mongoose.Types.ObjectId();
    bus.predictionStatus(
      new PredictionModel({
        stayId,
        task: 'mortality',
        cutoffTime: new Date(),
        requestedBy: userId,
      }),
      'running',
    );
    expect(rooms).toEqual([`stay:${String(stayId)}`, userRoom(String(userId))]);
  });
});
