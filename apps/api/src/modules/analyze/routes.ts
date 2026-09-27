import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { z } from 'zod';
import { Router } from 'express';
import multer from 'multer';
import { Types } from 'mongoose';

import {
  EHR_VARIABLES,
  EHR_CATEGORICAL_VARIABLES,
  EHR_CATEGORICAL_VALUES,
  NOTE_TYPES,
  TASKS,
  type NoteType,
  type Task,
} from '@pneumovision/shared';

import type { Env } from '../../config/env.js';
import {
  ImageModel,
  NoteModel,
  PatientModel,
  StayModel,
  VitalsModel,
} from '../../db/models.js';
import { asyncHandler, ok } from '../../lib/http.js';
import { requireRole } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';
import type { PredictionService } from '../../services/predictionService.js';

const analyzeRequest = z.object({
  task: z.enum(TASKS),
  stayId: z.string().refine(Types.ObjectId.isValid, 'stayId must be a valid ObjectId').optional(),
});

const fileUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 20 * 1024 * 1024,
    files: 32,
  },
});

const ehrRow = z.record(z.union([z.number(), z.string(), z.null(), z.undefined()]));
const ehrPayload = z.object({
  rows: z.array(z.object({
    ts: z.string().datetime(),
    values: ehrRow,
  })).max(48),
}).optional();

const notePayload = z.object({
  type: z.enum(NOTE_TYPES),
  text: z.string().trim().min(1).max(100_000),
  authoredAt: z.string().datetime().optional(),
});

function isPng(buffer: Buffer): boolean {
  return buffer.length >= 8 &&
    buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
}

function isJpeg(buffer: Buffer): boolean {
  return buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff;
}

function assertImage(file: Express.Multer.File | undefined): asserts file is Express.Multer.File {
  if (!file) return;
  if (!isPng(file.buffer) && !isJpeg(file.buffer)) {
    throw ApiError.badRequest('INVALID_IMAGE', 'CXR must be a JPEG or PNG image');
  }
}

function parseJsonField<T>(raw: string | undefined, schema: z.ZodType<T>, field: string): T | undefined {
  if (!raw) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw ApiError.badRequest('INVALID_JSON', field + ' must contain valid JSON');
  }
  const result = schema.safeParse(parsed);
  if (!result.success) {
    throw ApiError.badRequest('INVALID_INPUT', field + ' is invalid');
  }
  return result.data;
}

function nextPseudoId() {
  return 'PV-' + Math.floor(100000 + Math.random() * 900000);
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function normalizeVitalsValues(values: Record<string, unknown>) {
  const out: Record<string, unknown> = {};
  for (const variable of EHR_VARIABLES) {
    const value = values[variable];
    if (value === undefined || value === null || value === '') {
      out[variable] = null;
      continue;
    }
    if (EHR_CATEGORICAL_VARIABLES.includes(variable)) {
      const allowed = EHR_CATEGORICAL_VALUES[variable] ?? [];
      if (typeof value !== 'string' || !allowed.includes(value as never)) {
        throw ApiError.badRequest('INVALID_EHR', 'Invalid value for ' + variable);
      }
      out[variable] = value;
      continue;
    }
    const number = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(number)) {
      throw ApiError.badRequest('INVALID_EHR', 'Invalid numeric value for ' + variable);
    }
    out[variable] = number;
  }
  return out;
}

async function ensureAdhocStay(requestedBy: Types.ObjectId) {
  const patient = await PatientModel.create({
    pseudoId: nextPseudoId(),
    demographics: { age: 0, sex: 'M' },
    createdBy: requestedBy,
  });

  return StayModel.create({
    patientId: patient._id,
    ward: 'ADHOC',
    bedLabel: 'ANALYZE',
    admittedAt: new Date(),
    status: 'active',
    availability: { ehr: false, cxr: false, notes: false },
  });
}

export function analyzeRoutes(
  env: Env,
  service: PredictionService,
): Router {
  const router = Router();
  const clinical = requireRole('clinician', 'radiologist', 'admin');

  router.post(
    '/analyze',
    clinical,
    fileUpload.fields([
      { name: 'cxr', maxCount: 1 },
      { name: 'ehr', maxCount: 1 },
      { name: 'notes', maxCount: 32 },
    ]),
    asyncHandler(async (req, res) => {
      const bodyResult = analyzeRequest.safeParse(req.body ?? {});
      if (!bodyResult.success) {
        throw ApiError.badRequest('INVALID_INPUT', 'task is required and must be mortality or pneumonia');
      }

      const task: Task = bodyResult.data.task;
      const requestedBy = new Types.ObjectId(req.user!.id);
      const files = (req.files ?? {}) as Record<string, Express.Multer.File[]>;
      const cxr = files.cxr?.[0];
      const ehrFile = files.ehr?.[0];
      const noteFiles = files.notes ?? [];

      assertImage(cxr);

      const ehrFromField = parseJsonField(req.body.ehrJson, ehrPayload, 'ehrJson');
      const noteField = parseJsonField(req.body.notesJson, z.array(notePayload).max(32), 'notesJson') ?? [];

      if (!cxr && !ehrFile && !ehrFromField && noteFiles.length === 0 && noteField.length === 0) {
        throw ApiError.badRequest('NO_MODALITY', 'At least one modality is required');
      }

      if (ehrFile && !ehrFile.buffer.toString('utf8', 0, 3).startsWith('ts,')) {
        const text = ehrFile.buffer.toString('utf8');
        const lines = text.split(/\r?\n/).filter(Boolean);
        if (lines.length > 49) throw ApiError.badRequest('INVALID_EHR', 'EHR CSV may contain at most 48 data rows');
      }

      let stay = bodyResult.data.stayId
        ? await StayModel.findById(bodyResult.data.stayId)
        : null;

      if (!stay) {
        stay = await ensureAdhocStay(requestedBy);
      }

      const cutoff = new Date();

      if (ehrFromField?.rows) {
        const docs = ehrFromField.rows.map((row) => ({
          ts: new Date(row.ts),
          meta: { stayId: stay!._id, source: 'analyze' },
          values: normalizeVitalsValues(row.values),
        }));
        if (docs.length) {
          await VitalsModel.insertMany(docs, { ordered: true });
          stay.availability.ehr = true;
        }
      }

      if (ehrFile) {
        const text = ehrFile.buffer.toString('utf8');
        const lines = text.split(/\r?\n/).filter(Boolean);
        const header = lines[0]?.split(',').map((v) => v.trim()) ?? [];
        const timestampColumn = header.findIndex((v) => v === 'ts' || v === 'timestamp');
        if (timestampColumn < 0) throw ApiError.badRequest('INVALID_EHR', 'CSV must include ts or timestamp');
        const docs = [];
        for (const line of lines.slice(1, 49)) {
          const cells = line.split(',');
          const ts = new Date(cells[timestampColumn]!);
          if (Number.isNaN(ts.getTime())) throw ApiError.badRequest('INVALID_EHR', 'CSV contains an invalid timestamp');
          const values: Record<string, unknown> = {};
          for (let i = 0; i < header.length; i++) if (i !== timestampColumn) values[header[i]!] = cells[i] ?? null;
          docs.push({ ts, meta: { stayId: stay._id, source: 'analyze' }, values: normalizeVitalsValues(values) });
        }
        if (docs.length) {
          await VitalsModel.insertMany(docs, { ordered: true });
          stay.availability.ehr = true;
        }
      }

      if (cxr) {
        const ext = isJpeg(cxr.buffer) ? 'jpg' : 'png';
        const hash = createHash('sha256').update(cxr.buffer).digest('hex');
        const dir = join(process.cwd(), env.DEMO_DATA_DIR, 'images');
        await mkdir(dir, { recursive: true });
        const fileName = `analyze-${String(stay._id)}-${hash}.${ext}`;
        await writeFile(join(dir, fileName), cxr.buffer);
        await ImageModel.create({
          stayId: stay._id,
          filePath: fileName,
          takenAt: new Date(),
          view: 'AP',
          width: 384,
          height: 384,
          sha256: hash,
          uploadedBy: requestedBy,
        });
        stay.availability.cxr = true;
      }

      const notes = [
        ...noteField,
        ...noteFiles.map((file) => ({
          type: (req.body.noteType as string) || 'radiology',
          text: file.buffer.toString('utf8'),
          authoredAt: new Date().toISOString(),
        })),
      ];

      for (const note of notes) {
        if (task === 'mortality' && note.type === 'discharge') {
          continue;
        }
        await NoteModel.create({
          stayId: stay._id,
          type: note.type,
          authoredAt: note.authoredAt ? new Date(note.authoredAt) : new Date(),
          text: note.text,
          tokenCount: estimateTokens(note.text),
        });
        stay.availability.notes = true;
      }

      await stay.save();

      const prediction = await service.request({
        stayId: stay._id,
        task,
        cutoffTime: cutoff,
        requestedBy,
        requestId: req.requestId,
      });

      res.status(prediction.status === 'done' ? 201 : 202);
      ok(res, {
        predictionId: String(prediction._id),
        stayId: String(stay._id),
        status: prediction.status,
      });
    }),
  );

  return router;
}
