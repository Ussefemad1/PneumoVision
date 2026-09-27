import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { Request } from 'express';
import { z } from 'zod';
import { Router } from 'express';
import { Types } from 'mongoose';

import {
  EHR_VARIABLES, EHR_CATEGORICAL_VARIABLES, EHR_CATEGORICAL_VALUES,
  NOTE_TYPES, TASKS, type NoteType, type Task,
} from '@pneumovision/shared';
import type { Env } from '../../config/env.js';
import { ImageModel, NoteModel, PatientModel, StayModel, VitalsModel } from '../../db/models.js';
import { asyncHandler, ok } from '../../lib/http.js';
import { requireRole } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';
import type { PredictionService } from '../../services/predictionService.js';

const analyzeRequest = z.object({
  task: z.enum(TASKS),
  stayId: z.string().refine(Types.ObjectId.isValid, 'stayId must be a valid ObjectId').optional(),
});
const ehrPayload = z.object({
  rows: z.array(z.object({
    ts: z.string().datetime(),
    values: z.record(z.union([z.number(), z.string(), z.null(), z.undefined()])),
  })).max(48),
}).optional();
const notePayload = z.object({
  type: z.enum(NOTE_TYPES),
  text: z.string().trim().min(1).max(100_000),
  authoredAt: z.string().datetime().optional(),
});

interface MultipartFile { fieldName: string; fileName: string; contentType: string; buffer: Buffer }
interface MultipartBody { fields: Record<string, string>; files: MultipartFile[] }

async function parseMultipart(req: Request): Promise<MultipartBody> {
  const contentType = req.headers['content-type'] ?? '';
  const match = /^multipart\/form-data;\\s*boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
  if (!match) throw ApiError.badRequest('INVALID_CONTENT_TYPE', 'Use multipart/form-data');
  const boundary = Buffer.from(`--${match[1] ?? match[2]}`);
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const part = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += part.length;
    if (total > 25 * 1024 * 1024) throw ApiError.badRequest('PAYLOAD_TOO_LARGE', 'Analyze payload exceeds 25 MB');
    chunks.push(part);
  }
  const body = Buffer.concat(chunks);
  const fields: Record<string, string> = {};
  const files: MultipartFile[] = [];
  let cursor = 0;
  while (true) {
    const start = body.indexOf(boundary, cursor);
    if (start < 0) break;
    const headerStart = start + boundary.length;
    if (body.subarray(headerStart, headerStart + 2).equals(Buffer.from('--'))) break;
    const contentStart = body.indexOf(Buffer.from('\\r\\n\\r\\n'), headerStart);
    if (contentStart < 0) break;
    const headers = body.subarray(headerStart + 2, contentStart).toString('utf8');
    const next = body.indexOf(boundary, contentStart + 4);
    if (next < 0) break;
    const contentEnd = next - 2;
    const content = body.subarray(contentStart + 4, contentEnd);
    const disposition = /content-disposition:\\s*form-data;\\s*name="([^"]+)"(?:;\\s*filename="([^"]*)")?/i.exec(headers);
    if (!disposition) { cursor = next; continue; }
    const fieldName = disposition[1]!;
    const fileName = disposition[2];
    const typeMatch = /content-type:\\s*([^\\r\\n]+)/i.exec(headers);
    if (fileName !== undefined) files.push({ fieldName, fileName, contentType: typeMatch?.[1]?.trim() ?? 'application/octet-stream', buffer: Buffer.from(content) });
    else fields[fieldName] = content.toString('utf8');
    cursor = next;
  }
  return { fields, files };
}

function isPng(buffer: Buffer): boolean {
  return buffer.length >= 8 && buffer.subarray(0, 8).equals(Buffer.from([0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]));
}
function isJpeg(buffer: Buffer): boolean {
  return buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff;
}
function parseJsonField<T>(raw: string | undefined, schema: z.ZodType<T>, field: string): T | undefined {
  if (!raw) return undefined;
  let parsed: unknown;
  try { parsed = JSON.parse(raw); } catch { throw ApiError.badRequest('INVALID_JSON', field + ' must contain valid JSON'); }
  const result = schema.safeParse(parsed);
  if (!result.success) throw ApiError.badRequest('INVALID_INPUT', field + ' is invalid');
  return result.data;
}
function nextPseudoId() { return 'PV-' + Math.floor(100000 + Math.random() * 900000); }
function estimateTokens(text: string) { return Math.ceil(text.length / 4); }

function normalizeVitalsValues(values: Record<string, unknown>) {
  const out: Record<string, unknown> = {};
  for (const variable of EHR_VARIABLES) {
    const value = values[variable];
    if (value === undefined || value === null || value === '') { out[variable] = null; continue; }
    if (EHR_CATEGORICAL_VARIABLES.includes(variable)) {
      const allowed = EHR_CATEGORICAL_VALUES[variable] ?? [];
      if (typeof value !== 'string' || !allowed.includes(value as never)) throw ApiError.badRequest('INVALID_EHR', 'Invalid value for ' + variable);
      out[variable] = value;
    } else {
      const number = typeof value === 'number' ? value : Number(value);
      if (!Number.isFinite(number)) throw ApiError.badRequest('INVALID_EHR', 'Invalid numeric value for ' + variable);
      out[variable] = number;
    }
  }
  return out;
}

async function ensureAdhocStay(requestedBy: Types.ObjectId) {
  const patient = await PatientModel.create({ pseudoId: nextPseudoId(), demographics: { age: 0, sex: 'M' }, createdBy: requestedBy });
  return StayModel.create({
    patientId: patient._id, ward: 'ADHOC', bedLabel: 'ANALYZE', admittedAt: new Date(),
    status: 'active', availability: { ehr: false, cxr: false, notes: false },
  });
}

export function analyzeRoutes(env: Env, service: PredictionService): Router {
  const router = Router();
  const clinical = requireRole('clinician', 'radiologist', 'admin');

  router.post('/analyze', clinical, asyncHandler(async (req, res) => {
    const multipart = await parseMultipart(req);
    const bodyResult = analyzeRequest.safeParse(multipart.fields);
    if (!bodyResult.success) throw ApiError.badRequest('INVALID_INPUT', 'task is required and must be mortality or pneumonia');

    const task: Task = bodyResult.data.task;
    const requestedBy = new Types.ObjectId(req.user!.id);
    const cxr = multipart.files.find((f) => f.fieldName === 'cxr');
    const ehrFile = multipart.files.find((f) => f.fieldName === 'ehr');
    const noteFiles = multipart.files.filter((f) => f.fieldName === 'notes');
    if (cxr && !isPng(cxr.buffer) && !isJpeg(cxr.buffer)) throw ApiError.badRequest('INVALID_IMAGE', 'CXR must be a JPEG or PNG image');

    const ehrFromField = parseJsonField(multipart.fields.ehrJson, ehrPayload, 'ehrJson');
    const noteField = parseJsonField(multipart.fields.notesJson, z.array(notePayload).max(32), 'notesJson') ?? [];
    if (!cxr && !ehrFile && !ehrFromField && noteFiles.length === 0 && noteField.length === 0) {
      throw ApiError.badRequest('NO_MODALITY', 'At least one modality is required');
    }

    let stay = bodyResult.data.stayId ? await StayModel.findById(bodyResult.data.stayId) : null;
    if (!stay) stay = await ensureAdhocStay(requestedBy);
    const cutoff = new Date();

    if (ehrFromField?.rows) {
      const docs = ehrFromField.rows.map((row) => ({ ts: new Date(row.ts), meta: { stayId: stay!._id, source: 'analyze' }, values: normalizeVitalsValues(row.values) }));
      if (docs.length) { await VitalsModel.insertMany(docs, { ordered: true }); stay.availability.ehr = true; }
    }
    if (ehrFile) {
      const lines = ehrFile.buffer.toString('utf8').split(/\\r?\\n/).filter(Boolean);
      if (lines.length > 49) throw ApiError.badRequest('INVALID_EHR', 'EHR CSV may contain at most 48 data rows');
      const header = lines[0]?.split(',').map((v) => v.trim()) ?? [];
      const timestampColumn = header.findIndex((v) => v === 'ts' || v === 'timestamp');
      if (timestampColumn < 0) throw ApiError.badRequest('INVALID_EHR', 'CSV must include ts or timestamp');
      const docs = lines.slice(1).map((line) => {
        const cells = line.split(',');
        const ts = new Date(cells[timestampColumn]!);
        if (Number.isNaN(ts.getTime())) throw ApiError.badRequest('INVALID_EHR', 'CSV contains an invalid timestamp');
        const values: Record<string, unknown> = {};
        for (let i = 0; i < header.length; i++) if (i !== timestampColumn) values[header[i]!] = cells[i] ?? null;
        return { ts, meta: { stayId: stay!._id, source: 'analyze' }, values: normalizeVitalsValues(values) };
      });
      if (docs.length) { await VitalsModel.insertMany(docs, { ordered: true }); stay.availability.ehr = true; }
    }
    if (cxr) {
      const ext = isJpeg(cxr.buffer) ? 'jpg' : 'png';
      const hash = createHash('sha256').update(cxr.buffer).digest('hex');
      const dir = join(process.cwd(), env.DEMO_DATA_DIR, 'images');
      await mkdir(dir, { recursive: true });
      const fileName = `analyze-${String(stay._id)}-${hash}.${ext}`;
      await writeFile(join(dir, fileName), cxr.buffer);
      await ImageModel.create({ stayId: stay._id, filePath: fileName, takenAt: new Date(), view: 'AP', width: 384, height: 384, sha256: hash, uploadedBy: requestedBy });
      stay.availability.cxr = true;
    }

    const notes = [
      ...noteField,
      ...noteFiles.map((file) => ({ type: (multipart.fields.noteType as NoteType) || 'radiology', text: file.buffer.toString('utf8'), authoredAt: new Date().toISOString() })),
    ];
    for (const note of notes) {
      if (task === 'mortality' && note.type === 'discharge') continue;
      await NoteModel.create({ stayId: stay._id, type: note.type, authoredAt: note.authoredAt ? new Date(note.authoredAt) : new Date(), text: note.text, tokenCount: estimateTokens(note.text) });
      stay.availability.notes = true;
    }
    await stay.save();

    const prediction = await service.request({ stayId: stay._id, task, cutoffTime: cutoff, requestedBy, requestId: req.requestId });
    res.status(prediction.status === 'done' ? 201 : 202);
    ok(res, { predictionId: String(prediction._id), stayId: String(stay._id), status: prediction.status });
  }));
  return router;
}
