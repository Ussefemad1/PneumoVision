import { randomInt } from 'node:crypto';

import {
  ADHOC_WARD,
  ANALYZE_LIMITS,
  EHR_WINDOW_HOURS,
  NOTE_TYPES,
  NOTE_TYPES_BY_TASK,
  analyzeEhrJson,
  analyzeFields,
  analyzeNote,
  estimateTokens,
  parseEhrCsv,
  validateEhrHour,
  validateVitalsValues,
  type AnalyzeExcluded,
  type AnalyzeResponse,
  type EhrIssue,
  type NoteType,
  type Task,
  type VitalsInput,
} from '@pneumovision/shared';
import { Router, type Request, type Response } from 'express';
import { rateLimit } from 'express-rate-limit';
import { Types } from 'mongoose';
import { z } from 'zod';

import type { Env } from '../../config/env.js';
import { ImageModel, NoteModel, PatientModel, StayModel, VitalsModel } from '../../db/models.js';
import { asyncHandler, ok } from '../../lib/http.js';
import { sha256Hex, sniffImage, writeImage, type SniffedImage } from '../../lib/imageStore.js';
import { readMultipart, type MultipartFile, type MultipartSpec } from '../../lib/multipart.js';
import { requireRole } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';
import type { PredictionService } from '../../services/predictionService.js';

/**
 * `POST /analyze` — score whatever evidence a clinician has to hand.
 *
 * Accepts any combination of a chest X-ray, EHR rows (JSON or CSV) and notes
 * (JSON or .txt files), stores them against an existing stay or a fresh
 * ad-hoc one, and requests a prediction through the same `PredictionService`
 * the stay page uses — so F8 filtering, missingness handling and the report
 * are identical.
 *
 * Timing: the submission instant T is captured once. Uploaded images and
 * notes are stamped T, EHR rows after T are rejected, and the cutoff is
 * T + 1 ms, so everything submitted satisfies `gatherInputs`' strictly-before
 * rule without loosening it.
 */

const MULTIPART: MultipartSpec = {
  files: {
    cxr: {
      maxBytes: ANALYZE_LIMITS.cxrBytes,
      maxCount: 1,
      tooLargeCode: 'CXR_TOO_LARGE',
      tooManyCode: 'TOO_MANY_CXR',
    },
    ehr: {
      maxBytes: ANALYZE_LIMITS.ehrCsvBytes,
      maxCount: 1,
      tooLargeCode: 'EHR_CSV_TOO_LARGE',
      tooManyCode: 'TOO_MANY_EHR_FILES',
    },
    notes: {
      maxBytes: ANALYZE_LIMITS.noteFileBytes,
      maxCount: ANALYZE_LIMITS.noteFiles,
      tooLargeCode: 'NOTE_FILE_TOO_LARGE',
      tooManyCode: 'TOO_MANY_NOTES',
    },
  },
  fieldBytes: ANALYZE_LIMITS.fieldBytes,
  maxFields: 8,
  totalBytes:
    ANALYZE_LIMITS.cxrBytes +
    ANALYZE_LIMITS.ehrCsvBytes +
    ANALYZE_LIMITS.noteFiles * ANALYZE_LIMITS.noteFileBytes +
    4 * ANALYZE_LIMITS.fieldBytes,
};

const HOUR_MS = 3_600_000;

// ── Field parsing ────────────────────────────────────────────────────────────

function parseJsonField<T>(raw: string | undefined, schema: z.ZodType<T>, field: string) {
  if (raw === undefined || raw.trim() === '') return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw ApiError.badRequest('INVALID_JSON', `${field} must contain valid JSON`);
  }
  const result = schema.safeParse(parsed);
  if (!result.success) {
    const issue = result.error.issues[0];
    throw ApiError.badRequest(
      'INVALID_INPUT',
      `${field} is invalid${issue ? ` at ${issue.path.join('.') || '(root)'}: ${issue.message}` : ''}`,
    );
  }
  return result.data;
}

const ehrError = (issue: EhrIssue, where: string) =>
  ApiError.badRequest(issue.code, `${where}: ${issue.message}`);

// ── EHR ──────────────────────────────────────────────────────────────────────

interface EhrRow {
  ts: Date;
  values: VitalsInput;
}

/**
 * Resolves JSON and CSV rows to timestamps against T, validating each with
 * the shared validator the page's CSV preview also uses.
 */
function collectEhrRows(
  fields: Record<string, string>,
  file: MultipartFile | undefined,
  submittedAt: Date,
): EhrRow[] {
  const rows: EhrRow[] = [];
  const t = submittedAt.getTime();

  const place = (time: { hour: number } | { ts: string }, values: VitalsInput, where: string) => {
    let ts: Date;
    if ('hour' in time) {
      const issue = validateEhrHour(time.hour);
      if (issue) throw ehrError(issue, where);
      ts = new Date(t + time.hour * HOUR_MS);
    } else {
      ts = new Date(time.ts);
      if (ts.getTime() > t) {
        throw ehrError(
          { code: 'EHR_FUTURE_TIMESTAMP', message: `${time.ts} is after submission` },
          where,
        );
      }
      if (ts.getTime() <= t - EHR_WINDOW_HOURS * HOUR_MS) {
        throw ehrError(
          {
            code: 'EHR_OUTSIDE_WINDOW',
            message: `${time.ts} is outside the ${EHR_WINDOW_HOURS}-hour window`,
          },
          where,
        );
      }
    }
    rows.push({ ts, values });
  };

  const json = parseJsonField(fields.ehrJson, analyzeEhrJson, 'ehrJson');
  const parsed = file ? parseEhrCsv(file.buffer.toString('utf8')) : null;
  if (parsed?.headerError) throw ehrError(parsed.headerError, 'EHR CSV header');

  // Count first, so an over-long file gets the limit error, not a window
  // error from its 49th row.
  if ((json?.rows.length ?? 0) + (parsed?.rows.length ?? 0) > ANALYZE_LIMITS.ehrRows) {
    throw ApiError.badRequest(
      'EHR_TOO_MANY_ROWS',
      `At most ${ANALYZE_LIMITS.ehrRows} hourly EHR rows are accepted`,
    );
  }

  json?.rows.forEach((row, i) => {
    const validated = validateVitalsValues(row.values);
    if (!validated.ok) throw ehrError(validated.errors[0]!, `ehrJson row ${i + 1}`);
    place(
      'hour' in row ? { hour: row.hour } : { ts: row.ts },
      validated.values,
      `ehrJson row ${i + 1}`,
    );
  });

  if (parsed) {
    for (const row of parsed.rows) {
      if (row.errors.length > 0) throw ehrError(row.errors[0]!, `EHR CSV line ${row.line}`);
      place(row.time!, row.values, `EHR CSV line ${row.line}`);
    }
  }

  // Two rows in one hourly bin would silently overwrite each other.
  const windowStart = t + 1 - EHR_WINDOW_HOURS * HOUR_MS;
  const bins = new Set<number>();
  for (const row of rows) {
    const bin = Math.floor((row.ts.getTime() - windowStart) / HOUR_MS);
    if (bins.has(bin)) {
      throw ApiError.badRequest(
        'EHR_DUPLICATE_HOUR',
        `Two EHR rows fall in the same hour (${row.ts.toISOString()})`,
      );
    }
    bins.add(bin);
  }

  // A row with every cell blank is a gap, not data.
  return rows.filter((r) => Object.keys(r.values).length > 0);
}

// ── Notes ────────────────────────────────────────────────────────────────────

interface IncomingNote {
  type: NoteType;
  text: string;
}

const NOTE_MAX_CHARS = ANALYZE_LIMITS.noteFileBytes;

function collectNotes(
  task: Task,
  fields: Record<string, string>,
  files: MultipartFile[],
): { accepted: IncomingNote[]; excluded: AnalyzeExcluded[] } {
  const fromJson =
    parseJsonField(
      fields.notesJson,
      z
        .array(analyzeNote.extend({ text: z.string().trim().min(1).max(NOTE_MAX_CHARS) }))
        .max(ANALYZE_LIMITS.noteFiles),
      'notesJson',
    ) ?? [];

  const types = parseJsonField(fields.noteTypes, z.array(z.enum(NOTE_TYPES)), 'noteTypes') ?? [];
  if (types.length !== files.length) {
    throw ApiError.badRequest(
      'NOTE_TYPES_MISMATCH',
      `noteTypes has ${types.length} entr${types.length === 1 ? 'y' : 'ies'} for ${files.length} note file(s)`,
    );
  }
  const fromFiles = files.map((file, i): IncomingNote => {
    const text = file.buffer.toString('utf8').trim();
    if (!text) throw ApiError.badRequest('NOTE_EMPTY', `Note file "${file.fileName}" is empty`);
    return { type: types[i]!, text };
  });

  // F8: discharge notes describe the outcome; for mortality they are neither
  // stored nor sent. gatherInputs would drop them anyway — not storing them
  // keeps a leak from ever being one query away.
  const permitted = NOTE_TYPES_BY_TASK[task];
  const accepted: IncomingNote[] = [];
  const excluded: AnalyzeExcluded[] = [];
  const sort = (note: IncomingNote, source: AnalyzeExcluded['source'], index: number) => {
    if (permitted.includes(note.type)) accepted.push(note);
    else excluded.push({ source, index, type: note.type, reason: 'outcome-leakage' });
  };
  fromJson.forEach((note, i) => sort(note, 'notesJson', i));
  fromFiles.forEach((note, i) => sort(note, 'notes', i));
  return { accepted, excluded };
}

// ── Ad-hoc stays ─────────────────────────────────────────────────────────────

const randomPseudoId = () => `PV-${randomInt(100_000, 1_000_000)}`;

function isDuplicateKey(err: unknown): boolean {
  return typeof err === 'object' && err !== null && (err as { code?: unknown }).code === 11000;
}

/**
 * A patient for inputs that belong to no admitted stay. No demographics:
 * nothing is known, so nothing is invented. The pseudoId relies on the unique
 * index and retries on collision rather than hoping randomness suffices.
 */
export async function createAdhocPatient(
  createdBy: Types.ObjectId | null,
  nextPseudoId: () => string = randomPseudoId,
) {
  await PatientModel.init(); // the unique index must exist for the retry to mean anything
  for (let attempt = 0; attempt < 8; attempt++) {
    try {
      return await PatientModel.create({ pseudoId: nextPseudoId(), createdBy });
    } catch (err) {
      if (!isDuplicateKey(err)) throw err;
    }
  }
  throw new Error('could not allocate a unique pseudoId');
}

async function createAdhocStay(createdBy: Types.ObjectId, admittedAt: Date) {
  const patient = await createAdhocPatient(createdBy);
  return StayModel.create({
    patientId: patient._id,
    ward: ADHOC_WARD,
    bedLabel: 'ANALYZE',
    admittedAt,
    status: 'adhoc',
    availability: { ehr: false, cxr: false, notes: false },
  });
}

// ── Route ────────────────────────────────────────────────────────────────────

export interface AnalyzeOptions {
  /** Analyze requests per user per minute. */
  rateLimitPerMinute?: number;
}

export function analyzeRoutes(
  env: Env,
  service: PredictionService,
  { rateLimitPerMinute = 10 }: AnalyzeOptions = {},
): Router {
  const router = Router();
  const clinical = requireRole('clinician', 'radiologist', 'admin');

  // Per user, not per IP: clinicians behind one hospital NAT share an IP.
  const limiter = rateLimit({
    windowMs: 60_000,
    limit: rateLimitPerMinute,
    standardHeaders: 'draft-8',
    legacyHeaders: false,
    keyGenerator: (req: Request) => req.user?.id ?? 'anonymous',
    handler: (req: Request, res: Response) => {
      res.status(429).json({
        error: {
          code: 'RATE_LIMITED',
          message: `At most ${rateLimitPerMinute} analyses per minute — try again shortly`,
          requestId: req.requestId,
        },
      });
    },
  });

  router.post(
    '/analyze',
    clinical,
    limiter,
    asyncHandler(async (req, res) => {
      const submittedAt = new Date();
      const { fields, files } = await readMultipart(req, MULTIPART);

      const parsedFields = analyzeFields.safeParse(fields);
      if (!parsedFields.success) {
        throw ApiError.badRequest(
          'INVALID_INPUT',
          'task must be mortality or pneumonia, and stayId (if given) a valid id',
        );
      }
      const { task, stayId } = parsedFields.data;
      const requestedBy = new Types.ObjectId(req.user!.id);

      // ── Validate everything before storing anything ──────────────────────
      const cxrFile = files.find((f) => f.fieldName === 'cxr');
      let image: SniffedImage | null = null;
      if (cxrFile) {
        image = sniffImage(cxrFile.buffer);
        if (!image) {
          throw ApiError.badRequest('INVALID_IMAGE', 'The chest X-ray must be a valid PNG or JPEG');
        }
      }

      const ehrRows = collectEhrRows(
        fields,
        files.find((f) => f.fieldName === 'ehr'),
        submittedAt,
      );
      const { accepted: notes, excluded } = collectNotes(
        task,
        fields,
        files.filter((f) => f.fieldName === 'notes'),
      );

      if (!image && ehrRows.length === 0 && notes.length === 0) {
        throw ApiError.badRequest(
          'NO_MODALITY',
          excluded.length > 0
            ? 'Nothing left to analyze: discharge notes are excluded from mortality predictions'
            : 'Provide at least one of: chest X-ray, EHR rows, or a clinical note',
        );
      }

      const existing = stayId ? await StayModel.findById(stayId) : null;
      if (stayId && !existing) throw ApiError.notFound('Stay');
      const stay = existing ?? (await createAdhocStay(requestedBy, submittedAt));

      // ── Persist, all stamped at T ────────────────────────────────────────
      if (ehrRows.length > 0) {
        await VitalsModel.insertMany(
          ehrRows.map((row) => ({
            ts: row.ts,
            meta: { stayId: stay._id, source: 'analyze' },
            values: row.values,
          })),
          { ordered: true },
        );
        stay.set('availability.ehr', true);
      }

      if (cxrFile && image) {
        const sha256 = sha256Hex(cxrFile.buffer);
        const ext = image.contentType === 'image/jpeg' ? 'jpg' : 'png';
        const filePath = `analyze-${String(stay._id)}-${sha256}.${ext}`;
        await writeImage(env, filePath, cxrFile.buffer);
        await ImageModel.create({
          stayId: stay._id,
          filePath,
          takenAt: submittedAt,
          // Unknown for an upload; AP is the usual ICU portable view.
          view: 'AP',
          width: image.width,
          height: image.height,
          sha256,
          uploadedBy: requestedBy,
        });
        stay.set('availability.cxr', true);
      }

      if (notes.length > 0) {
        await NoteModel.insertMany(
          notes.map((note) => ({
            stayId: stay._id,
            type: note.type,
            authoredAt: submittedAt,
            text: note.text,
            tokenCount: estimateTokens(note.text),
          })),
        );
        stay.set('availability.notes', true);
      }
      await stay.save();

      // T + 1 ms: everything stamped T is strictly before the cutoff.
      const cutoffTime = new Date(submittedAt.getTime() + 1);
      const prediction = await service.request({
        stayId: stay._id,
        task,
        cutoffTime,
        requestedBy,
        requestId: req.requestId,
        excluded: excluded.map((e) => ({ source: 'upload', type: e.type, reason: e.reason })),
      });

      const body: AnalyzeResponse = {
        predictionId: String(prediction._id),
        stayId: String(stay._id),
        stayStatus: stay.status,
        status: prediction.status,
        excluded,
      };
      res.status(202);
      ok(res, body);
    }),
  );

  return router;
}
