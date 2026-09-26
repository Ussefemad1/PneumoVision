import { z } from 'zod';

import { MODALITIES, PREDICTION_STATUSES, TASKS } from '../constants.js';
import { EHR_WINDOW_HOURS } from '../generated/ehr-variables.js';
import { isoDateTime, objectId } from './common.js';

/** A calibrated probability. Only ever displayed after temperature scaling (F3). */
export const probability = z.number().min(0).max(1);

/**
 * An optional probability that always parses to `number | null`, never
 * `undefined`. The inference service omits absent branches entirely; the
 * database and the UI both want an explicit null, so normalise once here
 * rather than at every call site.
 */
const nullableProbability = probability.nullish().transform((v) => v ?? null);

const nullableNumber = z
  .number()
  .nullish()
  .transform((v) => v ?? null);

export const taskSchema = z.enum(TASKS);
export const modalitySchema = z.enum(MODALITIES);

// ── F1: per-modality predictions ─────────────────────────────────────────────

export const unimodalScores = z.object({
  ehr: nullableProbability,
  cxr: nullableProbability,
  rr: nullableProbability,
  dn: nullableProbability,
});
export type UnimodalScores = z.infer<typeof unimodalScores>;

// ── F4: confidence-based patching ────────────────────────────────────────────

export const jointScores = z.object({
  high: probability,
  low: probability,
});
export type JointScores = z.infer<typeof jointScores>;

// ── F5: missingness module ───────────────────────────────────────────────────

export const missingness = z.object({
  vector: z.object({
    ehr: z.boolean(),
    cxr: z.boolean(),
    rr: z.boolean(),
    dn: z.boolean(),
  }),
  prediction: probability,
});
export type Missingness = z.infer<typeof missingness>;

// ── F6: late-fusion α weights ────────────────────────────────────────────────

/**
 * Softmax-normalised weights over the branches that contributed. Branches for
 * absent modalities are null, not zero — the fusion masks them out entirely,
 * and the contribution chart must not imply they were weighed and discarded.
 */
export const alphas = z.object({
  high: z.number(),
  low: z.number(),
  miss: z.number(),
  ehr: nullableNumber,
  cxr: nullableNumber,
  rr: nullableNumber,
  dn: nullableNumber,
});
export type Alphas = z.infer<typeof alphas>;

// ── F2: token-level confidence maps ──────────────────────────────────────────

export const noteSpan = z.object({
  noteId: z.string(),
  start: z.number().int().min(0),
  end: z.number().int().min(0),
  confidence: probability,
});
export type NoteSpan = z.infer<typeof noteSpan>;

export const confidenceMaps = z.object({
  theta: probability,
  fractionAbove: z.object({
    ehr: nullableProbability,
    cxr: nullableProbability,
    rr: nullableProbability,
    dn: nullableProbability,
  }),
  /** One confidence per hour of the 48-hour window. */
  ehrTimesteps: z
    .array(probability)
    .length(EHR_WINDOW_HOURS)
    .nullish()
    .transform((v) => v ?? null),
  /** Square ViT patch grid, row-major. CLS token excluded. */
  cxrPatchGrid: z
    .array(z.array(probability))
    .nullish()
    .transform((v) => v ?? null),
  noteSpans: z
    .array(noteSpan)
    .nullish()
    .transform((v) => v ?? null),
});
export type ConfidenceMaps = z.infer<typeof confidenceMaps>;

// ── The full result ──────────────────────────────────────────────────────────

export const predictionResult = z.object({
  /** The final calibrated ŷ_late shown as the headline number. */
  probability,
  unimodal: unimodalScores,
  joint: jointScores,
  missingness,
  alphas,
  confidence: confidenceMaps,
});
export type PredictionResult = z.infer<typeof predictionResult>;

export const predictionStatus = z.enum(PREDICTION_STATUSES);

export const prediction = z.object({
  id: objectId,
  stayId: objectId,
  task: taskSchema,
  cutoffTime: isoDateTime,
  status: predictionStatus,
  modelVersion: z.string(),
  /** Hash of the exact inputs, so a prediction stays reproducible (P6). */
  inputHash: z.string(),
  latencyMs: z.number().int().min(0).nullable(),
  requestedBy: objectId.nullable(),
  result: predictionResult.nullable(),
  error: z.string().nullable(),
  createdAt: isoDateTime,
});
export type Prediction = z.infer<typeof prediction>;

export const createPredictionRequest = z.object({
  task: taskSchema,
  /** Defaults to now. Only data strictly before this instant is sent (F8). */
  cutoffTime: isoDateTime.optional(),
});
export type CreatePredictionRequest = z.infer<typeof createPredictionRequest>;

export const predictionsQuery = z.object({
  task: taskSchema.optional(),
});

// ── P2: risk trajectory ──────────────────────────────────────────────────────

export const trajectoryPoint = z.object({
  predictionId: objectId,
  cutoffTime: isoDateTime,
  probability,
});
export type TrajectoryPoint = z.infer<typeof trajectoryPoint>;

export const riskTrajectory = z.object({
  task: taskSchema,
  points: z.array(trajectoryPoint),
  /** Markers for when a CXR or note arrived, drawn on the trajectory chart. */
  events: z.array(
    z.object({
      at: isoDateTime,
      kind: z.enum(['cxr', 'note']),
      label: z.string(),
    }),
  ),
  threshold: probability,
});
export type RiskTrajectory = z.infer<typeof riskTrajectory>;

// ── P4: ward dashboard ───────────────────────────────────────────────────────

export const wardRow = z.object({
  stayId: objectId,
  pseudoId: z.string(),
  ward: z.string(),
  bedLabel: z.string(),
  age: z.number().int(),
  sex: z.enum(['M', 'F']),
  availability: z.object({ ehr: z.boolean(), cxr: z.boolean(), notes: z.boolean() }),
  mortality: probability.nullable(),
  pneumonia: probability.nullable(),
  /** Recent mortality probabilities, oldest first, for the sparkline. */
  sparkline: z.array(probability),
  openAlerts: z.number().int().min(0),
  lastScoredAt: isoDateTime.nullable(),
});
export type WardRow = z.infer<typeof wardRow>;

export const wardDashboard = z.object({
  wards: z.array(z.string()),
  rows: z.array(wardRow),
});
export type WardDashboard = z.infer<typeof wardDashboard>;

// ── P1: ICU replay ───────────────────────────────────────────────────────────

export const simulationStatus = z.object({
  stayId: objectId,
  running: z.boolean(),
  speed: z.number().int().positive(),
  /** Hours of the stay replayed so far. */
  cursorHour: z.number().int().min(0),
  totalHours: z.number().int().min(0),
});
export type SimulationStatus = z.infer<typeof simulationStatus>;

export const startSimulationRequest = z.object({
  speed: z.number().int().positive().max(240).default(60),
});
export type StartSimulationRequest = z.infer<typeof startSimulationRequest>;
