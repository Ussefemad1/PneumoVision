import { z } from 'zod';

import {
  ALERT_SEVERITIES,
  ALERT_STATUSES,
  CXR_VIEWS,
  NOTE_TYPES,
  ROLES,
  STAY_STATUSES,
  TASKS,
} from '../constants.js';
import { EHR_VARIABLES } from '../generated/ehr-variables.js';
import { isoDateTime, objectId } from './common.js';

// ── Users / auth ─────────────────────────────────────────────────────────────

export const roleSchema = z.enum(ROLES);

export const loginRequest = z.object({
  email: z.string().email().max(254),
  password: z.string().min(8).max(200),
});
export type LoginRequest = z.infer<typeof loginRequest>;

export const currentUser = z.object({
  id: objectId,
  email: z.string().email(),
  name: z.string(),
  role: roleSchema,
});
export type CurrentUser = z.infer<typeof currentUser>;

// ── Patients ─────────────────────────────────────────────────────────────────

/** Pseudonymous identifier. Real identifiers never enter the system. */
export const pseudoId = z.string().regex(/^PV-\d{6}$/, 'must look like PV-000123');

export const demographics = z.object({
  age: z.number().int().min(0).max(120),
  sex: z.enum(['M', 'F']),
});
export type Demographics = z.infer<typeof demographics>;

export const patient = z.object({
  id: objectId,
  pseudoId,
  demographics,
  createdAt: isoDateTime,
});
export type Patient = z.infer<typeof patient>;

// ── Stays ────────────────────────────────────────────────────────────────────

/** F5: which modalities this stay actually has. Drives the availability badges. */
export const modalityAvailability = z.object({
  ehr: z.boolean(),
  cxr: z.boolean(),
  notes: z.boolean(),
});
export type ModalityAvailability = z.infer<typeof modalityAvailability>;

export const stay = z.object({
  id: objectId,
  patientId: objectId,
  ward: z.string().min(1).max(40),
  bedLabel: z.string().min(1).max(20),
  admittedAt: isoDateTime,
  dischargedAt: isoDateTime.nullable(),
  status: z.enum(STAY_STATUSES),
  availability: modalityAvailability,
});
export type Stay = z.infer<typeof stay>;

/** A stay joined with its patient and latest risk, as the stay page needs it. */
export const stayDetail = stay.extend({
  patient,
  latestPredictions: z.record(z.enum(TASKS), z.number().min(0).max(1).nullable()),
  openAlertCount: z.number().int().min(0),
});
export type StayDetail = z.infer<typeof stayDetail>;

// ── Vitals ───────────────────────────────────────────────────────────────────

/**
 * One hourly bin of the 17 raw clinical variables. Values are nullable: a gap
 * means nothing was charted that hour, and the UI renders it as a gap rather
 * than interpolating (F7). Categorical variables arrive as their verbatim
 * MIMIC strings; the inference service does the discretisation.
 */
export const vitalsValues = z.object(
  Object.fromEntries(
    EHR_VARIABLES.map((v) => [v, z.union([z.number(), z.string()]).nullable().optional()]),
  ) as Record<
    (typeof EHR_VARIABLES)[number],
    z.ZodOptional<z.ZodNullable<z.ZodUnion<[z.ZodNumber, z.ZodString]>>>
  >,
);
export type VitalsValues = z.infer<typeof vitalsValues>;

export const vitalsPoint = z.object({
  ts: isoDateTime,
  values: vitalsValues,
});
export type VitalsPoint = z.infer<typeof vitalsPoint>;

export const vitalsQuery = z.object({
  from: isoDateTime.optional(),
  to: isoDateTime.optional(),
});
export type VitalsQuery = z.infer<typeof vitalsQuery>;

// ── Images ───────────────────────────────────────────────────────────────────

export const image = z.object({
  id: objectId,
  stayId: objectId,
  takenAt: isoDateTime,
  view: z.enum(CXR_VIEWS),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  sha256: z.string().length(64),
  /** Demo: served from the API. Later: a short-lived presigned MinIO URL. */
  url: z.string(),
});
export type Image = z.infer<typeof image>;

// ── Notes ────────────────────────────────────────────────────────────────────

export const noteTypeSchema = z.enum(NOTE_TYPES);

export const note = z.object({
  id: objectId,
  stayId: objectId,
  type: noteTypeSchema,
  authoredAt: isoDateTime,
  text: z.string(),
  /** F9: surfaced so the UI can warn when BioBERT will chunk the note. */
  tokenCount: z.number().int().min(0),
});
export type Note = z.infer<typeof note>;

// ── Alerts ───────────────────────────────────────────────────────────────────

export const alertSeverity = z.enum(ALERT_SEVERITIES);
export const alertStatus = z.enum(ALERT_STATUSES);

export const alert = z.object({
  id: objectId,
  stayId: objectId,
  predictionId: objectId.nullable(),
  task: z.enum(TASKS),
  severity: alertSeverity,
  /** Human-readable description of the rule that fired. */
  rule: z.string(),
  value: z.number(),
  status: alertStatus,
  acknowledgedBy: objectId.nullable(),
  acknowledgedAt: isoDateTime.nullable(),
  resolvedAt: isoDateTime.nullable(),
  createdAt: isoDateTime,
});
export type Alert = z.infer<typeof alert>;

/** Alert rows on the alerts page carry enough context to act without a fetch. */
export const alertWithContext = alert.extend({
  stay: z.object({
    id: objectId,
    ward: z.string(),
    bedLabel: z.string(),
    pseudoId,
  }),
});
export type AlertWithContext = z.infer<typeof alertWithContext>;

export const alertsQuery = z.object({
  status: alertStatus.optional(),
  severity: alertSeverity.optional(),
  stayId: objectId.optional(),
});
export type AlertsQuery = z.infer<typeof alertsQuery>;
