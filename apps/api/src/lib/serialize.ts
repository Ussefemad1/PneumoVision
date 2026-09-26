import type {
  Alert,
  Image,
  Note,
  Patient,
  Prediction,
  Stay,
  VitalsPoint,
} from '@pneumovision/shared';

/**
 * Mongoose documents to API shapes.
 *
 * Kept in one place so `_id` never leaks as `_id`, dates are always ISO
 * strings, and nothing accidentally serialises a field the client should not
 * see. Each function takes a `.lean()` result.
 */

type Lean = Record<string, unknown>;

const id = (v: unknown): string => String(v);
/** Stringify a scalar field without risking "[object Object]". */
const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const iso = (v: unknown): string =>
  v instanceof Date ? v.toISOString() : new Date(String(v)).toISOString();
const isoOrNull = (v: unknown): string | null => (v === null || v === undefined ? null : iso(v));

export function serializePatient(doc: Lean): Patient {
  const demographics = doc.demographics as { age: number; sex: 'M' | 'F' };
  return {
    id: id(doc._id),
    pseudoId: String(doc.pseudoId),
    demographics: { age: demographics.age, sex: demographics.sex },
    createdAt: iso(doc.createdAt),
  };
}

export function serializeStay(doc: Lean): Stay {
  const availability = doc.availability as { ehr: boolean; cxr: boolean; notes: boolean };
  return {
    id: id(doc._id),
    patientId: id(doc.patientId),
    ward: String(doc.ward),
    bedLabel: String(doc.bedLabel),
    admittedAt: iso(doc.admittedAt),
    dischargedAt: isoOrNull(doc.dischargedAt),
    status: doc.status as Stay['status'],
    availability,
  };
}

export function serializeVitals(doc: Lean): VitalsPoint {
  return {
    ts: iso(doc.ts),
    values: doc.values ?? {},
  };
}

/** `urlFor` builds the link the client should fetch the pixels from. */
export function serializeImage(doc: Lean, urlFor: (imageId: string) => string): Image {
  return {
    id: id(doc._id),
    stayId: id(doc.stayId),
    takenAt: iso(doc.takenAt),
    view: doc.view as Image['view'],
    width: Number(doc.width),
    height: Number(doc.height),
    sha256: String(doc.sha256),
    url: urlFor(id(doc._id)),
  };
}

export function serializeNote(doc: Lean): Note {
  return {
    id: id(doc._id),
    stayId: id(doc.stayId),
    type: doc.type as Note['type'],
    authoredAt: iso(doc.authoredAt),
    text: String(doc.text),
    tokenCount: Number(doc.tokenCount),
  };
}

export function serializePrediction(doc: Lean): Prediction {
  return {
    id: id(doc._id),
    stayId: id(doc.stayId),
    task: doc.task as Prediction['task'],
    cutoffTime: iso(doc.cutoffTime),
    status: doc.status as Prediction['status'],
    modelVersion: String(doc.modelVersion),
    inputHash: str(doc.inputHash),
    latencyMs: doc.latencyMs === null || doc.latencyMs === undefined ? null : Number(doc.latencyMs),
    requestedBy: doc.requestedBy ? id(doc.requestedBy) : null,
    result: (doc.result ?? null) as Prediction['result'],
    error: (doc.error ?? null) as string | null,
    createdAt: iso(doc.createdAt),
  };
}

export function serializeAlert(doc: Lean): Alert {
  return {
    id: id(doc._id),
    stayId: id(doc.stayId),
    predictionId: doc.predictionId ? id(doc.predictionId) : null,
    task: doc.task as Alert['task'],
    severity: doc.severity as Alert['severity'],
    rule: String(doc.rule),
    value: Number(doc.value),
    status: doc.status as Alert['status'],
    acknowledgedBy: doc.acknowledgedBy ? id(doc.acknowledgedBy) : null,
    acknowledgedAt: isoOrNull(doc.acknowledgedAt),
    resolvedAt: isoOrNull(doc.resolvedAt),
    createdAt: iso(doc.createdAt),
  };
}
