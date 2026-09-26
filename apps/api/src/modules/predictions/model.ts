import { MODALITIES, PREDICTION_STATUSES, TASKS } from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

const probability = { type: Number, min: 0, max: 1 } as const;

/** F2: token-level confidence maps, kept so a report can be re-rendered later. */
const confidenceSchema = new Schema(
  {
    theta: { ...probability, required: true },
    fractionAbove: {
      ehr: { ...probability, default: null },
      cxr: { ...probability, default: null },
      rr: { ...probability, default: null },
      dn: { ...probability, default: null },
    },
    ehrTimesteps: { type: [Number], default: null },
    cxrPatchGrid: { type: [[Number]], default: null },
    noteSpans: {
      type: [
        new Schema(
          {
            noteId: { type: String, required: true },
            start: { type: Number, required: true, min: 0 },
            end: { type: Number, required: true, min: 0 },
            confidence: { ...probability, required: true },
          },
          { _id: false },
        ),
      ],
      default: null,
    },
  },
  { _id: false },
);

/**
 * The complete MedPatch output: every intermediate the paper defines, not just
 * the final probability. The report page is built directly from this.
 */
const resultSchema = new Schema(
  {
    // ŷ_late — the calibrated headline number.
    probability: { ...probability, required: true },
    // F1: each modality's own prediction.
    unimodal: {
      ehr: { ...probability, default: null },
      cxr: { ...probability, default: null },
      rr: { ...probability, default: null },
      dn: { ...probability, default: null },
    },
    // F4: the confidence-patched high/low groups.
    joint: {
      high: { ...probability, required: true },
      low: { ...probability, required: true },
    },
    // F5: the missingness indicator vector and its branch prediction.
    missingness: {
      vector: {
        ehr: { type: Boolean, required: true },
        cxr: { type: Boolean, required: true },
        rr: { type: Boolean, required: true },
        dn: { type: Boolean, required: true },
      },
      prediction: { ...probability, required: true },
    },
    // F6: softmax-normalised late-fusion weights. Null for absent branches.
    alphas: {
      high: { type: Number, required: true },
      low: { type: Number, required: true },
      miss: { type: Number, required: true },
      ehr: { type: Number, default: null },
      cxr: { type: Number, default: null },
      rr: { type: Number, default: null },
      dn: { type: Number, default: null },
    },
    confidence: { type: confidenceSchema, required: true },
  },
  { _id: false },
);

const predictionSchema = new Schema(
  {
    stayId: { type: Schema.Types.ObjectId, ref: 'Stay', required: true },
    task: { type: String, required: true, enum: TASKS },
    /** Only data strictly before this instant is sent to the model (F8). */
    cutoffTime: { type: Date, required: true },
    status: { type: String, required: true, enum: PREDICTION_STATUSES, default: 'queued' },
    modelVersion: { type: String, required: true, default: 'unknown' },
    /**
     * P6: hash of the exact inputs, so an old prediction stays reproducible.
     * Not `required`: it is only known once inputs have been gathered, and
     * Mongoose treats the empty string as a missing value.
     */
    inputHash: { type: String, default: '' },
    latencyMs: { type: Number, default: null, min: 0 },
    requestedBy: { type: Schema.Types.ObjectId, ref: 'User', default: null },
    /** Which modalities were actually sent, for the reduced-input banner. */
    inputsUsed: {
      type: [String],
      enum: MODALITIES,
      default: [],
    },
    result: { type: resultSchema, default: null },
    error: { type: String, default: null },
  },
  { timestamps: true, strict: 'throw' },
);

// History tables and the risk trajectory both read newest-first per task.
predictionSchema.index({ stayId: 1, task: 1, cutoffTime: -1 });

export type PredictionDoc = HydratedDocument<InferSchemaType<typeof predictionSchema>>;
export const PredictionModel = model('Prediction', predictionSchema);
