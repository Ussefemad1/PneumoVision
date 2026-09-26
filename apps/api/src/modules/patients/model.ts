import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * Pseudonymous patient record. Real identifiers never enter the system — the
 * `pseudoId` (PV-000123) is the only handle clinicians see.
 *
 * TODO(phase-4): replace `demographics` with `demographicsEnc`, an AES-256-GCM
 * envelope carrying { age, sex } plus the key version. Plain for the demo so
 * the seed stays inspectable.
 */
const patientSchema = new Schema(
  {
    pseudoId: {
      type: String,
      required: true,
      unique: true,
      match: /^PV-\d{6}$/,
    },
    demographics: {
      age: { type: Number, required: true, min: 0, max: 120 },
      sex: { type: String, required: true, enum: ['M', 'F'] },
    },
    createdBy: { type: Schema.Types.ObjectId, ref: 'User', default: null },
  },
  { timestamps: true, strict: 'throw' },
);

export type PatientDoc = HydratedDocument<InferSchemaType<typeof patientSchema>>;
export const PatientModel = model('Patient', patientSchema);
