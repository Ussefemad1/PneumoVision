import { STAY_STATUSES } from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * An ICU stay. `availability` drives the F5 modality badges and is kept in
 * sync whenever an image or note is added.
 */
const staySchema = new Schema(
  {
    patientId: { type: Schema.Types.ObjectId, ref: 'Patient', required: true, index: true },
    ward: { type: String, required: true, trim: true, maxlength: 40 },
    bedLabel: { type: String, required: true, trim: true, maxlength: 20 },
    admittedAt: { type: Date, required: true },
    dischargedAt: { type: Date, default: null },
    status: { type: String, required: true, enum: STAY_STATUSES, default: 'active' },
    availability: {
      ehr: { type: Boolean, required: true, default: false },
      cxr: { type: Boolean, required: true, default: false },
      notes: { type: Boolean, required: true, default: false },
    },
  },
  { timestamps: true, strict: 'throw' },
);

// The ward dashboard filters active stays by ward.
staySchema.index({ status: 1, ward: 1 });

export type StayDoc = HydratedDocument<InferSchemaType<typeof staySchema>>;
export const StayModel = model('Stay', staySchema);
