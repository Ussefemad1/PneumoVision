import { ALERT_SEVERITIES, ALERT_STATUSES, TASKS } from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * A risk alert (P3). Raised by the alert engine when a prediction crosses a
 * threshold or rises sharply within a window.
 *
 * TODO(phase-3): acknowledge/resolve transitions must also append to the
 * hash-chained audit log. The demo records who and when on the alert itself.
 */
const alertSchema = new Schema(
  {
    stayId: { type: Schema.Types.ObjectId, ref: 'Stay', required: true, index: true },
    predictionId: { type: Schema.Types.ObjectId, ref: 'Prediction', default: null },
    task: { type: String, required: true, enum: TASKS },
    severity: { type: String, required: true, enum: ALERT_SEVERITIES },
    /** Human-readable description of the rule that fired. */
    rule: { type: String, required: true, maxlength: 200 },
    value: { type: Number, required: true },
    status: { type: String, required: true, enum: ALERT_STATUSES, default: 'open' },
    acknowledgedBy: { type: Schema.Types.ObjectId, ref: 'User', default: null },
    acknowledgedAt: { type: Date, default: null },
    resolvedAt: { type: Date, default: null },
  },
  { timestamps: true, strict: 'throw' },
);

// The alerts centre filters by status and severity; the dashboard counts open
// alerts per stay.
alertSchema.index({ status: 1, severity: 1 });
alertSchema.index({ stayId: 1, status: 1 });

export type AlertDoc = HydratedDocument<InferSchemaType<typeof alertSchema>>;
export const AlertModel = model('Alert', alertSchema);
