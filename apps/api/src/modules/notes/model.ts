import { NOTE_TYPES } from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * A clinical note.
 *
 * `type` decides both which encoder the text reaches (discharge -> DN, the
 * rest -> RR) and whether it may be sent at all: discharge notes leak the
 * outcome and are never included in a mortality prediction (F8).
 *
 * TODO(phase-4): replace `text` with `textEnc` (AES-256-GCM + key version) and
 * gate reads behind the object-level check that hides note text from the
 * researcher role. Plain text for the demo.
 */
const noteSchema = new Schema(
  {
    stayId: { type: Schema.Types.ObjectId, ref: 'Stay', required: true, index: true },
    type: { type: String, required: true, enum: NOTE_TYPES },
    authoredAt: { type: Date, required: true },
    text: { type: String, required: true, maxlength: 100_000 },
    // F9: surfaced so the notes UI can warn when BioBERT will chunk the text.
    tokenCount: { type: Number, required: true, min: 0 },
  },
  { timestamps: true, strict: 'throw' },
);

noteSchema.index({ stayId: 1, authoredAt: -1 });
// The leakage filter selects by type and time on every prediction.
noteSchema.index({ stayId: 1, type: 1, authoredAt: 1 });

export type NoteDoc = HydratedDocument<InferSchemaType<typeof noteSchema>>;
export const NoteModel = model('Note', noteSchema);
