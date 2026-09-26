import { CXR_VIEWS } from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * A chest radiograph. Pixels never live in Mongo — only a pointer.
 *
 * TODO(phase-4): drop `filePath` for `objectKey` against a private MinIO
 * bucket, served through 5-minute presigned URLs. The demo writes synthetic
 * PNGs to a local folder and streams them from the API.
 */
const imageSchema = new Schema(
  {
    stayId: { type: Schema.Types.ObjectId, ref: 'Stay', required: true, index: true },
    filePath: { type: String, required: true, maxlength: 512 },
    objectKey: { type: String, default: null, maxlength: 512 },
    takenAt: { type: Date, required: true },
    view: { type: String, required: true, enum: CXR_VIEWS },
    width: { type: Number, required: true, min: 1 },
    height: { type: Number, required: true, min: 1 },
    sha256: { type: String, required: true, match: /^[0-9a-f]{64}$/ },
    uploadedBy: { type: Schema.Types.ObjectId, ref: 'User', default: null },
  },
  { timestamps: true, strict: 'throw' },
);

// The imaging tab lists a stay's radiographs newest first.
imageSchema.index({ stayId: 1, takenAt: -1 });

export type ImageDoc = HydratedDocument<InferSchemaType<typeof imageSchema>>;
export const ImageModel = model('Image', imageSchema);
