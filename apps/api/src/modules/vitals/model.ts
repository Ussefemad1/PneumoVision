import {
  EHR_CATEGORICAL_VALUES,
  EHR_CATEGORICAL_VARIABLES,
  EHR_CONTINUOUS_VARIABLES,
  EHR_VARIABLES,
} from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * The 17 raw clinical variables for one hourly bin.
 *
 * Built from the shared constants rather than hand-listed, so it cannot drift
 * from `medpatch/ehr_utils/resources/discretizer_config.json`. Every field is
 * optional and nullable: a missing value means nothing was charted that hour,
 * which the UI renders as a gap and the inference service hands to the
 * discretizer to impute exactly as training does.
 *
 * Continuous variables are numbers; categorical ones are the verbatim MIMIC
 * strings, validated against the allowed value list.
 */
const valuesDefinition: Record<string, unknown> = {};
for (const variable of EHR_CONTINUOUS_VARIABLES) {
  valuesDefinition[variable] = { type: Number, default: null };
}
for (const variable of EHR_CATEGORICAL_VARIABLES) {
  valuesDefinition[variable] = {
    type: String,
    default: null,
    enum: {
      values: [...EHR_CATEGORICAL_VALUES[variable]!, null],
      message: `invalid {PATH}: {VALUE}`,
    },
  };
}

/**
 * Native MongoDB time-series collection (requires MongoDB >= 5.0; the demo
 * pins mongodb-memory-server to 7.x).
 *
 * `timestamps: true` is deliberately off — `ts` *is* the time field, and
 * time-series collections reject the updatedAt writes Mongoose would issue.
 */
const vitalsSchema = new Schema(
  {
    ts: { type: Date, required: true },
    meta: {
      stayId: { type: Schema.Types.ObjectId, ref: 'Stay', required: true },
      source: { type: String, required: true, default: 'seed', maxlength: 40 },
    },
    values: { type: valuesDefinition, required: true },
  },
  {
    timeseries: { timeField: 'ts', metaField: 'meta', granularity: 'hours' },
    // Mongoose must create the collection itself for the time-series options
    // to be applied; an implicitly created one would be an ordinary collection.
    autoCreate: true,
    autoIndex: false,
    strict: 'throw',
    versionKey: false,
  },
);

export type VitalsDoc = HydratedDocument<InferSchemaType<typeof vitalsSchema>>;
export const VitalsModel = model('Vitals', vitalsSchema);

/** Re-exported so seeding and validation share one list. */
export const VITALS_VARIABLES = EHR_VARIABLES;
