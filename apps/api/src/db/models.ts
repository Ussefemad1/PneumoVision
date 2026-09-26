/**
 * Importing this module registers every Mongoose model exactly once. Anything
 * that needs a model — routes, the seed, tests — imports from here so
 * registration order is deterministic.
 */
export { UserModel, type UserDoc } from '../modules/users/model.js';
export { PatientModel, type PatientDoc } from '../modules/patients/model.js';
export { StayModel, type StayDoc } from '../modules/stays/model.js';
export { VitalsModel, type VitalsDoc } from '../modules/vitals/model.js';
export { ImageModel, type ImageDoc } from '../modules/imaging/model.js';
export { NoteModel, type NoteDoc } from '../modules/notes/model.js';
export { PredictionModel, type PredictionDoc } from '../modules/predictions/model.js';
export { AlertModel, type AlertDoc } from '../modules/alerts/model.js';
