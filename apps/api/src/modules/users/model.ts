import { ROLES, type Role } from '@pneumovision/shared';
import { Schema, model, type HydratedDocument, type InferSchemaType } from 'mongoose';

/**
 * Platform users.
 *
 * TODO(phase-3): MFA (`mfa.enabled` / `mfa.secretEnc`), `lastLoginAt`,
 * `failedLoginCount` and `lockedUntil` for account lockout. The demo has
 * password auth only.
 */
const userSchema = new Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
      maxlength: 254,
      match: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    },
    // argon2id. Never selected by default, so a stray `.find()` cannot leak it.
    passwordHash: { type: String, required: true, select: false },
    name: { type: String, required: true, trim: true, maxlength: 120 },
    role: { type: String, required: true, enum: ROLES },
    isActive: { type: Boolean, required: true, default: true },
  },
  { timestamps: true, strict: 'throw' },
);

// Only active users are ever looked up at login.
userSchema.index({ email: 1, isActive: 1 });

export type UserDoc = HydratedDocument<InferSchemaType<typeof userSchema>>;
export const UserModel = model('User', userSchema);

/** Compile-time guard that the schema enum matches the shared Role union. */
const _roleCheck: readonly Role[] = ROLES;
void _roleCheck;
