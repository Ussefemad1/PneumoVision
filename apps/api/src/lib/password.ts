import { hash, verify } from '@node-rs/argon2';

/**
 * Password hashing with argon2id (Section 7).
 *
 * Parameters follow the OWASP baseline: 19 MiB, 2 iterations, parallelism 1.
 * @node-rs/argon2 ships prebuilt binaries, so a clean clone needs no compiler.
 */
const OPTIONS = {
  // 2 = Argon2id in @node-rs/argon2's Algorithm enum.
  algorithm: 2 as const,
  memoryCost: 19_456,
  timeCost: 2,
  parallelism: 1,
};

export async function hashPassword(plain: string): Promise<string> {
  return hash(plain, OPTIONS);
}

/** Constant-time verification; never throws on a malformed stored hash. */
export async function verifyPassword(storedHash: string, plain: string): Promise<boolean> {
  try {
    return await verify(storedHash, plain, OPTIONS);
  } catch {
    return false;
  }
}
