import { randomBytes } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ROLES, type Role } from '@pneumovision/shared';

import { loadEnv } from '../config/env.js';
import { connectMongo, disconnectMongo } from '../db/connect.js';
import { createLogger } from '../lib/logger.js';
import { hashPassword } from '../lib/password.js';
import { UserModel } from '../modules/users/model.js';

/**
 * Account administration from the server shell. Production has no public
 * sign-up and no user-management screen yet, so the seeded admin adds people
 * here (infra/aws/README.md, "Add a user"):
 *
 *   docker compose exec api node apps/api/dist/cli/users.js create \
 *     --email dr.a@example.org --name "Dr A" --role clinician
 *   docker compose exec api node apps/api/dist/cli/users.js reset-password --email dr.a@example.org
 *   docker compose exec api node apps/api/dist/cli/users.js deactivate --email dr.a@example.org
 *
 * A password is generated (never taken from the command line, so it stays out
 * of shell history) and printed once.
 */

export function generatePassword(): string {
  // 18 random bytes -> 24 URL-safe characters (~144 bits).
  return randomBytes(18).toString('base64url');
}

function isRole(value: string): value is Role {
  return (ROLES as readonly string[]).includes(value);
}

export async function createUser(input: {
  email: string;
  name: string;
  role: string;
}): Promise<string> {
  if (!isRole(input.role)) throw new Error(`role must be one of ${ROLES.join(', ')}`);
  if (await UserModel.exists({ email: input.email.toLowerCase().trim() }))
    throw new Error(`a user with email ${input.email} already exists`);
  const password = generatePassword();
  await UserModel.create({
    email: input.email,
    name: input.name,
    role: input.role,
    passwordHash: await hashPassword(password),
  });
  return password;
}

export async function resetPassword(email: string): Promise<string> {
  const password = generatePassword();
  const res = await UserModel.updateOne(
    { email: email.toLowerCase().trim() },
    { $set: { passwordHash: await hashPassword(password), isActive: true } },
  );
  if (res.matchedCount === 0) throw new Error(`no user with email ${email}`);
  return password;
}

export async function deactivateUser(email: string): Promise<void> {
  const res = await UserModel.updateOne(
    { email: email.toLowerCase().trim() },
    { $set: { isActive: false } },
  );
  if (res.matchedCount === 0) throw new Error(`no user with email ${email}`);
}

const USAGE = `usage:
  users.js create --email <email> --name <name> --role <${ROLES.join('|')}>
  users.js reset-password --email <email>
  users.js deactivate --email <email>`;

async function main(argv: string[]): Promise<number> {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      email: { type: 'string' },
      name: { type: 'string' },
      role: { type: 'string' },
    },
  });
  const command = positionals[0];
  const email = values.email;
  if (!command || !email) {
    console.error(USAGE);
    return 2;
  }
  const env = loadEnv();
  const logger = createLogger({ level: 'warn', isProduction: env.NODE_ENV === 'production' });
  await connectMongo(env.MONGO_URI, logger);
  try {
    if (command === 'create') {
      if (!values.name || !values.role) {
        console.error(USAGE);
        return 2;
      }
      const password = await createUser({ email, name: values.name, role: values.role });
      process.stdout.write(`created ${values.role} ${email}\npassword (shown once): ${password}\n`);
    } else if (command === 'reset-password') {
      const password = await resetPassword(email);
      process.stdout.write(`new password for ${email} (shown once): ${password}
`);
    } else if (command === 'deactivate') {
      await deactivateUser(email);
      process.stdout.write(`deactivated ${email}
`);
    } else {
      console.error(USAGE);
      return 2;
    }
    return 0;
  } catch (err) {
    console.error(`error: ${(err as Error).message}`);
    return 1;
  } finally {
    await disconnectMongo();
  }
}

// Run only as a script, not when imported by the tests.
if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/'))) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (err: unknown) => {
      console.error(err);
      process.exit(1);
    },
  );
}
