import { z } from 'zod';

import { DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE } from '../constants.js';

/** Mongo ObjectId as it appears over the wire. */
export const objectId = z.string().regex(/^[0-9a-f]{24}$/i, 'must be a 24-character object id');

/** ISO-8601 timestamp. Everything crossing the API boundary is a string. */
export const isoDateTime = z.string().datetime({ offset: true });

export const paginationQuery = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(MAX_PAGE_SIZE).default(DEFAULT_PAGE_SIZE),
});
export type PaginationQuery = z.infer<typeof paginationQuery>;

export const paginationMeta = z.object({
  page: z.number().int(),
  limit: z.number().int(),
  total: z.number().int(),
  totalPages: z.number().int(),
});
export type PaginationMeta = z.infer<typeof paginationMeta>;

/** Success envelope: `{ data, meta? }`. */
export const envelope = <T extends z.ZodTypeAny>(data: T) =>
  z.object({ data, meta: paginationMeta.optional() });

/** Failure envelope: `{ error: { code, message, requestId } }`. */
export const errorEnvelope = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    requestId: z.string(),
    details: z.array(z.object({ path: z.string(), message: z.string() })).optional(),
  }),
});
export type ErrorEnvelope = z.infer<typeof errorEnvelope>;
