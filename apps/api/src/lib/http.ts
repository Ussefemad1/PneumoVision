import type { NextFunction, Request, RequestHandler, Response } from 'express';
import { Types } from 'mongoose';

import { ApiError } from '../middleware/errorHandler.js';

/** Wraps an async handler so a rejected promise reaches the error middleware. */
export function asyncHandler(
  fn: (req: Request, res: Response, next: NextFunction) => Promise<unknown>,
): RequestHandler {
  return (req, res, next) => {
    void fn(req, res, next).catch(next);
  };
}

/** Parses a route parameter as an ObjectId, 404-ing rather than 500-ing. */
export function objectIdParam(req: Request, name: string): Types.ObjectId {
  const raw = req.params[name];
  if (!raw || !Types.ObjectId.isValid(raw)) {
    throw ApiError.badRequest('INVALID_ID', `${name} is not a valid id`);
  }
  return new Types.ObjectId(raw);
}

/** Success envelope. */
export function ok<T>(res: Response, data: T, meta?: unknown): void {
  res.json(meta === undefined ? { data } : { data, meta });
}

/**
 * Rejects any object containing Mongo operator keys. Route inputs are already
 * zod-validated, but this is a cheap second line against operator injection
 * reaching a query.
 */
export function assertNoOperators(value: unknown, path = 'body'): void {
  if (value === null || typeof value !== 'object') return;
  for (const [key, child] of Object.entries(value)) {
    if (key.startsWith('$') || key.includes('.')) {
      throw ApiError.badRequest('INVALID_INPUT', `Illegal key "${key}" in ${path}`);
    }
    assertNoOperators(child, `${path}.${key}`);
  }
}
