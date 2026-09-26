import type { NextFunction, Request, Response } from 'express';
import { ZodError } from 'zod';

import type { Logger } from '../lib/logger.js';

/** Error carrying an HTTP status and a stable machine-readable code. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }

  static badRequest(code: string, message: string) {
    return new ApiError(400, code, message);
  }
  static unauthorized(message = 'Authentication required') {
    return new ApiError(401, 'UNAUTHENTICATED', message);
  }
  static forbidden(message = 'Insufficient permissions') {
    return new ApiError(403, 'FORBIDDEN', message);
  }
  static notFound(resource = 'Resource') {
    return new ApiError(404, 'NOT_FOUND', `${resource} not found`);
  }
  static conflict(code: string, message: string) {
    return new ApiError(409, code, message);
  }
}

export function notFoundHandler(req: Request, res: Response): void {
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: `No route for ${req.method} ${req.path}`,
      requestId: req.requestId,
    },
  });
}

/**
 * Terminal error handler. Responses always use the
 * `{ error: { code, message, requestId } }` envelope, and 5xx messages are
 * generic so internal detail never leaks to the client — it goes to the log.
 */
export function createErrorHandler(logger: Logger) {
  return function errorHandler(
    err: unknown,
    req: Request,
    res: Response,
    // Express identifies the error handler by arity; the parameter must stay.
    _next: NextFunction,
  ): void {
    if (err instanceof ZodError) {
      logger.warn({ requestId: req.requestId, issues: err.issues }, 'request validation failed');
      res.status(400).json({
        error: {
          code: 'VALIDATION_FAILED',
          message: 'Request validation failed',
          requestId: req.requestId,
          details: err.issues.map((i) => ({ path: i.path.join('.'), message: i.message })),
        },
      });
      return;
    }

    if (err instanceof ApiError) {
      logger.warn({ requestId: req.requestId, code: err.code }, err.message);
      res
        .status(err.status)
        .json({ error: { code: err.code, message: err.message, requestId: req.requestId } });
      return;
    }

    logger.error({ requestId: req.requestId, err }, 'unhandled error');
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: 'An unexpected error occurred',
        requestId: req.requestId,
      },
    });
  };
}
