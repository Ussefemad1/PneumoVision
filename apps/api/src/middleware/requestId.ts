import { randomUUID } from 'node:crypto';
import type { NextFunction, Request, Response } from 'express';

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      /** Correlation id, echoed to the client and propagated to inference. */
      requestId: string;
    }
  }
}

/**
 * Assigns a request id used in logs, error envelopes and audit entries, and
 * forwarded to the inference service so a prediction can be traced end to end.
 * An inbound X-Request-Id is accepted only if it looks like a UUID, so a client
 * cannot inject arbitrary text into the logs.
 */
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function requestId(req: Request, res: Response, next: NextFunction): void {
  const inbound = req.header('x-request-id');
  req.requestId = inbound && UUID_RE.test(inbound) ? inbound : randomUUID();
  res.setHeader('x-request-id', req.requestId);
  next();
}
