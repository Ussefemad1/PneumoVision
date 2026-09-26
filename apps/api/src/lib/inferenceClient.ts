import { createHmac } from 'node:crypto';

import type { PredictionResult, Task } from '@pneumovision/shared';

import type { Env } from '../config/env.js';
import type { Logger } from './logger.js';

/**
 * HMAC-signed client for the inference service.
 *
 * Identical in the demo and the full stack: the signature covers
 * `timestamp + "." + body`, and the service rejects anything whose timestamp
 * is outside the allowed skew. The demo simply reaches a uvicorn process on
 * localhost instead of a container on the private network.
 */

export interface EhrPayload {
  variables: string[];
  /** 48 rows x 17 columns; null where nothing was charted. */
  values: (number | string | null)[][];
}

export interface NotePayload {
  id: string;
  text: string;
  type: 'radiology' | 'progress' | 'nursing' | 'discharge';
}

export interface PredictInput {
  task: Task;
  /** Seeds the deterministic mock so a stay keeps a coherent risk level. */
  seed: string;
  ehr: EhrPayload | null;
  cxr: { presignedUrl: string } | null;
  notes: NotePayload[] | null;
  theta: number;
  returnExplanations?: boolean;
}

export interface PredictOutput {
  result: PredictionResult;
  modelVersion: string;
  latencyMs: number;
  mock: boolean;
}

export class InferenceError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'InferenceError';
  }
}

export class InferenceClient {
  constructor(
    private readonly env: Env,
    private readonly logger: Logger,
  ) {}

  private sign(body: string): { timestamp: string; signature: string } {
    const timestamp = String(Math.floor(Date.now() / 1000));
    const signature = createHmac('sha256', this.env.INFERENCE_HMAC_SECRET)
      .update(`${timestamp}.${body}`)
      .digest('hex');
    return { timestamp, signature };
  }

  async predict(input: PredictInput, requestId: string): Promise<PredictOutput> {
    const body = JSON.stringify({
      task: input.task,
      seed: input.seed,
      ehr: input.ehr,
      cxr: input.cxr,
      notes: input.notes,
      theta: input.theta,
      returnExplanations: input.returnExplanations ?? true,
    });
    const { timestamp, signature } = this.sign(body);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.env.INFERENCE_TIMEOUT_MS);
    const startedAt = Date.now();

    try {
      const res = await fetch(`${this.env.INFERENCE_URL}/v1/predict`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-pv-timestamp': timestamp,
          'x-pv-signature': signature,
          'x-request-id': requestId,
        },
        body,
        signal: controller.signal,
      });

      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new InferenceError(
          `inference returned ${res.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`,
          res.status,
        );
      }

      const json = (await res.json()) as PredictOutput;
      // Content-free: task and availability only, never the inputs.
      this.logger.debug(
        {
          requestId,
          task: input.task,
          availability: {
            ehr: input.ehr !== null,
            cxr: input.cxr !== null,
            notes: (input.notes?.length ?? 0) > 0,
          },
          latencyMs: Date.now() - startedAt,
        },
        'inference ok',
      );
      return json;
    } catch (err) {
      if (err instanceof InferenceError) throw err;
      if (err instanceof Error && err.name === 'AbortError') {
        throw new InferenceError(`inference timed out after ${this.env.INFERENCE_TIMEOUT_MS}ms`);
      }
      throw new InferenceError(
        `inference unreachable at ${this.env.INFERENCE_URL}: ${(err as Error).message}`,
      );
    } finally {
      clearTimeout(timer);
    }
  }

  /** Readiness probe. Unsigned — the health route carries no patient data. */
  async healthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.env.INFERENCE_URL}/v1/health`, {
        signal: AbortSignal.timeout(3000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }
}
