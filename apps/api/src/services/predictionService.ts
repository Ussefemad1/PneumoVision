import { predictionResult, type ExcludedInput, type Task } from '@pneumovision/shared';
import type { Types } from 'mongoose';

import type { Env } from '../config/env.js';
import { PredictionModel, StayModel, type PredictionDoc } from '../db/models.js';
import { readStoredImage } from '../lib/imageStore.js';
import type { InferenceClient } from '../lib/inferenceClient.js';
import type { Logger } from '../lib/logger.js';
import type { RealtimeBus } from '../realtime/bus.js';
import { evaluatePrediction } from './alertEngine.js';
import { gatherInputs } from './predictionInputs.js';

/**
 * Orchestrates one prediction: gather inputs (F8-filtered), call the inference
 * service, persist the full F1–F6 result, evaluate alert rules, and broadcast
 * status over the realtime bus.
 *
 * `PredictionService` is the seam BullMQ drops into. The demo's in-process
 * implementation awaits the work and returns the finished document; the queued
 * implementation will return as soon as the job is enqueued and drive the same
 * `runPrediction` body inside a worker. Callers depend on the interface, so
 * swapping them is a wiring change in `createServer`, not a route rewrite.
 */
export interface PredictionService {
  /**
   * Requests a prediction. Resolves once the prediction *document* exists —
   * which in the demo means it has already finished, and under BullMQ will
   * mean it is queued. Check `status` rather than assuming.
   */
  request(args: RequestArgs): Promise<PredictionDoc>;
}

export interface RequestArgs {
  stayId: Types.ObjectId;
  task: Task;
  cutoffTime: Date;
  requestedBy: Types.ObjectId | null;
  requestId: string;
  /** Inputs the caller withheld before storing anything (ad-hoc uploads). */
  excluded?: ExcludedInput[];
}

export interface PredictionDeps {
  env: Env;
  logger: Logger;
  inference: InferenceClient;
  bus: RealtimeBus;
}

/**
 * Runs the whole pipeline to completion. Shared by both implementations — the
 * BullMQ worker will call exactly this.
 */
export async function runPrediction(
  prediction: PredictionDoc,
  deps: PredictionDeps,
  requestId: string,
): Promise<PredictionDoc> {
  const { logger, bus, inference, env } = deps;
  const stayId = prediction.stayId;

  prediction.status = 'running';
  await prediction.save();
  bus.predictionStatus(prediction, 'running');

  try {
    const inputs = await gatherInputs(stayId, prediction.task, prediction.cutoffTime);
    prediction.inputHash = inputs.inputHash;
    prediction.inputsUsed = inputs.used;
    prediction.excludedInputs.push(...inputs.excluded);

    // The bytes travel inside the signed body. A URL would need a clinician
    // cookie the service does not have, and on a single-container deploy the
    // file only exists on this container's disk.
    const image = inputs.cxr ? await readStoredImage(env, inputs.cxr) : null;

    const output = await inference.predict(
      {
        task: prediction.task,
        // Seeding by stay keeps a patient's risk coherent across re-scorings
        // while the service is in mock mode.
        seed: String(stayId),
        ehr: inputs.ehr,
        cxr:
          inputs.cxr && image
            ? {
                imageId: inputs.cxr.imageId,
                contentType: image.contentType,
                sha256: image.sha256,
                dataB64: image.bytes.toString('base64'),
              }
            : null,
        notes: inputs.notes.length > 0 ? inputs.notes : null,
        theta: env.DEFAULT_THETA,
        returnExplanations: true,
      },
      requestId,
    );

    // Validate at the boundary rather than trusting the service: a shape
    // change upstream should fail loudly here, not corrupt a stored report.
    const result = predictionResult.parse(output.result);

    prediction.status = 'done';
    prediction.modelVersion = output.modelVersion;
    prediction.latencyMs = output.latencyMs;
    // `.set()` rather than assignment: Mongoose infers the nested
    // `cxrPatchGrid` path as a DocumentArray, which a plain number[][] is not
    // assignable to. Validation still runs on save.
    prediction.set('result', result);
    prediction.error = null;
    await prediction.save();

    bus.predictionStatus(prediction, 'done');

    // Ad-hoc analyses belong to no admitted patient: nobody is on the ward to
    // be alerted about.
    const stay = await StayModel.findById(stayId).select({ status: 1 }).lean();
    if (stay?.status === 'adhoc') return prediction;

    const alert = await evaluatePrediction({
      stayId,
      predictionId: prediction._id,
      task: prediction.task,
      probability: result.probability,
      cutoffTime: prediction.cutoffTime,
    });
    if (alert) bus.alertCreated(alert);

    return prediction;
  } catch (err) {
    const message = err instanceof Error ? err.message : 'unknown inference failure';
    prediction.status = 'failed';
    prediction.error = message;
    await prediction.save();
    bus.predictionStatus(prediction, 'failed');
    logger.error(
      { requestId, predictionId: String(prediction._id), err: message },
      'prediction failed',
    );
    return prediction;
  }
}

/**
 * DEMO_MODE implementation: runs inline, no Redis, no worker process.
 *
 * TODO(phase-6): `QueuedPredictionService` enqueues a BullMQ job and returns
 * the queued document immediately; a worker calls `runPrediction`. Nothing
 * outside this file changes.
 */
export class InProcessPredictionService implements PredictionService {
  constructor(private readonly deps: PredictionDeps) {}

  async request(args: RequestArgs): Promise<PredictionDoc> {
    const stay = await StayModel.findById(args.stayId).lean();
    if (!stay) throw new Error(`stay ${String(args.stayId)} not found`);

    const prediction = await PredictionModel.create({
      stayId: args.stayId,
      task: args.task,
      cutoffTime: args.cutoffTime,
      status: 'queued',
      requestedBy: args.requestedBy,
      modelVersion: 'pending',
      excludedInputs: args.excluded ?? [],
    });
    this.deps.bus.predictionStatus(prediction, 'queued');

    return runPrediction(prediction, this.deps, args.requestId);
  }
}
