import { CLINICAL_ROLES, createPredictionRequest, predictionsQuery } from '@pneumovision/shared';
import { Router } from 'express';
import { Types } from 'mongoose';

import { ImageModel, NoteModel, PredictionModel, StayModel } from '../../db/models.js';
import { asyncHandler, objectIdParam, ok } from '../../lib/http.js';
import { serializePrediction } from '../../lib/serialize.js';
import { requireRole } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';
import { ALERT_DISPLAY_THRESHOLD } from '../../services/alertEngine.js';
import type { PredictionService } from '../../services/predictionService.js';

export function predictionRoutes(service: PredictionService): Router {
  const router = Router();
  const clinical = requireRole(...CLINICAL_ROLES);

  /**
   * Requests a prediction.
   *
   * Returns 201 with the finished prediction in the demo, where scoring runs
   * in-process. TODO(phase-6): under BullMQ this becomes 202 with a queued
   * document, and the client waits for the `prediction:done` socket event —
   * which it already listens for, so the UI needs no change.
   */
  router.post(
    '/stays/:stayId/predictions',
    requireRole('clinician', 'radiologist', 'admin'),
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const body = createPredictionRequest.parse(req.body ?? {});

      const stay = await StayModel.findById(stayId).lean();
      if (!stay) throw ApiError.notFound('Stay');

      const cutoffTime = body.cutoffTime ? new Date(body.cutoffTime) : new Date();
      if (Number.isNaN(cutoffTime.getTime())) {
        throw ApiError.badRequest('INVALID_CUTOFF', 'cutoffTime is not a valid date');
      }

      const prediction = await service.request({
        stayId,
        task: body.task,
        cutoffTime,
        requestedBy: req.user ? new Types.ObjectId(req.user.id) : null,
        requestId: req.requestId,
      });

      res.status(prediction.status === 'done' ? 201 : 202);
      ok(res, serializePrediction(prediction.toObject() as Record<string, unknown>));
    }),
  );

  router.get(
    '/stays/:stayId/predictions',
    clinical,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const { task } = predictionsQuery.parse(req.query);

      const docs = await PredictionModel.find({ stayId, ...(task ? { task } : {}) })
        .sort({ cutoffTime: -1 })
        .limit(100)
        .lean();

      ok(res, docs.map(serializePrediction));
    }),
  );

  router.get(
    '/predictions/:predictionId',
    clinical,
    asyncHandler(async (req, res) => {
      const predictionId = objectIdParam(req, 'predictionId');
      const doc = await PredictionModel.findById(predictionId).lean();
      if (!doc) throw ApiError.notFound('Prediction');
      ok(res, serializePrediction(doc));
    }),
  );

  /**
   * P2: the risk trajectory, plus markers for when a radiograph or note
   * arrived so a clinician can see what new evidence moved the curve.
   */
  router.get(
    '/stays/:stayId/risk-trajectory',
    clinical,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const task = predictionsQuery.parse(req.query).task ?? 'mortality';

      const [predictions, images, notes] = await Promise.all([
        PredictionModel.find({ stayId, task, status: 'done' }).sort({ cutoffTime: 1 }).lean(),
        ImageModel.find({ stayId }).sort({ takenAt: 1 }).lean(),
        NoteModel.find({ stayId }).sort({ authoredAt: 1 }).lean(),
      ]);

      ok(res, {
        task,
        points: predictions
          .filter((p) => p.result)
          .map((p) => ({
            predictionId: String(p._id),
            cutoffTime: p.cutoffTime.toISOString(),
            probability: p.result!.probability,
          })),
        events: [
          ...images.map((i) => ({
            at: i.takenAt.toISOString(),
            kind: 'cxr' as const,
            label: `${i.view} radiograph`,
          })),
          ...notes.map((n) => ({
            at: n.authoredAt.toISOString(),
            kind: 'note' as const,
            label: `${n.type} note`,
          })),
        ].sort((a, b) => a.at.localeCompare(b.at)),
        threshold: ALERT_DISPLAY_THRESHOLD,
      });
    }),
  );

  return router;
}
