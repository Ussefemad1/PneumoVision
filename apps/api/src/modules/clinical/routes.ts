import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { isAbsolute, join, normalize } from 'node:path';

import { CLINICAL_ROLES, paginationQuery, vitalsQuery } from '@pneumovision/shared';
import { Router } from 'express';

import type { Env } from '../../config/env.js';
import {
  AlertModel,
  ImageModel,
  NoteModel,
  PatientModel,
  PredictionModel,
  StayModel,
  VitalsModel,
} from '../../db/models.js';
import { asyncHandler, objectIdParam, ok } from '../../lib/http.js';
import {
  serializeImage,
  serializeNote,
  serializePatient,
  serializeStay,
  serializeVitals,
} from '../../lib/serialize.js';
import { requireRole } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';

/**
 * Patients, stays, and the three modality collections.
 *
 * TODO(phase-4): POST/PATCH/DELETE for each of these, CSV vitals import,
 * multipart image upload with magic-byte validation, and MinIO-backed
 * storage. The demo is read-only over seeded data.
 */
export function clinicalRoutes(env: Env, imageUrlFor: (id: string) => string): Router {
  const router = Router();
  const clinical = requireRole(...CLINICAL_ROLES);

  // ── Patients ───────────────────────────────────────────────────────────
  router.get(
    '/patients',
    clinical,
    asyncHandler(async (req, res) => {
      const { page, limit } = paginationQuery.parse(req.query);
      const search = typeof req.query.search === 'string' ? req.query.search.trim() : '';

      // Anchored, escaped prefix match — never a user-built regex.
      const filter = search
        ? { pseudoId: new RegExp(`^${search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'i') }
        : {};

      const [docs, total] = await Promise.all([
        PatientModel.find(filter)
          .sort({ pseudoId: 1 })
          .skip((page - 1) * limit)
          .limit(limit)
          .lean(),
        PatientModel.countDocuments(filter),
      ]);

      ok(res, docs.map(serializePatient), {
        page,
        limit,
        total,
        totalPages: Math.max(1, Math.ceil(total / limit)),
      });
    }),
  );

  // ── Stays ──────────────────────────────────────────────────────────────
  router.get(
    '/stays/:stayId',
    clinical,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const stay = await StayModel.findById(stayId).lean();
      if (!stay) throw ApiError.notFound('Stay');

      const patient = await PatientModel.findById(stay.patientId).lean();
      if (!patient) throw ApiError.notFound('Patient');

      const [latestMortality, latestPneumonia, openAlertCount] = await Promise.all([
        PredictionModel.findOne({ stayId, task: 'mortality', status: 'done' })
          .sort({ cutoffTime: -1 })
          .lean(),
        PredictionModel.findOne({ stayId, task: 'pneumonia', status: 'done' })
          .sort({ cutoffTime: -1 })
          .lean(),
        AlertModel.countDocuments({ stayId, status: 'open' }),
      ]);

      ok(res, {
        ...serializeStay(stay),
        patient: serializePatient(patient),
        latestPredictions: {
          mortality: latestMortality?.result?.probability ?? null,
          pneumonia: latestPneumonia?.result?.probability ?? null,
        },
        openAlertCount,
      });
    }),
  );

  // ── Vitals ─────────────────────────────────────────────────────────────
  router.get(
    '/stays/:stayId/vitals',
    clinical,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const { from, to } = vitalsQuery.parse(req.query);

      const range: Record<string, Date> = {};
      if (from) range.$gte = new Date(from);
      if (to) range.$lte = new Date(to);

      const docs = await VitalsModel.find({
        'meta.stayId': stayId,
        ...(Object.keys(range).length > 0 ? { ts: range } : {}),
      })
        .sort({ ts: 1 })
        .lean();

      ok(res, docs.map(serializeVitals));
    }),
  );

  // ── Images ─────────────────────────────────────────────────────────────
  router.get(
    '/stays/:stayId/images',
    clinical,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const docs = await ImageModel.find({ stayId }).sort({ takenAt: -1 }).lean();
      ok(
        res,
        docs.map((d) => serializeImage(d, imageUrlFor)),
      );
    }),
  );

  /**
   * Streams the image bytes.
   *
   * TODO(phase-4): replaced by `GET /images/:id/url` returning a 5-minute
   * presigned MinIO URL. Until then the API serves the file itself, and the
   * path is resolved from the database record only — never from user input —
   * then re-checked to be inside the demo directory.
   */
  router.get(
    '/images/:imageId/file',
    clinical,
    asyncHandler(async (req, res) => {
      const imageId = objectIdParam(req, 'imageId');
      const image = await ImageModel.findById(imageId).lean();
      if (!image) throw ApiError.notFound('Image');

      const root = join(process.cwd(), env.DEMO_DATA_DIR, 'images');
      const resolved = normalize(join(root, image.filePath));
      if (!resolved.startsWith(root) || isAbsolute(image.filePath)) {
        throw ApiError.badRequest('INVALID_PATH', 'Image path escapes the demo directory');
      }

      try {
        await stat(resolved);
      } catch {
        throw ApiError.notFound('Image file');
      }

      res.setHeader('Content-Type', 'image/png');
      res.setHeader('Cache-Control', 'private, max-age=300');
      createReadStream(resolved).pipe(res);
    }),
  );

  // ── Notes ──────────────────────────────────────────────────────────────
  router.get(
    '/stays/:stayId/notes',
    clinical,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const docs = await NoteModel.find({ stayId }).sort({ authoredAt: -1 }).lean();
      ok(res, docs.map(serializeNote));
    }),
  );

  return router;
}
