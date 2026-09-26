import { CLINICAL_ROLES, alertsQuery } from '@pneumovision/shared';
import { Router } from 'express';
import { Types } from 'mongoose';

import { AlertModel, PatientModel, StayModel } from '../../db/models.js';
import { asyncHandler, objectIdParam, ok } from '../../lib/http.js';
import { serializeAlert } from '../../lib/serialize.js';
import { requireRole } from '../../middleware/auth.js';
import { ApiError } from '../../middleware/errorHandler.js';
import type { RealtimeBus } from '../../realtime/bus.js';

/**
 * Alerts centre (P3).
 *
 * TODO(phase-3): every acknowledge/resolve must also append a hash-chained
 * audit entry. TODO(phase-8): `GET/PUT /alerts/rules` for admin-configurable
 * thresholds — the rules live in `services/alertEngine.ts` for now.
 */
export function alertRoutes(bus: RealtimeBus): Router {
  const router = Router();
  const clinical = requireRole(...CLINICAL_ROLES);

  router.get(
    '/alerts',
    clinical,
    asyncHandler(async (req, res) => {
      const filter = alertsQuery.parse(req.query);

      const alerts = await AlertModel.find({
        ...(filter.status ? { status: filter.status } : {}),
        ...(filter.severity ? { severity: filter.severity } : {}),
        ...(filter.stayId ? { stayId: new Types.ObjectId(filter.stayId) } : {}),
      })
        .sort({ createdAt: -1 })
        .limit(200)
        .lean();

      // Join the stay and patient so a row is actionable without a follow-up
      // request per alert.
      const stayIds = [...new Set(alerts.map((a) => String(a.stayId)))];
      const stays = await StayModel.find({ _id: { $in: stayIds } }).lean();
      const patients = await PatientModel.find({
        _id: { $in: stays.map((s) => s.patientId) },
      }).lean();

      const patientById = new Map(patients.map((p) => [String(p._id), p]));
      const stayById = new Map(
        stays.map((s) => [
          String(s._id),
          {
            id: String(s._id),
            ward: s.ward,
            bedLabel: s.bedLabel,
            pseudoId: patientById.get(String(s.patientId))?.pseudoId ?? 'unknown',
          },
        ]),
      );

      ok(
        res,
        alerts.map((a) => ({
          ...serializeAlert(a),
          stay: stayById.get(String(a.stayId)) ?? {
            id: String(a.stayId),
            ward: '—',
            bedLabel: '—',
            pseudoId: 'unknown',
          },
        })),
      );
    }),
  );

  router.patch(
    '/alerts/:alertId/acknowledge',
    clinical,
    asyncHandler(async (req, res) => {
      const alert = await AlertModel.findById(objectIdParam(req, 'alertId'));
      if (!alert) throw ApiError.notFound('Alert');

      if (alert.status === 'open') {
        alert.status = 'acknowledged';
        alert.acknowledgedBy = req.user ? new Types.ObjectId(req.user.id) : null;
        alert.acknowledgedAt = new Date();
        await alert.save();
        bus.alertUpdated(alert);
      }

      ok(res, serializeAlert(alert.toObject() as Record<string, unknown>));
    }),
  );

  router.patch(
    '/alerts/:alertId/resolve',
    clinical,
    asyncHandler(async (req, res) => {
      const alert = await AlertModel.findById(objectIdParam(req, 'alertId'));
      if (!alert) throw ApiError.notFound('Alert');

      if (alert.status !== 'resolved') {
        alert.status = 'resolved';
        alert.resolvedAt = new Date();
        // Resolving without acknowledging still records who acted.
        if (!alert.acknowledgedBy && req.user) {
          alert.acknowledgedBy = new Types.ObjectId(req.user.id);
          alert.acknowledgedAt = new Date();
        }
        await alert.save();
        bus.alertUpdated(alert);
      }

      ok(res, serializeAlert(alert.toObject() as Record<string, unknown>));
    }),
  );

  return router;
}
