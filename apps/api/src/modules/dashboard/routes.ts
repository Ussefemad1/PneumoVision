import { CLINICAL_ROLES } from '@pneumovision/shared';
import { Router } from 'express';

import { AlertModel, PatientModel, PredictionModel, StayModel } from '../../db/models.js';
import { asyncHandler, ok } from '../../lib/http.js';
import { requireRole } from '../../middleware/auth.js';

/**
 * Ward dashboard (P4): every active stay, sorted by risk, with a sparkline,
 * modality badges and an unacknowledged-alert count.
 */
export function dashboardRoutes(): Router {
  const router = Router();

  router.get(
    '/dashboard/ward',
    requireRole(...CLINICAL_ROLES),
    asyncHandler(async (req, res) => {
      const ward = typeof req.query.ward === 'string' && req.query.ward ? req.query.ward : null;

      const stays = await StayModel.find({ status: 'active', ...(ward ? { ward } : {}) }).lean();
      const stayIds = stays.map((s) => s._id);

      const [patients, predictions, openAlerts, allWards] = await Promise.all([
        PatientModel.find({ _id: { $in: stays.map((s) => s.patientId) } }).lean(),
        // Every completed prediction for these stays, oldest first, so the
        // sparkline and the latest value both come from one pass.
        PredictionModel.find({ stayId: { $in: stayIds }, status: 'done' })
          .sort({ cutoffTime: 1 })
          .select({ stayId: 1, task: 1, cutoffTime: 1, 'result.probability': 1 })
          .lean(),
        AlertModel.aggregate<{ _id: unknown; count: number }>([
          { $match: { stayId: { $in: stayIds }, status: 'open' } },
          { $group: { _id: '$stayId', count: { $sum: 1 } } },
        ]),
        StayModel.distinct('ward', { status: 'active' }),
      ]);

      const patientById = new Map(patients.map((p) => [String(p._id), p]));
      const alertCountByStay = new Map(openAlerts.map((a) => [String(a._id), a.count]));

      const byStay = new Map<
        string,
        { mortality: number[]; pneumonia: number[]; last: Date | null }
      >();
      for (const p of predictions) {
        const key = String(p.stayId);
        const entry = byStay.get(key) ?? { mortality: [], pneumonia: [], last: null };
        const value = p.result?.probability;
        if (typeof value === 'number') {
          if (p.task === 'mortality') entry.mortality.push(value);
          else entry.pneumonia.push(value);
          entry.last = p.cutoffTime;
        }
        byStay.set(key, entry);
      }

      const rows = stays.map((stay) => {
        const key = String(stay._id);
        const series = byStay.get(key);
        const patient = patientById.get(String(stay.patientId));
        const mortalitySeries = series?.mortality ?? [];

        return {
          stayId: key,
          pseudoId: patient?.pseudoId ?? 'unknown',
          ward: stay.ward,
          bedLabel: stay.bedLabel,
          age: patient?.demographics?.age ?? 0,
          sex: patient?.demographics?.sex ?? 'M',
          availability: stay.availability,
          mortality: mortalitySeries.at(-1) ?? null,
          pneumonia: series?.pneumonia.at(-1) ?? null,
          sparkline: mortalitySeries.slice(-12),
          openAlerts: alertCountByStay.get(key) ?? 0,
          lastScoredAt: series?.last?.toISOString() ?? null,
        };
      });

      // Highest risk first; never-scored stays sink to the bottom rather than
      // being treated as zero risk.
      rows.sort((a, b) => (b.mortality ?? -1) - (a.mortality ?? -1));

      ok(res, { wards: allWards.sort(), rows });
    }),
  );

  return router;
}
