import { startSimulationRequest } from '@pneumovision/shared';
import { Router } from 'express';

import { asyncHandler, objectIdParam, ok } from '../../lib/http.js';
import { requireRole } from '../../middleware/auth.js';
import type { SimulationService } from './service.js';

/** ICU Replay control (P1). */
export function simulationRoutes(simulation: SimulationService): Router {
  const router = Router();
  const allowed = requireRole('clinician', 'researcher', 'admin');

  router.post(
    '/simulation/stays/:stayId/start',
    allowed,
    asyncHandler(async (req, res) => {
      const stayId = objectIdParam(req, 'stayId');
      const { speed } = startSimulationRequest.parse(req.body ?? {});
      ok(res, await simulation.start(stayId, speed, req.requestId));
    }),
  );

  router.post('/simulation/stays/:stayId/stop', allowed, (req, res) => {
    ok(res, simulation.stop(objectIdParam(req, 'stayId')));
  });

  router.get('/simulation/stays/:stayId/status', allowed, (req, res) => {
    ok(res, simulation.statusOf(String(objectIdParam(req, 'stayId'))));
  });

  return router;
}
