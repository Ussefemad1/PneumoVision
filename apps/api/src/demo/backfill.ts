import type { Task } from '@pneumovision/shared';

import { PredictionModel, StayModel, VitalsModel } from '../db/models.js';
import type { Logger } from '../lib/logger.js';
import type { PredictionService } from '../services/predictionService.js';

/**
 * Scores every seeded stay at several points in its history.
 *
 * Without this the demo opens on an empty dashboard: no risk to sort by, no
 * trajectory to plot, and no alerts, because nothing has been scored yet.
 * Backfilling walks each stay's timeline and requests predictions at regular
 * cutoffs, exactly as a clinician clicking "Run prediction" would — same
 * service, same F8 filtering, same alert evaluation. The alerts that appear
 * on first load are therefore real outputs of the rules, not fixtures.
 */

/** How many mortality scorings to spread across each stay. */
const POINTS_PER_STAY = 8;

export async function backfillPredictions(
  service: PredictionService,
  logger: Logger,
): Promise<number> {
  const alreadyScored = await PredictionModel.estimatedDocumentCount();
  if (alreadyScored > 0) {
    logger.info({ predictions: alreadyScored }, 'predictions already present, skipping backfill');
    return 0;
  }

  const stays = await StayModel.find({ status: 'active' }).lean();
  let created = 0;

  for (const stay of stays) {
    const [first, last] = await Promise.all([
      VitalsModel.findOne({ 'meta.stayId': stay._id }).sort({ ts: 1 }).lean(),
      VitalsModel.findOne({ 'meta.stayId': stay._id }).sort({ ts: -1 }).lean(),
    ]);
    if (!first || !last) continue;

    const startMs = first.ts.getTime();
    const endMs = last.ts.getTime();
    // Begin a third of the way in, so the first scoring already has a
    // reasonable window of history behind it rather than two charted hours.
    const from = startMs + (endMs - startMs) / 3;
    const step = (endMs - from) / (POINTS_PER_STAY - 1);

    for (let i = 0; i < POINTS_PER_STAY; i++) {
      const cutoff = new Date(from + step * i + 1);
      await service.request({
        stayId: stay._id,
        task: 'mortality',
        cutoffTime: cutoff,
        requestedBy: null,
        requestId: `backfill-${String(stay._id)}-${i}`,
      });
      created++;
    }

    // One pneumonia scoring at the end of the window, so both gauges on the
    // stay overview have a value.
    await service.request({
      stayId: stay._id,
      task: 'pneumonia' satisfies Task,
      cutoffTime: new Date(endMs + 1),
      requestedBy: null,
      requestId: `backfill-${String(stay._id)}-pneumonia`,
    });
    created++;
  }

  logger.info({ predictions: created, stays: stays.length }, 'backfilled demo predictions');
  return created;
}
