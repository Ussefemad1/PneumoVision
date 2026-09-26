import type { SimulationStatus } from '@pneumovision/shared';
import type { Types } from 'mongoose';

import { StayModel, VitalsModel } from '../../db/models.js';
import type { Logger } from '../../lib/logger.js';
import type { RealtimeBus } from '../../realtime/bus.js';
import type { PredictionService } from '../../services/predictionService.js';

/**
 * ICU Replay (P1).
 *
 * Walks a stay's existing hourly vitals forward in wall-clock time, emitting
 * each hour over the socket and re-scoring every few hours. This demonstrates
 * near-real-time bedside surveillance on retrospective data without inventing
 * any new measurements — it replays what the stay already contains.
 *
 * TODO(phase-8): move the ticking into a BullMQ repeatable job so replays
 * survive an API restart and can run on a worker. The demo keeps timers in
 * process memory, which is why a restart clears them.
 */

/** Re-score every N replayed hours. Scoring every hour is needless churn. */
const RESCORE_EVERY_HOURS = 4;

interface ActiveReplay {
  stayId: string;
  timer: NodeJS.Timeout;
  speed: number;
  cursorHour: number;
  totalHours: number;
  startHour: number;
}

export class SimulationService {
  private readonly active = new Map<string, ActiveReplay>();

  constructor(
    private readonly bus: RealtimeBus,
    private readonly predictions: PredictionService,
    private readonly logger: Logger,
  ) {}

  async start(stayId: Types.ObjectId, speed: number, requestId: string): Promise<SimulationStatus> {
    const key = String(stayId);
    this.stop(stayId);

    const stay = await StayModel.findById(stayId).lean();
    if (!stay) throw new Error('stay not found');

    const points = await VitalsModel.find({ 'meta.stayId': stayId }).sort({ ts: 1 }).lean();
    if (points.length === 0) throw new Error('stay has no vitals to replay');

    // Start partway in, so the replay reaches interesting territory quickly
    // rather than spending its first minute on a stable baseline.
    const startHour = Math.floor(points.length * 0.5);

    // speed is "simulated hours per real minute"; x60 means one hour a second.
    const intervalMs = Math.max(250, Math.round(60_000 / speed));

    const replay: ActiveReplay = {
      stayId: key,
      speed,
      cursorHour: startHour,
      totalHours: points.length,
      startHour,
      timer: setInterval(() => {
        void this.tick(stayId, points, requestId);
      }, intervalMs),
    };
    this.active.set(key, replay);

    const status = this.statusOf(key);
    this.bus.simulationStatus(key, status);
    this.logger.info({ stayId: key, speed, intervalMs }, 'replay started');
    return status;
  }

  private async tick(
    stayId: Types.ObjectId,
    points: { ts: Date; values: unknown }[],
    requestId: string,
  ): Promise<void> {
    const key = String(stayId);
    const replay = this.active.get(key);
    if (!replay) return;

    const point = points[replay.cursorHour];
    if (!point) {
      this.stop(stayId);
      return;
    }

    this.bus.vitalsAppended(key, {
      ts: point.ts.toISOString(),
      values: point.values as Record<string, unknown>,
    });

    const hoursElapsed = replay.cursorHour - replay.startHour;
    if (hoursElapsed > 0 && hoursElapsed % RESCORE_EVERY_HOURS === 0) {
      // Cut off at the replayed instant, so the prediction only ever sees
      // data the replay has reached — the same F8 rule as a live request.
      try {
        await this.predictions.request({
          stayId,
          task: 'mortality',
          cutoffTime: new Date(point.ts.getTime() + 1),
          requestedBy: null,
          requestId,
        });
      } catch (err) {
        this.logger.warn({ stayId: key, err: (err as Error).message }, 'replay scoring failed');
      }
    }

    replay.cursorHour += 1;
    this.bus.simulationStatus(key, this.statusOf(key));

    if (replay.cursorHour >= replay.totalHours) this.stop(stayId);
  }

  stop(stayId: Types.ObjectId | string): SimulationStatus {
    const key = String(stayId);
    const replay = this.active.get(key);
    if (replay) {
      clearInterval(replay.timer);
      this.active.delete(key);
      const status: SimulationStatus = {
        stayId: key,
        running: false,
        speed: replay.speed,
        cursorHour: replay.cursorHour,
        totalHours: replay.totalHours,
      };
      this.bus.simulationStatus(key, status);
      this.logger.info({ stayId: key }, 'replay stopped');
      return status;
    }
    return { stayId: key, running: false, speed: 60, cursorHour: 0, totalHours: 0 };
  }

  statusOf(stayId: string): SimulationStatus {
    const replay = this.active.get(stayId);
    if (!replay) {
      return { stayId, running: false, speed: 60, cursorHour: 0, totalHours: 0 };
    }
    return {
      stayId,
      running: true,
      speed: replay.speed,
      cursorHour: replay.cursorHour,
      totalHours: replay.totalHours,
    };
  }

  /** Clears every timer. Called on shutdown so the process can exit. */
  stopAll(): void {
    for (const key of [...this.active.keys()]) this.stop(key);
  }
}
