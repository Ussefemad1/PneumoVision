import type { AlertSeverity, Task } from '@pneumovision/shared';
import type { Types } from 'mongoose';

import { AlertModel, PredictionModel, type AlertDoc } from '../db/models.js';

/**
 * Alerting rules (P3).
 *
 * Two kinds fire:
 *  - **threshold** — risk at or above a severity band;
 *  - **rise** — risk climbed by at least Δ within the last N hours, which
 *    catches a patient deteriorating from a low but rising baseline before
 *    they ever cross an absolute threshold.
 *
 * TODO(phase-8): make these admin-configurable through the `settings`
 * collection and `PUT /alerts/rules`. The demo uses the constants below.
 */

export interface AlertRules {
  thresholds: { severity: AlertSeverity; at: number }[];
  rise: { delta: number; withinHours: number; severity: AlertSeverity };
}

export const DEFAULT_ALERT_RULES: AlertRules = {
  thresholds: [
    { severity: 'critical', at: 0.7 },
    { severity: 'warning', at: 0.5 },
  ],
  rise: { delta: 0.15, withinHours: 6, severity: 'warning' },
};

/** The threshold drawn on the risk-trajectory chart. */
export const ALERT_DISPLAY_THRESHOLD = 0.5;

const SEVERITY_ORDER: Record<AlertSeverity, number> = { info: 0, warning: 1, critical: 2 };

export interface EvaluateArgs {
  stayId: Types.ObjectId;
  predictionId: Types.ObjectId;
  task: Task;
  probability: number;
  cutoffTime: Date;
  rules?: AlertRules;
}

/**
 * Evaluates the rules for one finished prediction and creates at most one
 * alert — the most severe rule that matched.
 *
 * Deliberately deduplicated: while an alert for this stay and task is still
 * open, another is not raised, so a deteriorating patient produces one
 * actionable item rather than a stream of them. A *more severe* rule does
 * raise a new alert, because that is new information.
 */
export async function evaluatePrediction(args: EvaluateArgs): Promise<AlertDoc | null> {
  const rules = args.rules ?? DEFAULT_ALERT_RULES;
  const matches: { severity: AlertSeverity; rule: string }[] = [];

  for (const band of rules.thresholds) {
    if (args.probability >= band.at) {
      matches.push({
        severity: band.severity,
        rule: `${taskLabel(args.task)} risk at or above ${(band.at * 100).toFixed(0)}%`,
      });
    }
  }

  const earlier = await PredictionModel.findOne({
    stayId: args.stayId,
    task: args.task,
    status: 'done',
    cutoffTime: {
      $lt: args.cutoffTime,
      $gte: new Date(args.cutoffTime.getTime() - rules.rise.withinHours * 3_600_000),
    },
  })
    .sort({ cutoffTime: 1 })
    .lean();

  const previous = earlier?.result?.probability;
  if (typeof previous === 'number' && args.probability - previous >= rules.rise.delta) {
    matches.push({
      severity: rules.rise.severity,
      rule:
        `${taskLabel(args.task)} risk rose ${((args.probability - previous) * 100).toFixed(1)} ` +
        `points in ${rules.rise.withinHours}h`,
    });
  }

  if (matches.length === 0) return null;

  const best = matches.reduce((a, b) =>
    SEVERITY_ORDER[b.severity] > SEVERITY_ORDER[a.severity] ? b : a,
  );

  const existing = await AlertModel.findOne({
    stayId: args.stayId,
    task: args.task,
    status: 'open',
  }).lean();

  if (existing && SEVERITY_ORDER[existing.severity] >= SEVERITY_ORDER[best.severity]) {
    return null;
  }

  return AlertModel.create({
    stayId: args.stayId,
    predictionId: args.predictionId,
    task: args.task,
    severity: best.severity,
    rule: best.rule,
    value: args.probability,
  });
}

function taskLabel(task: Task): string {
  return task === 'mortality' ? 'Mortality' : 'Pneumonia';
}
