import mongoose from 'mongoose';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { AlertModel, PredictionModel } from '../src/db/models.js';
import { DEFAULT_ALERT_RULES, evaluatePrediction } from '../src/services/alertEngine.js';
import { clearCollections, startMemoryMongo, stopMemoryMongo } from './helpers/mongo.js';

const T0 = new Date('2026-03-01T00:00:00.000Z');
const at = (hours: number) => new Date(T0.getTime() + hours * 3_600_000);

let stayId: mongoose.Types.ObjectId;
const predictionId = () => new mongoose.Types.ObjectId();

beforeAll(async () => {
  await startMemoryMongo();
}, 180_000);

afterAll(async () => {
  await stopMemoryMongo();
});

beforeEach(async () => {
  await clearCollections();
  stayId = new mongoose.Types.ObjectId();
});

/** Records a completed prediction so the rise rule has history to compare. */
async function recordPrediction(probability: number, hour: number) {
  await PredictionModel.create({
    stayId,
    task: 'mortality',
    cutoffTime: at(hour),
    status: 'done',
    modelVersion: 'test',
    result: {
      probability,
      unimodal: { ehr: probability },
      joint: { high: probability, low: probability },
      missingness: {
        vector: { ehr: true, cxr: false, rr: false, dn: false },
        prediction: probability,
      },
      alphas: { high: 0.4, low: 0.3, miss: 0.3 },
      confidence: { theta: 0.75, fractionAbove: { ehr: 0.5 } },
    },
  });
}

describe('threshold rules', () => {
  it('raises nothing below the lowest threshold', async () => {
    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.3,
      cutoffTime: at(10),
    });
    expect(alert).toBeNull();
  });

  it('raises a warning at 50%', async () => {
    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.55,
      cutoffTime: at(10),
    });
    expect(alert?.severity).toBe('warning');
    expect(alert?.rule).toMatch(/50%/);
  });

  it('picks the most severe matching rule, not the first', async () => {
    // 0.8 crosses both the 50% and 70% bands.
    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.8,
      cutoffTime: at(10),
    });
    expect(alert?.severity).toBe('critical');
  });
});

describe('rise rule', () => {
  it('fires when risk climbs by Δ within the window, even below the threshold', async () => {
    await recordPrediction(0.2, 8);

    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.4, // +20 points, still under the 50% threshold
      cutoffTime: at(10),
    });

    expect(alert).not.toBeNull();
    expect(alert?.rule).toMatch(/rose/);
  });

  it('ignores a rise that happened outside the lookback window', async () => {
    // 20 hours earlier is well outside the 6-hour window.
    await recordPrediction(0.2, -20);

    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.4,
      cutoffTime: at(10),
    });
    expect(alert).toBeNull();
  });

  it('ignores a gradual rise below Δ', async () => {
    await recordPrediction(0.3, 8);
    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.38, // +8 points, under the 15-point Δ
      cutoffTime: at(10),
    });
    expect(alert).toBeNull();
  });
});

describe('deduplication', () => {
  it('does not raise a second alert of equal severity while one is open', async () => {
    const first = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.55,
      cutoffTime: at(10),
    });
    expect(first?.severity).toBe('warning');

    const second = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.58,
      cutoffTime: at(11),
    });
    expect(second).toBeNull();
    expect(await AlertModel.countDocuments({ stayId })).toBe(1);
  });

  it('does raise when the patient escalates to a higher severity', async () => {
    await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.55,
      cutoffTime: at(10),
    });

    const escalation = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.85,
      cutoffTime: at(11),
    });

    expect(escalation?.severity).toBe('critical');
    expect(await AlertModel.countDocuments({ stayId })).toBe(2);
  });

  it('keeps the two tasks independent', async () => {
    await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.55,
      cutoffTime: at(10),
    });
    const pneumonia = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'pneumonia',
      probability: 0.55,
      cutoffTime: at(10),
    });

    expect(pneumonia).not.toBeNull();
    expect(pneumonia?.task).toBe('pneumonia');
  });

  it('raises again once the earlier alert is resolved', async () => {
    const first = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.55,
      cutoffTime: at(10),
    });
    await AlertModel.findByIdAndUpdate(first!._id, { status: 'resolved', resolvedAt: new Date() });

    const second = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.56,
      cutoffTime: at(12),
    });
    expect(second).not.toBeNull();
  });
});

describe('rule configuration', () => {
  it('honours custom thresholds', async () => {
    const alert = await evaluatePrediction({
      stayId,
      predictionId: predictionId(),
      task: 'mortality',
      probability: 0.31,
      cutoffTime: at(10),
      rules: {
        thresholds: [{ severity: 'critical', at: 0.3 }],
        rise: DEFAULT_ALERT_RULES.rise,
      },
    });
    expect(alert?.severity).toBe('critical');
  });
});
