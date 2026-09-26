import type { Server as SocketServer } from 'socket.io';

import type { AlertDoc, PredictionDoc } from '../db/models.js';

/**
 * Realtime fan-out.
 *
 * Rooms mirror the eventual design: `stay:<id>` for everything about one
 * patient, `ward:<name>` for the dashboard. Events carry only what the UI
 * needs to update — never note text or demographics.
 */
export interface RealtimeBus {
  predictionStatus(prediction: PredictionDoc, status: string): void;
  alertCreated(alert: AlertDoc): void;
  alertUpdated(alert: AlertDoc): void;
  vitalsAppended(stayId: string, point: { ts: string; values: Record<string, unknown> }): void;
  simulationStatus(stayId: string, status: unknown): void;
}

export function stayRoom(stayId: string): string {
  return `stay:${stayId}`;
}

export function wardRoom(ward: string): string {
  return `ward:${ward}`;
}

export class SocketBus implements RealtimeBus {
  constructor(private readonly io: SocketServer) {}

  predictionStatus(prediction: PredictionDoc, status: string): void {
    const stayId = String(prediction.stayId);
    this.io.to(stayRoom(stayId)).emit(`prediction:${status}`, {
      predictionId: String(prediction._id),
      stayId,
      task: prediction.task,
      status,
      probability: prediction.result?.probability ?? null,
      cutoffTime: prediction.cutoffTime.toISOString(),
    });
  }

  alertCreated(alert: AlertDoc): void {
    this.io.emit('alert:new', this.serialiseAlert(alert));
  }

  alertUpdated(alert: AlertDoc): void {
    this.io.emit('alert:updated', this.serialiseAlert(alert));
  }

  vitalsAppended(stayId: string, point: { ts: string; values: Record<string, unknown> }): void {
    this.io.to(stayRoom(stayId)).emit('vitals:new', { stayId, ...point });
  }

  simulationStatus(stayId: string, status: unknown): void {
    this.io.to(stayRoom(stayId)).emit('simulation:status', status);
  }

  private serialiseAlert(alert: AlertDoc) {
    return {
      id: String(alert._id),
      stayId: String(alert.stayId),
      task: alert.task,
      severity: alert.severity,
      rule: alert.rule,
      value: alert.value,
      status: alert.status,
      createdAt: alert.get('createdAt')?.toISOString?.() ?? new Date().toISOString(),
    };
  }
}

/** No-op bus for tests and any context without a socket server. */
export class NullBus implements RealtimeBus {
  predictionStatus(): void {}
  alertCreated(): void {}
  alertUpdated(): void {}
  vitalsAppended(): void {}
  simulationStatus(): void {}
}
