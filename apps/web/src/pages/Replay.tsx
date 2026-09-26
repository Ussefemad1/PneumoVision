import type { RiskTrajectory, SimulationStatus, WardDashboard } from '@pneumovision/shared';
import { REPLAY_SPEEDS } from '@pneumovision/shared';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useState } from 'react';

import { apiErrorMessage, getData, postData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent, timeOnly } from '../lib/format.js';
import { SEVERITY_INFO } from '../lib/risk.js';
import {
  getSocket,
  type AlertEvent,
  type PredictionEvent,
  type SimulationEvent,
  type VitalsEvent,
} from '../lib/socket.js';
import { RiskTrajectoryChart } from '../components/charts.jsx';
import { RiskBadge } from '../components/risk.jsx';
import { Button, Card, EmptyState, Field } from '../components/ui.jsx';

/** Vitals worth watching tick past during a replay. */
const WATCHED = [
  'Heart Rate',
  'Respiratory rate',
  'Oxygen saturation',
  'Systolic blood pressure',
  'Temperature',
] as const;

interface Toast {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  rule: string;
  value: number;
}

/**
 * P1: ICU Replay.
 *
 * Walks a stay's recorded hours forward in wall-clock time, re-scoring
 * periodically, so the surveillance behaviour is visible on retrospective
 * data. Everything here is driven by socket events from the API.
 */
export function ReplayPage() {
  const queryClient = useQueryClient();
  const [stayId, setStayId] = useState('');
  const [speed, setSpeed] = useState<number>(60);
  const [status, setStatus] = useState<SimulationStatus | null>(null);
  const [ticks, setTicks] = useState<VitalsEvent[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [error, setError] = useState<string | null>(null);
  const toastTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const { data: dashboard } = useQuery({
    queryKey: ['dashboard', { ward: '' }],
    queryFn: () => getData<WardDashboard>('/dashboard/ward'),
  });

  const trajectory = useQuery({
    queryKey: ['trajectory', stayId, 'mortality'],
    queryFn: () =>
      getData<RiskTrajectory>(`/stays/${stayId}/risk-trajectory`, { task: 'mortality' }),
    enabled: Boolean(stayId),
  });

  const start = useMutation({
    mutationFn: () => postData<SimulationStatus>(`/simulation/stays/${stayId}/start`, { speed }),
    onSuccess: (s) => {
      setStatus(s);
      setError(null);
      setTicks([]);
    },
    onError: (err) => setError(apiErrorMessage(err, 'Could not start the replay')),
  });

  const stop = useMutation({
    mutationFn: () => postData<SimulationStatus>(`/simulation/stays/${stayId}/stop`),
    onSuccess: (s) => setStatus(s),
  });

  const pushToast = useCallback((toast: Toast) => {
    setToasts((prev) => [...prev.slice(-2), toast]);
    const timer = setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== toast.id));
    }, 8000);
    toastTimers.current.push(timer);
  }, []);

  useEffect(() => () => toastTimers.current.forEach(clearTimeout), []);

  // Subscribe to the selected stay and wire the four event kinds.
  useEffect(() => {
    if (!stayId) return;
    const socket = getSocket();
    socket.emit('subscribe:stay', stayId);

    const onVitals = (event: VitalsEvent) => {
      if (event.stayId !== stayId) return;
      setTicks((prev) => [...prev.slice(-11), event]);
    };
    const onSimulation = (event: SimulationEvent) => {
      if (event.stayId === stayId) setStatus(event);
    };
    const onPrediction = (event: PredictionEvent) => {
      if (event.stayId !== stayId) return;
      void queryClient.invalidateQueries({ queryKey: ['trajectory', stayId] });
    };
    const onAlert = (event: AlertEvent) => {
      if (event.stayId !== stayId) return;
      pushToast({ id: event.id, severity: event.severity, rule: event.rule, value: event.value });
    };

    socket.on('vitals:new', onVitals);
    socket.on('simulation:status', onSimulation);
    socket.on('prediction:done', onPrediction);
    socket.on('alert:new', onAlert);

    return () => {
      socket.emit('unsubscribe:stay', stayId);
      socket.off('vitals:new', onVitals);
      socket.off('simulation:status', onSimulation);
      socket.off('prediction:done', onPrediction);
      socket.off('alert:new', onAlert);
    };
  }, [stayId, queryClient, pushToast]);

  // Stop the replay when leaving the page, so a timer is not left running.
  useEffect(() => {
    const current = stayId;
    return () => {
      if (current) void postData(`/simulation/stays/${current}/stop`).catch(() => undefined);
    };
  }, [stayId]);

  const running = status?.running ?? false;
  const progress = status && status.totalHours > 0 ? status.cursorHour / status.totalHours : 0;
  const selected = dashboard?.rows.find((r) => r.stayId === stayId);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-semibold text-ink">ICU Replay</h1>
        <p className="text-sm text-ink-muted">
          Replays a stay hour by hour and re-scores as it goes — near-real-time surveillance
          demonstrated on recorded data.
        </p>
      </header>

      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[220px] flex-1">
            <label htmlFor="stay" className="mb-1 block text-xs font-medium text-ink-secondary">
              Patient
            </label>
            <select
              id="stay"
              value={stayId}
              onChange={(e) => {
                setStayId(e.target.value);
                setStatus(null);
                setTicks([]);
              }}
              className="w-full rounded-md border border-hairline bg-surface-2 px-2.5 py-1.5 text-sm text-ink"
            >
              <option value="">Select a stay…</option>
              {dashboard?.rows.map((row) => (
                <option key={row.stayId} value={row.stayId}>
                  {row.pseudoId} — {row.ward} {row.bedLabel} ({percent(row.mortality)})
                </option>
              ))}
            </select>
          </div>

          <div>
            <span className="mb-1 block text-xs font-medium text-ink-secondary">Speed</span>
            <div className="flex rounded-md border border-hairline p-0.5">
              {REPLAY_SPEEDS.map((option) => (
                <button
                  key={option}
                  onClick={() => setSpeed(option)}
                  disabled={running}
                  className={cn(
                    'rounded px-2.5 py-1 text-xs transition disabled:opacity-50',
                    speed === option
                      ? 'bg-surface-3 font-medium text-ink'
                      : 'text-ink-muted hover:text-ink',
                  )}
                >
                  ×{option}
                </button>
              ))}
            </div>
          </div>

          <Button
            variant={running ? 'danger' : 'primary'}
            disabled={!stayId || start.isPending}
            onClick={() => (running ? stop.mutate() : start.mutate())}
          >
            {running ? 'Stop replay' : start.isPending ? 'Starting…' : 'Start replay'}
          </Button>
        </div>

        {error && (
          <p role="alert" className="mt-2 text-xs text-status-critical">
            {error}
          </p>
        )}

        {status && (
          <div className="mt-4">
            <div className="mb-1 flex items-center justify-between text-[11px] text-ink-muted">
              <span>
                Hour {status.cursorHour} of {status.totalHours}
              </span>
              <span>{running ? `running at ×${status.speed}` : 'stopped'}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
              <div
                className="h-full rounded-full bg-[var(--series-mortality)] transition-all"
                style={{ width: `${Math.min(100, progress * 100)}%` }}
              />
            </div>
          </div>
        )}
      </Card>

      {!stayId ? (
        <Card>
          <EmptyState
            icon="▷"
            title="Pick a patient to replay"
            description="Choose a deteriorating stay to watch the risk climb and an alert fire."
          />
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <Card
            title="Risk trajectory"
            subtitle="Updates live as the replay re-scores"
            actions={selected ? <RiskBadge probability={selected.mortality} /> : undefined}
          >
            <RiskTrajectoryChart
              mortality={trajectory.data?.points ?? []}
              threshold={trajectory.data?.threshold}
              height={260}
            />
          </Card>

          <Card title="Live vitals" subtitle="Most recent replayed hours">
            {ticks.length === 0 ? (
              <EmptyState
                icon="◷"
                title={running ? 'Waiting for the next hour…' : 'Not running'}
                description={
                  running ? undefined : 'Start the replay to stream hourly observations.'
                }
              />
            ) : (
              <ul className="space-y-2">
                {[...ticks].reverse().map((tick, index) => (
                  <li
                    key={`${tick.ts}-${index}`}
                    className={cn(
                      'rounded-lg border border-hairline p-2.5 transition',
                      index === 0 && 'border-[var(--series-mortality)]/50 bg-surface-3/40',
                    )}
                  >
                    <div className="mb-1 text-[11px] font-medium text-ink-muted">
                      {timeOnly(tick.ts)}
                    </div>
                    <dl className="grid grid-cols-2 gap-x-3 gap-y-1">
                      {WATCHED.map((key) => {
                        const value = tick.values[key];
                        if (value === null || value === undefined) return null;
                        return (
                          <div key={key} className="flex items-baseline justify-between gap-2">
                            <dt className="truncate text-[11px] text-ink-muted">
                              {key.replace(' blood pressure', ' BP')}
                            </dt>
                            <dd className="tnum text-xs font-medium text-ink">{String(value)}</dd>
                          </div>
                        );
                      })}
                    </dl>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}

      {/* Alert toasts */}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
        {toasts.map((toast) => {
          const info = SEVERITY_INFO[toast.severity];
          return (
            <div
              key={toast.id}
              role="alert"
              className="pointer-events-auto card flex items-start gap-2 p-3 shadow-xl"
            >
              <span aria-hidden="true" style={{ color: info.cssVar }}>
                {info.icon}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold text-ink">{info.label} alert</div>
                <div className="mt-0.5 text-[11px] text-ink-secondary">{toast.rule}</div>
              </div>
              <span className="tnum text-xs font-medium text-ink">{percent(toast.value)}</span>
            </div>
          );
        })}
      </div>

      {status && (
        <dl className="flex flex-wrap gap-6 px-1 text-xs">
          <Field label="Replayed hours">
            <span className="tnum">
              {status.cursorHour}/{status.totalHours}
            </span>
          </Field>
          <Field label="Scorings">
            <span className="tnum">{trajectory.data?.points.length ?? 0}</span>
          </Field>
        </dl>
      )}
    </div>
  );
}
