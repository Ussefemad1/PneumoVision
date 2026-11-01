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
import { Button, Card, EmptyState, Field, Kpi, Segmented } from '../components/ui.jsx';

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
    <div className="flex flex-col gap-[22px]">
      <header>
        <h1 className="text-[28px] font-bold tracking-[-0.02em] text-ink">ICU Replay</h1>
        <p className="mt-1.5 text-ink-secondary">
          Step a past stay forward hour by hour and watch the risk move. Demo only.
        </p>
      </header>

      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[220px] flex-1">
            <label htmlFor="stay" className="eyebrow mb-1.5 block">
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
              className="field-input w-full text-sm"
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
            <span className="eyebrow mb-1.5 block">Speed</span>
            <Segmented
              value={String(speed)}
              onChange={(value) => setSpeed(Number(value))}
              options={REPLAY_SPEEDS.map((option) => ({
                value: String(option),
                label: `×${option}`,
              }))}
              ariaLabel="Replay speed"
            />
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
          <p role="alert" className="note-critical mt-3 px-3 py-2 text-xs">
            {error}
          </p>
        )}

        {status && (
          <div className="mt-4">
            <div className="mb-1.5 flex items-center justify-between text-xs text-ink-secondary">
              <span>
                Hour {status.cursorHour} of {status.totalHours}
              </span>
              <span>{running ? `running at ×${status.speed}` : 'stopped'}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-softer">
              <div
                className="h-full rounded-full bg-accent transition-all"
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
            description="Their recorded hours are fed back through the model in order."
          />
        </Card>
      ) : (
        <>
          {/* All three come from state the page already holds. */}
          {status && (
            <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-4">
              <Kpi
                label="Hour"
                value={`${status.cursorHour} of ${status.totalHours}`}
                hint={running ? `Running at ×${status.speed}` : 'Stopped'}
              />
              <Kpi
                label="Current risk"
                value={percent(selected?.mortality ?? null)}
                hint="In-hospital mortality"
              />
              <Kpi
                label="Hours replayed"
                value={ticks.length}
                hint="Vitals received this session"
              />
            </div>
          )}
          <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
            <Card
              title="Risk trajectory"
              actions={selected ? <RiskBadge probability={selected.mortality} /> : undefined}
            >
              <RiskTrajectoryChart
                mortality={trajectory.data?.points ?? []}
                threshold={trajectory.data?.threshold}
                height={260}
              />
            </Card>

            <Card title="Live vitals">
              {ticks.length === 0 ? (
                <EmptyState
                  icon="◷"
                  title={running ? 'Waiting for the next hour…' : 'Not running'}
                />
              ) : (
                <ul className="space-y-2">
                  {[...ticks].reverse().map((tick, index) => (
                    <li
                      key={`${tick.ts}-${index}`}
                      className={cn(
                        'rounded-lg border border-hairline p-2.5 transition',
                        index === 0 && 'border-accent bg-softer',
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
        </>
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
              <span aria-hidden="true" style={{ color: info.inkVar }}>
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
