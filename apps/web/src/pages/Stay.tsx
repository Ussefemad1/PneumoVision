import type {
  Image,
  Note,
  Prediction,
  RiskTrajectory,
  StayDetail,
  Task,
  VitalsPoint,
} from '@pneumovision/shared';
import { EHR_CONTINUOUS_VARIABLES, EHR_VARIABLES, TEXT_CHUNK_TOKENS } from '@pneumovision/shared';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';

import { apiErrorMessage, getData, postData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent, shortDateTime } from '../lib/format.js';
import { getSocket } from '../lib/socket.js';
import {
  ConfidenceHeatmapOverlay,
  ConfidenceScaleLegend,
  EhrConfidenceBand,
  RiskTrajectoryChart,
} from '../components/charts.jsx';
import { NoteReader } from '../components/NoteReader.jsx';
import { ModalityBadges, RiskGauge } from '../components/risk.jsx';
import {
  Button,
  Card,
  EmptyState,
  ErrorState,
  Field,
  Skeleton,
  SkeletonRows,
} from '../components/ui.jsx';

const TABS = ['overview', 'vitals', 'imaging', 'notes', 'predictions'] as const;
type Tab = (typeof TABS)[number];

export function StayPage() {
  const { stayId = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const tab: Tab = TABS.find((t) => t === params.get('tab')) ?? 'overview';
  const queryClient = useQueryClient();

  const stayQuery = useQuery({
    queryKey: ['stay', stayId],
    queryFn: () => getData<StayDetail>(`/stays/${stayId}`),
    enabled: Boolean(stayId),
  });

  // Live updates for this patient: join the room, refresh on any prediction.
  useEffect(() => {
    if (!stayId) return;
    const socket = getSocket();
    socket.emit('subscribe:stay', stayId);
    const onDone = () => {
      void queryClient.invalidateQueries({ queryKey: ['stay', stayId] });
      void queryClient.invalidateQueries({ queryKey: ['trajectory', stayId] });
      void queryClient.invalidateQueries({ queryKey: ['predictions', stayId] });
    };
    socket.on('prediction:done', onDone);
    return () => {
      socket.emit('unsubscribe:stay', stayId);
      socket.off('prediction:done', onDone);
    };
  }, [stayId, queryClient]);

  if (stayQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (stayQuery.error || !stayQuery.data) {
    return (
      <ErrorState
        message={apiErrorMessage(stayQuery.error)}
        onRetry={() => void stayQuery.refetch()}
      />
    );
  }

  const stay = stayQuery.data;

  return (
    <div className="space-y-4">
      <StayHeader stay={stay} />

      <nav className="flex gap-1 border-b border-hairline" role="tablist">
        {TABS.map((item) => (
          <button
            key={item}
            role="tab"
            aria-selected={tab === item}
            onClick={() => setParams(item === 'overview' ? {} : { tab: item })}
            className={cn(
              '-mb-px border-b-2 px-3 py-2 text-sm capitalize transition',
              tab === item
                ? 'border-[var(--series-mortality)] font-medium text-ink'
                : 'border-transparent text-ink-secondary hover:text-ink',
            )}
          >
            {item}
          </button>
        ))}
      </nav>

      {tab === 'overview' && <OverviewTab stay={stay} />}
      {tab === 'vitals' && <VitalsTab stayId={stayId} />}
      {tab === 'imaging' && <ImagingTab stayId={stayId} />}
      {tab === 'notes' && <NotesTab stayId={stayId} />}
      {tab === 'predictions' && <PredictionsTab stayId={stayId} />}
    </div>
  );
}

function StayHeader({ stay }: { stay: StayDetail }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold text-ink">{stay.patient.pseudoId}</h1>
            <span className="rounded-full bg-surface-3 px-2 py-0.5 text-[11px] capitalize text-ink-secondary">
              {stay.status}
            </span>
          </div>
          <p className="mt-0.5 text-sm text-ink-muted">
            {stay.patient.demographics.age}y · {stay.patient.demographics.sex} · {stay.ward} · bed{' '}
            {stay.bedLabel}
          </p>
        </div>

        <dl className="flex flex-wrap items-start gap-6">
          <Field label="Admitted">{shortDateTime(stay.admittedAt)}</Field>
          <Field label="Open alerts">
            <span className="tnum">{stay.openAlertCount}</span>
          </Field>
          <div>
            <dt className="text-[11px] uppercase tracking-wide text-ink-muted">Modalities</dt>
            <dd className="mt-1">
              <ModalityBadges availability={stay.availability} />
            </dd>
          </div>
        </dl>
      </div>
    </Card>
  );
}

// ── Overview ────────────────────────────────────────────────────────────────

function OverviewTab({ stay }: { stay: StayDetail }) {
  const queryClient = useQueryClient();
  const [cutoff, setCutoff] = useState('');
  const [runError, setRunError] = useState<string | null>(null);

  const mortalityTrajectory = useQuery({
    queryKey: ['trajectory', stay.id, 'mortality'],
    queryFn: () =>
      getData<RiskTrajectory>(`/stays/${stay.id}/risk-trajectory`, { task: 'mortality' }),
  });
  const pneumoniaTrajectory = useQuery({
    queryKey: ['trajectory', stay.id, 'pneumonia'],
    queryFn: () =>
      getData<RiskTrajectory>(`/stays/${stay.id}/risk-trajectory`, { task: 'pneumonia' }),
  });

  const runPrediction = useMutation({
    mutationFn: (task: Task) =>
      postData<Prediction>(`/stays/${stay.id}/predictions`, {
        task,
        ...(cutoff ? { cutoffTime: new Date(cutoff).toISOString() } : {}),
      }),
    onSuccess: () => {
      setRunError(null);
      void queryClient.invalidateQueries({ queryKey: ['trajectory', stay.id] });
      void queryClient.invalidateQueries({ queryKey: ['stay', stay.id] });
      void queryClient.invalidateQueries({ queryKey: ['predictions', stay.id] });
    },
    onError: (err) => setRunError(apiErrorMessage(err, 'Could not run the prediction')),
  });

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card title="Current risk" subtitle="Calibrated probabilities from the active model">
        <div className="flex justify-around gap-2">
          <RiskGauge
            probability={stay.latestPredictions.mortality ?? null}
            label="Mortality"
            sublabel="48-hour"
          />
          <RiskGauge
            probability={stay.latestPredictions.pneumonia ?? null}
            label="Pneumonia"
            sublabel="phenotype"
          />
        </div>

        <div className="mt-5 space-y-2 border-t border-hairline pt-4">
          <label htmlFor="cutoff" className="block text-xs font-medium text-ink-secondary">
            Cut-off time
          </label>
          <input
            id="cutoff"
            type="datetime-local"
            value={cutoff}
            onChange={(e) => setCutoff(e.target.value)}
            className="w-full rounded-md border border-hairline bg-surface-2 px-2.5 py-1.5 text-sm text-ink"
          />
          <p className="text-[11px] text-ink-muted">
            Only data recorded before this instant is sent to the model. Leave empty to use now.
          </p>

          <div className="flex gap-2 pt-1">
            <Button
              variant="primary"
              size="sm"
              className="flex-1"
              disabled={runPrediction.isPending}
              onClick={() => runPrediction.mutate('mortality')}
            >
              {runPrediction.isPending ? 'Running…' : 'Run mortality'}
            </Button>
            <Button
              size="sm"
              className="flex-1"
              disabled={runPrediction.isPending}
              onClick={() => runPrediction.mutate('pneumonia')}
            >
              Run pneumonia
            </Button>
          </div>

          {runError && (
            <p role="alert" className="text-xs text-status-critical">
              {runError}
            </p>
          )}
        </div>
      </Card>

      <Card
        title="Risk trajectory"
        subtitle="Every scoring of this stay, with markers where new evidence arrived"
      >
        {mortalityTrajectory.isLoading ? (
          <Skeleton className="h-60 w-full" />
        ) : (mortalityTrajectory.data?.points.length ?? 0) === 0 ? (
          <EmptyState
            icon="◠"
            title="No predictions yet"
            description="Run a prediction to start the trajectory for this stay."
          />
        ) : (
          <>
            <RiskTrajectoryChart
              mortality={mortalityTrajectory.data?.points ?? []}
              pneumonia={pneumoniaTrajectory.data?.points ?? []}
              events={mortalityTrajectory.data?.events ?? []}
              threshold={mortalityTrajectory.data?.threshold}
            />
            <p className="mt-2 text-[11px] text-ink-muted">
              <span aria-hidden="true">▣</span> radiograph · <span aria-hidden="true">▤</span> note
            </p>
          </>
        )}
      </Card>
    </div>
  );
}

// ── Vitals ──────────────────────────────────────────────────────────────────

/** F7: the 17 variables, grouped the way a clinician scans them. */
const VITALS_GROUPS: { title: string; variables: string[] }[] = [
  {
    title: 'Haemodynamics',
    variables: [
      'Heart Rate',
      'Systolic blood pressure',
      'Diastolic blood pressure',
      'Mean blood pressure',
      'Capillary refill rate',
    ],
  },
  {
    title: 'Respiratory',
    variables: ['Respiratory rate', 'Oxygen saturation', 'Fraction inspired oxygen', 'pH'],
  },
  {
    title: 'Neurological (GCS)',
    variables: [
      'Glascow coma scale total',
      'Glascow coma scale eye opening',
      'Glascow coma scale motor response',
      'Glascow coma scale verbal response',
    ],
  },
  { title: 'Other', variables: ['Temperature', 'Glucose', 'Weight', 'Height'] },
];

function VitalsTab({ stayId }: { stayId: string }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['vitals', stayId],
    queryFn: () => getData<VitalsPoint[]>(`/stays/${stayId}/vitals`),
  });

  // The confidence band comes from the most recent completed prediction.
  const { data: predictions } = useQuery({
    queryKey: ['predictions', stayId],
    queryFn: () => getData<Prediction[]>(`/stays/${stayId}/predictions`),
  });
  const latest = predictions?.find((p) => p.status === 'done' && p.result);
  const confidence = latest?.result?.confidence;

  if (isLoading) return <SkeletonRows rows={6} />;
  if (error) return <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />;
  if (!data || data.length === 0) {
    return (
      <Card>
        <EmptyState
          icon="▤"
          title="No vitals recorded"
          description="This stay has no hourly observations. The EHR branch would be marked missing in any prediction."
        />
      </Card>
    );
  }

  // Show the most recent 48 hours — the window the model actually sees.
  const window = data.slice(-48);

  return (
    <div className="space-y-4">
      {confidence?.ehrTimesteps && (
        <Card
          title="Per-hour EHR confidence"
          subtitle={`From the latest ${latest?.task ?? ''} prediction · γ = max(σ(l̂), 1 − σ(l̂))`}
          actions={<ConfidenceScaleLegend theta={confidence.theta} />}
        >
          <EhrConfidenceBand values={confidence.ehrTimesteps} theta={confidence.theta} />
          <p className="mt-2 text-[11px] text-ink-muted">
            Hollow bars fall below θ and are pooled into the low-confidence group.
          </p>
        </Card>
      )}

      {VITALS_GROUPS.map((group) => (
        <Card key={group.title} title={group.title} bodyClassName="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-hairline bg-surface-3/60 text-left text-[11px] uppercase tracking-wide text-ink-muted">
                  <th className="sticky left-0 bg-surface-3 px-3 py-2 font-medium">Variable</th>
                  {window.map((point) => (
                    <th key={point.ts} className="px-2 py-2 text-center font-normal">
                      {new Date(point.ts).getHours()}h
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {group.variables.map((variable) => (
                  <tr key={variable} className="border-b border-hairline last:border-0">
                    <th
                      scope="row"
                      className="sticky left-0 whitespace-nowrap bg-surface-1 px-3 py-1.5 text-left text-xs font-medium text-ink-secondary"
                    >
                      {variable}
                      {!EHR_CONTINUOUS_VARIABLES.includes(variable as never) && (
                        <span className="ml-1 text-[10px] text-ink-muted">(cat)</span>
                      )}
                    </th>
                    {window.map((point) => {
                      const value = point.values[variable as keyof typeof point.values];
                      // A gap is a gap: never interpolated in the UI (F7).
                      return (
                        <td
                          key={point.ts}
                          className={cn(
                            'tnum px-2 py-1.5 text-center text-xs',
                            value === null || value === undefined
                              ? 'text-ink-muted/40'
                              : 'text-ink',
                          )}
                          title={value === null || value === undefined ? 'Not charted' : undefined}
                        >
                          {value === null || value === undefined
                            ? '·'
                            : typeof value === 'number'
                              ? value
                              : String(value).slice(0, 6)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ))}

      <p className="text-[11px] text-ink-muted">
        Showing the most recent {window.length} hourly bins of {EHR_VARIABLES.length} variables. A
        dot means nothing was charted that hour — gaps are shown, never interpolated.
      </p>
    </div>
  );
}

// ── Imaging ─────────────────────────────────────────────────────────────────

function ImagingTab({ stayId }: { stayId: string }) {
  const [overlay, setOverlay] = useState(0.55);
  const [selected, setSelected] = useState(0);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['images', stayId],
    queryFn: () => getData<Image[]>(`/stays/${stayId}/images`),
  });
  const { data: predictions } = useQuery({
    queryKey: ['predictions', stayId],
    queryFn: () => getData<Prediction[]>(`/stays/${stayId}/predictions`),
  });

  const grid = predictions?.find((p) => p.status === 'done' && p.result?.confidence.cxrPatchGrid)
    ?.result?.confidence;

  if (isLoading) return <SkeletonRows rows={3} />;
  if (error) return <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />;
  if (!data || data.length === 0) {
    return (
      <Card>
        <EmptyState
          icon="▣"
          title="No radiographs for this stay"
          description="Predictions will run with the CXR branch marked missing, and the report will show a reduced-input banner."
        />
      </Card>
    );
  }

  const image = data[Math.min(selected, data.length - 1)]!;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <Card
        title="Chest radiograph"
        subtitle={`${image.view} · ${shortDateTime(image.takenAt)}`}
        actions={grid?.cxrPatchGrid ? <ConfidenceScaleLegend theta={grid.theta} /> : undefined}
      >
        <div className="relative mx-auto aspect-square w-full max-w-xl overflow-hidden rounded-lg bg-black">
          <img
            src={image.url}
            alt={`Synthetic chest radiograph, ${image.view} view`}
            className="h-full w-full object-contain"
          />
          {grid?.cxrPatchGrid && overlay > 0 && (
            <ConfidenceHeatmapOverlay grid={grid.cxrPatchGrid} opacity={overlay} />
          )}
        </div>

        {grid?.cxrPatchGrid ? (
          <div className="mt-4 flex items-center gap-3">
            <label htmlFor="overlay" className="whitespace-nowrap text-xs text-ink-secondary">
              Heatmap opacity
            </label>
            <input
              id="overlay"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={overlay}
              onChange={(e) => setOverlay(Number(e.target.value))}
              className="flex-1 accent-[var(--series-mortality)]"
            />
            <span className="tnum w-10 text-right text-xs text-ink-muted">
              {Math.round(overlay * 100)}%
            </span>
          </div>
        ) : (
          <p className="mt-3 text-xs text-ink-muted">
            Run a prediction to generate the ViT patch-confidence overlay.
          </p>
        )}

        <p className="mt-3 text-[11px] text-ink-muted">
          {grid?.cxrPatchGrid
            ? `${grid.cxrPatchGrid.length}×${grid.cxrPatchGrid[0]?.length} patch grid from vit_small_patch16_384. `
            : ''}
          Images in this demo are procedurally generated, not real radiographs.
        </p>
      </Card>

      <Card
        title="Series"
        subtitle={`${data.length} study${data.length === 1 ? '' : 'ies'}`}
        bodyClassName="p-2"
      >
        <ul className="space-y-1">
          {data.map((item, index) => (
            <li key={item.id}>
              <button
                onClick={() => setSelected(index)}
                className={cn(
                  'flex w-full items-center gap-3 rounded-md p-2 text-left transition',
                  index === selected ? 'bg-surface-3' : 'hover:bg-surface-3/60',
                )}
              >
                <img
                  src={item.url}
                  alt=""
                  className="h-12 w-12 rounded border border-hairline object-cover"
                />
                <span className="min-w-0">
                  <span className="block text-xs font-medium text-ink">{item.view}</span>
                  <span className="block truncate text-[11px] text-ink-muted">
                    {shortDateTime(item.takenAt)}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

// ── Notes ───────────────────────────────────────────────────────────────────

function NotesTab({ stayId }: { stayId: string }) {
  const [openId, setOpenId] = useState<string | null>(null);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['notes', stayId],
    queryFn: () => getData<Note[]>(`/stays/${stayId}/notes`),
  });
  const { data: predictions } = useQuery({
    queryKey: ['predictions', stayId],
    queryFn: () => getData<Prediction[]>(`/stays/${stayId}/predictions`),
  });

  const confidence = predictions?.find((p) => p.status === 'done' && p.result?.confidence.noteSpans)
    ?.result?.confidence;

  const spansByNote = useMemo(() => {
    const map = new Map<
      string,
      typeof confidence extends undefined
        ? never
        : NonNullable<NonNullable<typeof confidence>['noteSpans']>
    >();
    for (const span of confidence?.noteSpans ?? []) {
      const list = map.get(span.noteId) ?? [];
      list.push(span);
      map.set(span.noteId, list);
    }
    return map;
  }, [confidence]);

  if (isLoading) return <SkeletonRows rows={4} />;
  if (error) return <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />;
  if (!data || data.length === 0) {
    return (
      <Card>
        <EmptyState
          icon="▤"
          title="No notes for this stay"
          description="Text branches will be marked missing in any prediction for this patient."
        />
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      {confidence && (
        <div className="flex justify-end">
          <ConfidenceScaleLegend theta={confidence.theta} />
        </div>
      )}

      {data.map((note) => {
        const spans = spansByNote.get(note.id) ?? [];
        const chunks = Math.ceil(note.tokenCount / TEXT_CHUNK_TOKENS);
        const isOpen = openId === note.id;

        return (
          <Card key={note.id} bodyClassName="p-0">
            <button
              onClick={() => setOpenId(isOpen ? null : note.id)}
              className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-surface-3/50"
              aria-expanded={isOpen}
            >
              <span className="rounded bg-surface-3 px-2 py-0.5 text-[11px] font-medium capitalize text-ink-secondary">
                {note.type}
              </span>
              {note.type === 'discharge' && (
                <span
                  title="Discharge notes are never sent for a mortality prediction (F8)"
                  className="rounded bg-status-warning/20 px-2 py-0.5 text-[11px] font-medium text-[#8a5d00] dark:text-status-warning"
                >
                  <span aria-hidden="true">◆</span> excluded from mortality
                </span>
              )}
              <span className="text-xs text-ink-muted">{shortDateTime(note.authoredAt)}</span>
              <span className="tnum ml-auto text-[11px] text-ink-muted">
                ~{note.tokenCount} tokens
                {chunks > 1 && (
                  <span
                    className="ml-1.5 text-[#8a5d00] dark:text-status-warning"
                    title={`BioBERT will split this into ${chunks} chunks of ${TEXT_CHUNK_TOKENS} tokens and mean-pool them`}
                  >
                    · {chunks} chunks
                  </span>
                )}
              </span>
              <span aria-hidden="true" className="text-ink-muted">
                {isOpen ? '▾' : '▸'}
              </span>
            </button>

            {isOpen && (
              <div className="border-t border-hairline bg-surface-2/40 px-4 py-3">
                {spans.length === 0 && confidence && (
                  <p className="mb-2 text-[11px] text-ink-muted">
                    No confidence spans were returned for this note.
                  </p>
                )}
                <NoteReader text={note.text} spans={spans} theta={confidence?.theta ?? 0.75} />
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
}

// ── Predictions ─────────────────────────────────────────────────────────────

function PredictionsTab({ stayId }: { stayId: string }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['predictions', stayId],
    queryFn: () => getData<Prediction[]>(`/stays/${stayId}/predictions`),
  });

  if (isLoading) return <SkeletonRows rows={6} />;
  if (error) return <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />;
  if (!data || data.length === 0) {
    return (
      <Card>
        <EmptyState
          icon="◷"
          title="No predictions yet"
          description="Use “Run prediction” on the Overview tab to score this stay."
        />
      </Card>
    );
  }

  return (
    <Card bodyClassName="p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-hairline bg-surface-3/60 text-left text-[11px] uppercase tracking-wide text-ink-muted">
              <th className="px-4 py-2 font-medium">Cut-off</th>
              <th className="px-4 py-2 font-medium">Task</th>
              <th className="px-4 py-2 font-medium">Risk</th>
              <th className="px-4 py-2 font-medium">Status</th>
              <th className="px-4 py-2 font-medium">Model</th>
              <th className="px-4 py-2 font-medium">Latency</th>
              <th className="px-4 py-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {data.map((prediction) => (
              <tr
                key={prediction.id}
                className="border-b border-hairline last:border-0 hover:bg-surface-3/50"
              >
                <td className="px-4 py-2 text-ink-secondary">
                  {shortDateTime(prediction.cutoffTime)}
                </td>
                <td className="px-4 py-2 capitalize text-ink-secondary">{prediction.task}</td>
                <td className="tnum px-4 py-2 font-medium text-ink">
                  {percent(prediction.result?.probability)}
                </td>
                <td className="px-4 py-2">
                  <span
                    className={cn(
                      'rounded-full px-2 py-0.5 text-[11px] font-medium',
                      prediction.status === 'done' && 'bg-status-good/15 text-status-good',
                      prediction.status === 'failed' &&
                        'bg-status-critical/15 text-status-critical',
                      (prediction.status === 'queued' || prediction.status === 'running') &&
                        'bg-surface-3 text-ink-secondary',
                    )}
                  >
                    {prediction.status}
                  </span>
                </td>
                <td className="px-4 py-2 text-xs text-ink-muted">{prediction.modelVersion}</td>
                <td className="tnum px-4 py-2 text-xs text-ink-muted">
                  {prediction.latencyMs === null ? '—' : `${prediction.latencyMs} ms`}
                </td>
                <td className="px-4 py-2 text-right">
                  {prediction.status === 'done' && (
                    <Link
                      to={`/predictions/${prediction.id}`}
                      className="text-xs font-medium text-[var(--series-mortality)] hover:underline"
                    >
                      Report →
                    </Link>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
