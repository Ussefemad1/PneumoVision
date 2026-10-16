import type { Modality, ModalitySource, Prediction, StayDetail } from '@pneumovision/shared';
import { TASK_SPEC } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { apiErrorMessage, getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent } from '../lib/format.js';
import { riskBand } from '../lib/risk.js';
import { AlphaContributionChart } from '../components/charts.jsx';
import { Disclaimer } from '../components/Disclaimer.jsx';
import { ReceivedInputs } from '../features/analyze/ReceivedInputs.jsx';
import { RiskGauge } from '../components/risk.jsx';
import { Button, Card, ErrorState, Skeleton } from '../components/ui.jsx';

const MODALITY_LABELS: Record<string, string> = {
  ehr: 'EHR (vitals)',
  cxr: 'Chest X-ray',
  rr: 'Radiology reports',
  dn: 'Discharge notes',
};

/**
 * The prediction report.
 *
 * F11: the layout mirrors how clinicians actually reason — each source read
 * on its own first (left), then the evidence pooled by confidence, then how
 * the branches were weighted, and only at the end the fused decision (right).
 */
export function PredictionReportPage() {
  const { predictionId = '' } = useParams();

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['prediction', predictionId],
    queryFn: () => getData<Prediction>(`/predictions/${predictionId}`),
  });

  const { data: stay } = useQuery({
    queryKey: ['stay', data?.stayId],
    queryFn: () => getData<StayDetail>(`/stays/${data!.stayId}`),
    enabled: Boolean(data?.stayId),
  });

  if (isLoading) return <Skeleton className="h-96 w-full" />;
  if (error || !data) {
    return <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />;
  }

  if (data.status !== 'done' || !data.result) {
    return (
      <Card title="Prediction not available">
        <p className="text-sm text-ink-secondary">
          This prediction is <span className="font-medium">{data.status}</span>.
          {data.error && <span className="block mt-2 text-status-critical">{data.error}</span>}
        </p>
      </Card>
    );
  }

  const result = data.result;
  // Only the modalities this task's model takes: mortality never uses
  // discharge notes (F8), so their absence is not a "reduced input".
  const taskModalities = TASK_SPEC[data.task].modalities.map((m) => m.toLowerCase() as Modality);
  const missing = taskModalities
    .filter((key) => !result.missingness.vector[key])
    .map((key) => MODALITY_LABELS[key] ?? key);

  // Old predictions carry no provenance: everything was simulated then.
  const sources = result.sources;
  const fusionKind = sources?.fusion ?? 'mock';
  // Joint, missingness and α come from the fusion model; only a trained one
  // makes them real.
  const fusionSimulated = fusionKind !== 'trained';
  const trained = taskModalities.filter((key) => sources?.[key]?.kind === 'trained');
  const averaged = trained.filter((key) => typeof result.unimodal[key] === 'number');
  const simulatedTag = fusionSimulated ? <SimulatedChip /> : undefined;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3 print:block">
        <div>
          <div className="flex items-center gap-2 text-sm text-ink-muted">
            {stay && (
              <Link
                to={stay.status === 'adhoc' ? '/analyze' : `/stays/${data.stayId}`}
                className="hover:underline"
              >
                {stay.status === 'adhoc' ? 'Analyses' : stay.patient.pseudoId}
              </Link>
            )}
            <span>/</span>
            <span className="capitalize">{data.task} prediction</span>
          </div>
          <h1 className="mt-0.5 text-lg font-semibold text-ink">Prediction report</h1>
        </div>
        <Button size="sm" onClick={() => window.print()} className="print:hidden">
          Print / save as PDF
        </Button>
      </div>

      {/* F5: reduced-input banner */}
      {missing.length > 0 && (
        <div
          role="note"
          className="flex items-start gap-2 rounded-lg border border-status-warning/40 bg-status-warning/10 px-4 py-3 text-sm"
        >
          <span aria-hidden="true" className="text-[#8a5d00] dark:text-status-warning">
            ◆
          </span>
          <p>
            <span className="font-medium text-ink">Reduced input</span>
            <span className="text-ink-secondary"> · Missing: {missing.join(', ')}</span>
          </p>
        </div>
      )}


      {/* F11: evidence left → decision right */}
      <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr_0.9fr]">
        {/* F1 */}
        <Card title="By source">
          <ul className="space-y-2">
            {taskModalities.map((key) => {
              const value = result.unimodal[key];
              const present = result.missingness.vector[key];
              const itemBand = riskBand(value ?? null);
              return (
                <li
                  key={key}
                  className={cn(
                    'flex items-center gap-3 rounded-lg border border-hairline px-3 py-2.5',
                    !present && 'opacity-60',
                  )}
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-ink">{MODALITY_LABELS[key]}</div>
                    {present && sources?.[key] && <SourceChip source={sources[key]} />}
                  </div>
                  {present && typeof value === 'number' ? (
                    <div className="text-right">
                      <div className="tnum text-sm font-semibold text-ink">{percent(value)}</div>
                      <div
                        className="flex items-center justify-end gap-1 text-[11px]"
                        style={{ color: itemBand.cssVar }}
                      >
                        <span aria-hidden="true">{itemBand.icon}</span>
                        {itemBand.label}
                      </div>
                    </div>
                  ) : (
                    <span className="text-xs text-ink-muted">— missing</span>
                  )}
                </li>
              );
            })}
          </ul>
        </Card>

        {/* F4 + F2 */}
        <Card
          title="Confidence"
          actions={simulatedTag}
        >
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-hairline p-3">
              <div className="text-[11px] uppercase tracking-wide text-ink-muted">
                High confidence
              </div>
              <div className="tnum mt-1 text-xl font-semibold text-ink">
                {percent(result.joint.high)}
              </div>
            </div>
            <div className="rounded-lg border border-dashed border-hairline p-3">
              <div className="text-[11px] uppercase tracking-wide text-ink-muted">
                Low confidence
              </div>
              <div className="tnum mt-1 text-xl font-semibold text-ink-secondary">
                {percent(result.joint.low)}
              </div>
            </div>
          </div>

          <div className="mt-4">
            <div className="mb-2 text-[11px] uppercase tracking-wide text-ink-muted">
              Above threshold
            </div>
            <ul className="space-y-1.5">
              {taskModalities.map((key) => {
                const fraction = result.confidence.fractionAbove[key];
                if (typeof fraction !== 'number') return null;
                return (
                  <li key={key} className="grid grid-cols-[6.5rem_1fr_auto] items-center gap-2">
                    <span className="truncate text-xs text-ink-secondary">
                      {MODALITY_LABELS[key]}
                    </span>
                    <span className="h-2 overflow-hidden rounded-full bg-surface-3">
                      <span
                        className="block h-full rounded-full bg-[var(--seq-400)]"
                        style={{ width: `${Math.max(2, fraction * 100)}%` }}
                      />
                    </span>
                    <span className="tnum w-11 text-right text-xs text-ink">
                      {percent(fraction)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="mt-4 rounded-lg bg-surface-2 p-3">
            <div className="text-[11px] uppercase tracking-wide text-ink-muted">
              Missingness branch
            </div>
            <div className="tnum mt-0.5 text-sm font-medium text-ink">
              {percent(result.missingness.prediction)}
            </div>
          </div>
        </Card>

        {/* F6 + fused decision */}
        <div className="space-y-4">
          <Card
            title="Contribution"
            actions={simulatedTag}
          >
            <AlphaContributionChart alphas={result.alphas} />
          </Card>

          <Card
            className="border-2"
            title={fusionKind === 'average' ? 'Combined score' : 'Fused decision'}
            actions={fusionKind === 'average' ? <AverageChip /> : simulatedTag}
          >
            <div className="flex flex-col items-center gap-3">
              <RiskGauge
                probability={result.probability}
                label={data.task === 'mortality' ? '48-hour mortality' : 'Pneumonia phenotype'}
                sublabel={
                  fusionKind === 'average'
                    ? 'Average'
                    : fusionKind === 'mock'
                      ? 'Simulated'
                      : 'Calibrated'
                }
              />
              {fusionKind === 'average' ? (
                <div className="w-full text-center text-xs text-ink-secondary">
                  <p className="tnum rounded-md bg-surface-2 px-2 py-1.5 text-ink">
                    ({averaged.map((key) => percent(result.unimodal[key])).join(' + ')}) ÷{' '}
                    {averaged.length} = {percent(result.probability)}
                  </p>
                </div>
              ) : (
                null
              )}
            </div>
          </Card>
        </div>
      </div>

      <ReceivedInputs prediction={data} />

      <Disclaimer />
    </div>
  );
}

/** Whether one modality's score came from a trained reader or is simulated. */
function SourceChip({ source }: { source: ModalitySource }) {
  if (source.kind === 'trained') {
    return (
      <span
        title={source.model ?? undefined}
        className="mt-1 inline-flex items-center gap-1 rounded-full bg-[var(--series-mortality)]/12 px-2 py-0.5 text-[10px] font-semibold text-[var(--series-mortality)]"
      >
        <span aria-hidden="true">✓</span> Trained
        <span className="tnum font-normal">
          ·{' '}
          {source.testAuroc !== null
            ? `AUROC ${source.testAuroc.toFixed(3)}`
            : 'AUROC pending'}
        </span>
      </span>
    );
  }
  return <SimulatedChip className="mt-1" />;
}

function AverageChip() {
  return (
    <span
      title="The plain mean of the trained models' scores. Not the MedPatch fusion."
      className="inline-flex items-center gap-1 rounded-full bg-[var(--series-mortality)]/12 px-2 py-0.5 text-[10px] font-semibold text-[var(--series-mortality)]"
    >
      <span aria-hidden="true">=</span> Average
    </span>
  );
}

function SimulatedChip({ className }: { className?: string | undefined }) {
  return (
    <span
      title="Simulated: no trained model for this part yet"
      className={cn(
        'inline-flex items-center gap-1 rounded-full border border-dashed border-hairline px-2 py-0.5 text-[10px] font-medium text-ink-muted',
        className,
      )}
    >
      <span aria-hidden="true">◇</span> Simulated
    </span>
  );
}
