import type { Prediction, StayDetail } from '@pneumovision/shared';
import { MEDPATCH_REFERENCE_METRICS } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { apiErrorMessage, getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { durationMs, percent, shortDateTime } from '../lib/format.js';
import { riskBand } from '../lib/risk.js';
import { AlphaContributionChart } from '../components/charts.jsx';
import { Disclaimer } from '../components/Disclaimer.jsx';
import { RiskGauge } from '../components/risk.jsx';
import { Button, Card, ErrorState, Field, Skeleton } from '../components/ui.jsx';

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
  const missing = Object.entries(result.missingness.vector)
    .filter(([, present]) => !present)
    .map(([key]) => MODALITY_LABELS[key] ?? key);

  const band = riskBand(result.probability);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3 print:block">
        <div>
          <div className="flex items-center gap-2 text-sm text-ink-muted">
            {stay && (
              <Link to={`/stays/${data.stayId}`} className="hover:underline">
                {stay.patient.pseudoId}
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
          <div>
            <p className="font-medium text-ink">Reduced-input prediction</p>
            <p className="mt-0.5 text-ink-secondary">
              This score was produced without {missing.join(', ')}. The missingness branch accounted
              for the absent {missing.length === 1 ? 'source' : 'sources'}, but the result is less
              well supported than a full multimodal prediction.
            </p>
          </div>
        </div>
      )}

      {/* F11: evidence left → decision right */}
      <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr_0.9fr]">
        {/* F1 */}
        <Card title="1 · Evidence by source" subtitle="What each modality predicts on its own">
          <ul className="space-y-2">
            {(['ehr', 'cxr', 'rr', 'dn'] as const).map((key) => {
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
                    <div className="text-[11px] text-ink-muted">
                      {present ? 'Contributed to the fused score' : 'Not available for this stay'}
                    </div>
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
          title="2 · Evidence strength"
          subtitle={`Tokens split at θ = ${result.confidence.theta.toFixed(2)}`}
        >
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-hairline p-3">
              <div className="text-[11px] uppercase tracking-wide text-ink-muted">
                High confidence
              </div>
              <div className="tnum mt-1 text-xl font-semibold text-ink">
                {percent(result.joint.high)}
              </div>
              <p className="mt-1 text-[11px] text-ink-muted">
                Pooled from tokens the model was confident about
              </p>
            </div>
            <div className="rounded-lg border border-dashed border-hairline p-3">
              <div className="text-[11px] uppercase tracking-wide text-ink-muted">
                Low confidence
              </div>
              <div className="tnum mt-1 text-xl font-semibold text-ink-secondary">
                {percent(result.joint.low)}
              </div>
              <p className="mt-1 text-[11px] text-ink-muted">
                Pooled from the remaining, uncertain tokens
              </p>
            </div>
          </div>

          <div className="mt-4">
            <div className="mb-2 text-[11px] uppercase tracking-wide text-ink-muted">
              Share of tokens above θ
            </div>
            <ul className="space-y-1.5">
              {(['ehr', 'cxr', 'rr', 'dn'] as const).map((key) => {
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
            <p className="mt-1 text-[11px] text-ink-muted">
              What the pattern of available sources alone implies
            </p>
          </div>
        </Card>

        {/* F6 + fused decision */}
        <div className="space-y-4">
          <Card title="3 · Modality contribution" subtitle="Softmax-normalised fusion weights (α̂)">
            <AlphaContributionChart alphas={result.alphas} />
          </Card>

          <Card className="border-2" title="4 · Fused decision">
            <div className="flex flex-col items-center gap-3">
              <RiskGauge
                probability={result.probability}
                label={data.task === 'mortality' ? '48-hour mortality' : 'Pneumonia phenotype'}
                sublabel="Calibrated"
              />
              <p className="text-center text-xs text-ink-secondary">
                Weighted combination of every branch above.{' '}
                <span style={{ color: band.cssVar }} className="font-medium">
                  {band.label} risk
                </span>
                .
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* Provenance */}
      <Card title="Provenance">
        <dl className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Field label="Model version">{data.modelVersion}</Field>
          <Field label="Cut-off time">{shortDateTime(data.cutoffTime)}</Field>
          <Field label="Latency">{durationMs(data.latencyMs)}</Field>
          <Field label="Requested">{shortDateTime(data.createdAt)}</Field>
          <Field label="θ">{result.confidence.theta.toFixed(2)}</Field>
          <Field label="Input hash">
            <code className="text-[11px]">{data.inputHash.slice(0, 12)}…</code>
          </Field>
        </dl>

        <p className="mt-4 border-t border-hairline pt-3 text-[11px] text-ink-muted">
          Reference (MedPatch paper, not this model):{' '}
          {data.task === 'mortality'
            ? `trimodal mortality AUROC ${MEDPATCH_REFERENCE_METRICS.mortality.auroc} / AUPRC ${MEDPATCH_REFERENCE_METRICS.mortality.auprc}`
            : `pneumonia phenotype AUROC ${MEDPATCH_REFERENCE_METRICS.pneumonia.auroc}`}
          . This prototype currently runs a mock model — the numbers above are not clinically
          meaningful.
        </p>
      </Card>

      <Disclaimer />
    </div>
  );
}
