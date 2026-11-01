import type {
  FusionInternals,
  FusionModelInfo,
  LogitProb,
  Modality,
  ModalitySource,
  Prediction,
  StayDetail,
} from '@pneumovision/shared';
import { TASK_SPEC } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { apiErrorMessage, getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { durationMs, percent, shortDateTime } from '../lib/format.js';
import { riskBand } from '../lib/risk.js';
import { AlphaContributionChart } from '../components/charts.jsx';
import { Disclaimer } from '../components/Disclaimer.jsx';
import { ReceivedInputs } from '../features/analyze/ReceivedInputs.jsx';
import { RiskGauge } from '../components/risk.jsx';
import { Button, Card, Chip, ErrorState, Skeleton } from '../components/ui.jsx';

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
  // Token confidences can be real (trained confidence heads) while the
  // high/low groups and missingness still need the fusion model.
  const realConfidence = taskModalities.some((key) => sources?.[key]?.confidence);
  // Fusion mode: the trained MedPatch model (Round 3). Its token-confidence
  // maps on this page are still simulated (decision 8, REAL_INFERENCE_PLAN).
  const fusionModel = result.fusionModel;
  // Fusion results stored before the standalone readers carried the fusion's
  // internal votes in `unimodal`: never show those as per-source scores.
  const legacyFusion = fusionModel !== null && fusionModel.standalone === null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3 print:block">
        <div>
          <div className="flex items-center gap-2 text-xs text-ink-secondary">
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
          <h1 className="mt-1 text-[28px] font-bold tracking-[-0.02em] text-ink">
            Prediction report
          </h1>
        </div>
        <Button size="sm" onClick={() => window.print()} className="print:hidden">
          Print / save as PDF
        </Button>
      </div>

      {/* The decision, up front. Deliberately no risk chip beside the gauge:
          RiskBadge would render a second element whose exact text is the
          probability, and PredictionReport.test.tsx does getByText on it. */}
      <Card
        className="border-2"
        title={fusionKind === 'average' ? 'Combined score' : 'Fused decision'}
        actions={
          fusionKind === 'average' ? (
            <AverageChip />
          ) : fusionModel ? (
            <FusionChip runId={fusionModel.runId} />
          ) : (
            simulatedTag
          )
        }
        bodyClassName="flex flex-wrap items-center gap-8 p-6"
      >
        <RiskGauge
          probability={result.probability}
          label={data.task === 'mortality' ? '48-hour mortality' : 'Pneumonia phenotype'}
          sublabel={
            fusionKind === 'average'
              ? 'Average'
              : fusionKind === 'mock'
                ? 'Simulated'
                : fusionModel && !fusionModel.calibrated
                  ? 'Uncalibrated probability'
                  : 'Calibrated'
          }
        />
        <div className="min-w-[260px] flex-1">
          <div className="eyebrow">
            {data.task === 'mortality' ? '48-hour mortality' : 'Pneumonia phenotype'}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-3">
            <p className="text-[26px] font-bold tracking-[-0.02em] text-ink">
              {riskBand(result.probability).label} risk
            </p>
            {/*
             * Level only, never the percentage: PredictionReport.test.tsx does
             * getByText on a probability string, which throws on a second
             * match. RiskBadge would render exactly that.
             */}
            <Chip
              className={cn(
                riskBand(result.probability).chipClass,
                'h-7 px-3 text-[13px] font-semibold',
              )}
            >
              <span aria-hidden="true">{riskBand(result.probability).icon}</span>
              {riskBand(result.probability).label}
            </Chip>
          </div>
          {fusionKind === 'average' ? (
            <p className="tnum mt-3 rounded-control bg-surface-3 px-3 py-2 text-xs text-ink">
              ({averaged.map((key) => percent(result.unimodal[key])).join(' + ')}) ÷{' '}
              {averaged.length} = {percent(result.probability)}
            </p>
          ) : fusionModel ? (
            <div className="mt-3">
              <FusionDetails model={fusionModel} />
            </div>
          ) : null}
        </div>

        {/* Everything here is already on the prediction or the stay query. */}
        <dl className="grid shrink-0 grid-cols-[auto_auto] gap-x-5 gap-y-2 text-[13px]">
          {stay && (
            <>
              <dt className="text-ink-secondary">Patient</dt>
              <dd className="font-medium text-ink">{stay.patient.pseudoId}</dd>
            </>
          )}
          <dt className="text-ink-secondary">Cut-off</dt>
          <dd className="tnum text-ink">{shortDateTime(data.cutoffTime)}</dd>
          <dt className="text-ink-secondary">Model</dt>
          <dd className="text-ink">{data.modelVersion}</dd>
          {data.latencyMs !== null && (
            <>
              <dt className="text-ink-secondary">Latency</dt>
              <dd className="tnum text-ink">{durationMs(data.latencyMs)}</dd>
            </>
          )}
        </dl>
      </Card>

      {/* F5: reduced-input banner */}
      {missing.length > 0 && (
        <div role="note" className="note-warning flex items-start gap-2 px-4 py-3 text-sm">
          <span aria-hidden="true" className="text-status-ink-warning">
            ◆
          </span>
          <p>
            <span className="font-medium text-ink">Reduced input</span>
            <span className="text-ink-secondary"> · Missing: {missing.join(', ')}</span>
          </p>
        </div>
      )}

      {/* F11: evidence left → decision right */}
      <div className="grid gap-4 xl:grid-cols-3 xl:items-start">
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
                    {present && sources?.[key] && !legacyFusion && (
                      <SourceChip source={sources[key]} inFusion={fusionModel !== null} />
                    )}
                  </div>
                  {present && fusionModel && legacyFusion ? (
                    <span className="max-w-[11rem] text-right text-xs text-ink-muted">
                      Older result: no standalone score (re-run to see it)
                    </span>
                  ) : present && fusionModel?.standalone === 'not-installed' ? (
                    <span className="max-w-[11rem] text-right text-xs text-ink-muted">
                      Standalone readers not installed
                    </span>
                  ) : present && fusionModel && typeof value !== 'number' ? (
                    <span className="max-w-[11rem] text-right text-xs text-ink-muted">
                      Standalone reader unavailable
                    </span>
                  ) : present && typeof value === 'number' ? (
                    <div className="text-right">
                      <div className="tnum text-sm font-semibold text-ink">{percent(value)}</div>
                      <div
                        className="flex items-center justify-end gap-1 text-[11px]"
                        style={{ color: itemBand.inkVar }}
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
        <Card title="Confidence" actions={realConfidence ? undefined : simulatedTag}>
          {realConfidence && fusionSimulated && (
            <div className="mb-2 flex justify-end">
              <SimulatedChip />
            </div>
          )}
          {fusionModel ? (
            <p className="rounded-lg bg-surface-2 p-3 text-xs text-ink-secondary">
              The fusion&apos;s joint high/low-confidence heads and its missingness head are
              internal outputs, not probabilities: they are under “Model details” with an
              explanation.
            </p>
          ) : (
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
          )}

          <div className="mt-4">
            <div className="mb-2 flex items-center justify-between text-[11px] uppercase tracking-wide text-ink-muted">
              Above threshold
              {fusionModel && !realConfidence && <SimulatedChip />}
            </div>
            <ul className="space-y-1.5">
              {taskModalities.map((key) => {
                const fraction = result.confidence.fractionAbove[key];
                if (typeof fraction !== 'number') return null;
                return (
                  <li
                    key={key}
                    className={cn(
                      'grid items-center gap-2',
                      realConfidence
                        ? 'grid-cols-[6.5rem_1fr_auto_5.5rem]'
                        : 'grid-cols-[6.5rem_1fr_auto]',
                    )}
                  >
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
                    {realConfidence && <ConfidenceTag level={sources?.[key]?.confidence ?? null} />}
                  </li>
                );
              })}
            </ul>
          </div>

          {!fusionModel && (
            <div className="mt-4 rounded-lg bg-surface-2 p-3">
              <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-ink-muted">
                Missingness branch
                {realConfidence && fusionSimulated && <SimulatedChip />}
              </div>
              <div className="tnum mt-0.5 text-sm font-medium text-ink">
                {percent(result.missingness.prediction)}
              </div>
            </div>
          )}
        </Card>

        {/* F6 */}
        <Card title="Contribution" actions={simulatedTag}>
          <AlphaContributionChart alphas={result.alphas} />
        </Card>
      </div>

      <ReceivedInputs prediction={data} />

      <Disclaimer />
    </div>
  );
}

/** Whether one modality's score came from a trained reader or is simulated. */
function SourceChip({ source, inFusion }: { source: ModalitySource; inFusion: boolean }) {
  if (source.kind === 'trained' && inFusion) {
    return (
      <span
        title={source.model ?? undefined}
        className="chip-accent mt-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold"
      >
        <span aria-hidden="true">✓</span> Standalone reader (Round 1)
        <span className="font-normal">· uncalibrated</span>
        {source.testAuroc !== null && (
          <span className="tnum font-normal">· AUROC {source.testAuroc.toFixed(3)}</span>
        )}
      </span>
    );
  }
  if (source.kind === 'trained') {
    return (
      <span
        title={source.model ?? undefined}
        className="chip-accent mt-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold"
      >
        <span aria-hidden="true">✓</span> Trained
        <span className="tnum font-normal">
          · {source.testAuroc !== null ? `AUROC ${source.testAuroc.toFixed(3)}` : 'AUROC pending'}
        </span>
      </span>
    );
  }
  return <SimulatedChip className="mt-1" />;
}

/** Where a row of token confidences came from. */
function ConfidenceTag({ level }: { level: 'calibrated' | 'uncalibrated' | null }) {
  if (level === null) return <SimulatedChip />;
  return (
    <span
      title={
        level === 'calibrated'
          ? 'Trained confidence head with temperature calibration (Round 2b)'
          : 'Trained confidence head, not yet calibrated (Round 2)'
      }
      className="inline-flex items-center justify-center gap-1 rounded-full chip-accent px-2 py-0.5 text-[10px] font-semibold text-accent-ink"
    >
      <span aria-hidden="true">✓</span>
      {level === 'calibrated' ? 'Calibrated' : 'Uncalibrated'}
    </span>
  );
}

function FusionChip({ runId }: { runId: string }) {
  return (
    <span
      title="The trained MedPatch confidence-guided fusion (c-msma), Round 3"
      className="inline-flex items-center gap-1 rounded-full chip-accent px-2 py-0.5 text-[10px] font-semibold text-accent-ink"
    >
      <span aria-hidden="true">✓</span> MedPatch fusion (Round 3, run {runId})
    </span>
  );
}

const HEAD_LABELS: Record<string, string> = {
  high: 'High-confidence head',
  low: 'Low-confidence head',
  late: 'Late fusion',
  combined: 'Mean of high, low and late',
  less_combined: 'Mean of high and late',
};

/** Every head, the run and its caveats, folded away by default. */
function FusionDetails({ model }: { model: FusionModelInfo }) {
  return (
    <details className="w-full text-xs text-ink-secondary">
      <summary className="cursor-pointer select-none font-medium text-ink">Model details</summary>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
        <dt className="text-ink-muted">Run</dt>
        <dd className="tnum">
          {model.runId} · sha256 {model.sha256}
        </dd>
        <dt className="text-ink-muted">Text encoder</dt>
        <dd>{model.bertModelName}</dd>
        <dt className="text-ink-muted">θ</dt>
        <dd className="tnum">{model.theta}</dd>
        <dt className="text-ink-muted">Modalities</dt>
        <dd>{model.modalities.join(', ')}</dd>
      </dl>
      <table className="mt-2 w-full">
        <caption className="sr-only">Probability from each fusion head</caption>
        <tbody>
          {Object.entries(model.heads).map(([head, p]) => (
            <tr key={head} className={cn(head === model.head && 'font-semibold text-ink')}>
              <th scope="row" className="py-0.5 text-left font-normal">
                {HEAD_LABELS[head] ?? head}
                {head === model.head && ' (chosen on validation)'}
              </th>
              <td className="tnum py-0.5 text-right">{percent(p)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {model.internals && <FusionInternalsTable internals={model.internals} />}
      {model.notes.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-4">
          {model.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </details>
  );
}

const VOTE_LABELS: Record<string, string> = {
  ehr: 'EHR vote',
  cxr: 'Chest X-ray vote',
  rr: 'Radiology-report vote',
  dn: 'Discharge-note vote',
};

/** CMSMAFusion's internal outputs, as raw logits and sigmoid, explicitly not probabilities. */
function FusionInternalsTable({ internals }: { internals: FusionInternals }) {
  const rows: [string, LogitProb][] = [
    ...Object.entries(internals.votes)
      .filter((entry): entry is [string, LogitProb] => entry[1] !== null)
      .map(([m, v]): [string, LogitProb] => [VOTE_LABELS[m] ?? m, v]),
    ['Joint head, high-confidence tokens', internals.high],
    ['Joint head, low-confidence tokens', internals.low],
    ['Missingness head', internals.missingness],
  ];
  return (
    <section className="mt-3">
      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-ink">
        Inside the MedPatch fusion (late-fusion votes — not probabilities)
      </h3>
      <p className="mt-1">
        These feed the late fusion and are trained only through its weighted sum, so their scale is
        unconstrained (a sigmoid near 0 % or 100 % is normal). The missingness head sees only which
        inputs are present, not what they say.
      </p>
      <table className="mt-1 w-full">
        <caption className="sr-only">Fusion-internal outputs: logit and sigmoid</caption>
        <thead>
          <tr className="text-ink-muted">
            <th scope="col" className="py-0.5 text-left font-normal">
              Output
            </th>
            <th scope="col" className="py-0.5 text-right font-normal">
              Logit
            </th>
            <th scope="col" className="py-0.5 text-right font-normal">
              Sigmoid
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, v]) => (
            <tr key={label}>
              <th scope="row" className="py-0.5 text-left font-normal">
                {label}
              </th>
              <td className="tnum py-0.5 text-right">{v.logit.toFixed(2)}</td>
              <td className="tnum py-0.5 text-right">{percent(v.probability)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function AverageChip() {
  return (
    <span
      title="The plain mean of the trained models' scores. Not the MedPatch fusion."
      className="inline-flex items-center gap-1 rounded-full chip-accent px-2 py-0.5 text-[10px] font-semibold text-accent-ink"
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
