import { TASK_SPEC, type AnalysisSummary, type PaginationMeta } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';

import { useAuth } from '../app/auth.jsx';
import { api, apiErrorMessage } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { relativeTime, shortDateTime } from '../lib/format.js';
import { Icon } from '../components/icons.jsx';
import { RiskBadge } from '../components/risk.jsx';
import { Button, Card, EmptyState, ErrorState, SkeletonRows } from '../components/ui.jsx';

const PAGE_SIZE = 25;

const MODALITY_LABELS = { EHR: 'EHR', CXR: 'CXR', RR: 'Notes (RR)', DN: 'Discharge (DN)' } as const;

/** A Link styled as the primary button — Button renders a <button>. */
const NEW_ANALYSIS_CLASS =
  'inline-flex h-11 items-center gap-2 rounded-control bg-accent px-4 text-sm font-semibold text-accent-on transition hover:brightness-110';

/**
 * The Analyze tab's landing page: the current user's past runs, with the form
 * one click away at `/analyze/new`.
 *
 * Ad-hoc stays are kept off the ward dashboard, so this is the only way back
 * to a report after leaving it.
 */
export function AnalysesPage() {
  const [params, setParams] = useSearchParams();
  const page = Math.max(1, Number(params.get('page')) || 1);
  // Admins see every user's runs, so the list says whose each one is.
  const showRequester = useAuth().user?.role === 'admin';

  const query = useQuery({
    queryKey: ['analyses', { page }],
    queryFn: async () => {
      const res = await api.get<{ data: AnalysisSummary[]; meta: PaginationMeta }>('/analyses', {
        params: { page, limit: PAGE_SIZE },
      });
      return res.data;
    },
    // A run submitted moments ago may still be scoring.
    refetchInterval: (q) =>
      q.state.data?.data.some((a) => a.status === 'queued' || a.status === 'running')
        ? 3_000
        : false,
  });

  const goTo = (next: number) => {
    const search = new URLSearchParams(params);
    if (next <= 1) search.delete('page');
    else search.set('page', String(next));
    setParams(search);
  };

  const meta = query.data?.meta;

  const columns = [
    'Submitted',
    'Patient',
    ...(showRequester ? ['Requested by'] : []),
    'Task',
    'Result',
    'Inputs sent',
    'Status',
    '',
  ];

  return (
    <div className="flex flex-col gap-[22px]">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px] font-bold tracking-[-0.02em] text-ink">Analyses</h1>
          <p className="mt-1.5 text-ink-secondary">
            Runs you have submitted, newest first. Reports stay available after you leave them.
          </p>
        </div>
        <Link to="/analyze/new" className={NEW_ANALYSIS_CLASS}>
          <Icon name="plus" size={18} strokeWidth={2} />
          New analysis
        </Link>
      </header>

      {query.isLoading ? (
        <Card bodyClassName="p-5">
          <SkeletonRows rows={6} />
        </Card>
      ) : query.error ? (
        <Card>
          <ErrorState message={apiErrorMessage(query.error)} onRetry={() => void query.refetch()} />
        </Card>
      ) : (query.data?.data.length ?? 0) === 0 ? (
        <Card>
          <EmptyState
            icon="◷"
            title="No analyses yet"
            description="Run one to see how the model reasons about a patient."
            action={
              <Link to="/analyze/new" className={NEW_ANALYSIS_CLASS}>
                New analysis
              </Link>
            }
          />
        </Card>
      ) : (
        <Card bodyClassName="p-0">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead>
                <tr className="bg-surface-3 text-left">
                  {columns.map((heading, i) => (
                    <th
                      key={heading || `spacer-${i}`}
                      className="eyebrow whitespace-nowrap px-3.5 py-3 first:pl-5 last:pr-5"
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {query.data?.data.map((analysis) => (
                  <AnalysisRow
                    key={analysis.predictionId}
                    analysis={analysis}
                    showRequester={showRequester}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {meta && meta.totalPages > 1 && (
        <nav
          className="flex items-center justify-between text-xs text-ink-secondary"
          aria-label="Pages"
        >
          <span className="tnum">
            Page {meta.page} of {meta.totalPages} · {meta.total} analyses
          </span>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => goTo(page - 1)}>
              ← Newer
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={page >= meta.totalPages}
              onClick={() => goTo(page + 1)}
            >
              Older →
            </Button>
          </div>
        </nav>
      )}
    </div>
  );
}

function AnalysisRow({
  analysis,
  showRequester,
}: {
  analysis: AnalysisSummary;
  showRequester: boolean;
}) {
  const used = new Set<string>(analysis.inputsUsed.map((m) => m.toUpperCase()));

  return (
    <tr className="border-t border-hairline hover:bg-softer">
      <td className="px-3.5 py-3 pl-5 text-ink-secondary">
        <div>{shortDateTime(analysis.createdAt)}</div>
        <div className="text-xs text-ink-muted">{relativeTime(analysis.createdAt)}</div>
      </td>
      <td className="px-3.5 py-3 font-semibold text-ink">{analysis.pseudoId}</td>
      {showRequester && (
        <td className="px-3.5 py-3 text-ink-secondary">{analysis.requestedBy?.name ?? '—'}</td>
      )}
      <td className="px-3.5 py-3 capitalize text-ink-secondary">{analysis.task}</td>
      <td className="px-3.5 py-3">
        {analysis.status === 'done' ? (
          <RiskBadge probability={analysis.probability} />
        ) : (
          <span className="text-ink-muted">—</span>
        )}
      </td>
      <td className="px-3.5 py-3">
        <div className="flex flex-wrap gap-1">
          {TASK_SPEC[analysis.task].modalities.map((m) => {
            const present = used.has(m);
            return (
              <span
                key={m}
                title={`${MODALITY_LABELS[m]}: ${present ? 'sent to the model' : 'missing'}`}
                className={present ? 'badge-on' : 'badge-off'}
              >
                <span aria-hidden="true">{present ? '✓' : '–'}</span>
                {MODALITY_LABELS[m]}
              </span>
            );
          })}
          {analysis.excludedCount > 0 && (
            <span
              title="Discharge notes are never sent for mortality (outcome leakage)"
              className="badge-off"
            >
              <span aria-hidden="true">⊘</span>
              {analysis.excludedCount} excluded
            </span>
          )}
        </div>
      </td>
      <td className="px-3.5 py-3">
        <span
          title={analysis.error ?? undefined}
          className={cn(
            'inline-flex h-6 items-center rounded-full px-2.5 text-xs font-semibold capitalize',
            analysis.status === 'done' && 'chip-low',
            analysis.status === 'failed' && 'chip-critical',
            (analysis.status === 'queued' || analysis.status === 'running') && 'chip-neutral',
          )}
        >
          {analysis.status}
        </span>
      </td>
      <td className="px-3.5 py-3 pr-5 text-right">
        {analysis.status === 'done' && (
          <Link
            to={`/predictions/${analysis.predictionId}`}
            className="whitespace-nowrap text-xs font-semibold text-accent-ink hover:underline"
          >
            Report →
          </Link>
        )}
      </td>
    </tr>
  );
}
