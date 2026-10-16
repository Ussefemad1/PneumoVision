import { TASK_SPEC, type AnalysisSummary, type PaginationMeta } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';

import { useAuth } from '../app/auth.jsx';
import { api, apiErrorMessage } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { relativeTime, shortDateTime } from '../lib/format.js';
import { RiskBadge } from '../components/risk.jsx';
import { Button, Card, EmptyState, ErrorState, SkeletonRows } from '../components/ui.jsx';

const PAGE_SIZE = 25;

const MODALITY_LABELS = { EHR: 'EHR', CXR: 'CXR', RR: 'Notes (RR)', DN: 'Discharge (DN)' } as const;

const NEW_ANALYSIS_CLASS =
  'inline-flex items-center gap-1.5 rounded-md bg-[var(--series-mortality)] px-3 py-1.5 text-sm font-medium text-white hover:opacity-90';

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

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-ink">Analyses</h1>
        </div>
        <Link to="/analyze/new" className={NEW_ANALYSIS_CLASS}>
          <span aria-hidden="true">+</span> New analysis
        </Link>
      </header>

      {query.isLoading ? (
        <SkeletonRows rows={6} />
      ) : query.error ? (
        <ErrorState message={apiErrorMessage(query.error)} onRetry={() => void query.refetch()} />
      ) : (query.data?.data.length ?? 0) === 0 ? (
        <Card>
          <EmptyState
            icon="◷"
            title="No analyses yet"
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
            <table className="w-full min-w-[820px] text-sm">
              <thead>
                <tr className="border-b border-hairline bg-surface-3/60 text-left text-[11px] uppercase tracking-wide text-ink-muted">
                  <th className="px-4 py-2 font-medium">Submitted</th>
                  <th className="px-4 py-2 font-medium">Patient</th>
                  {showRequester && <th className="px-4 py-2 font-medium">Requested by</th>}
                  <th className="px-4 py-2 font-medium">Task</th>
                  <th className="px-4 py-2 font-medium">Result</th>
                  <th className="px-4 py-2 font-medium">Inputs sent</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                  <th className="px-4 py-2 font-medium" />
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
        <nav className="flex items-center justify-between text-xs text-ink-muted" aria-label="Pages">
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
    <tr className="border-b border-hairline last:border-0 hover:bg-surface-3/50">
      <td className="px-4 py-2 text-ink-secondary">
        <div>{shortDateTime(analysis.createdAt)}</div>
        <div className="text-[11px] text-ink-muted">{relativeTime(analysis.createdAt)}</div>
      </td>
      <td className="px-4 py-2 font-medium text-ink">{analysis.pseudoId}</td>
      {showRequester && (
        <td className="px-4 py-2 text-ink-secondary">{analysis.requestedBy?.name ?? '—'}</td>
      )}
      <td className="px-4 py-2 capitalize text-ink-secondary">{analysis.task}</td>
      <td className="px-4 py-2">
        {analysis.status === 'done' ? (
          <RiskBadge probability={analysis.probability} />
        ) : (
          <span className="text-ink-muted">—</span>
        )}
      </td>
      <td className="px-4 py-2">
        <div className="flex flex-wrap gap-1">
          {TASK_SPEC[analysis.task].modalities.map((m) => {
            const present = used.has(m);
            return (
              <span
                key={m}
                title={`${MODALITY_LABELS[m]}: ${present ? 'sent to the model' : 'missing'}`}
                className={cn(
                  'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium',
                  present
                    ? 'bg-surface-3 text-ink-secondary ring-1 ring-hairline'
                    : 'text-ink-muted ring-1 ring-dashed ring-hairline',
                )}
              >
                <span aria-hidden="true">{present ? '✓' : '–'}</span>
                {MODALITY_LABELS[m]}
              </span>
            );
          })}
          {analysis.excludedCount > 0 && (
            <span
              title="Discharge notes are never sent for mortality (outcome leakage)"
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-muted ring-1 ring-dashed ring-hairline"
            >
              <span aria-hidden="true">⊘</span>
              {analysis.excludedCount} excluded
            </span>
          )}
        </div>
      </td>
      <td className="px-4 py-2">
        <span
          title={analysis.error ?? undefined}
          className={cn(
            'rounded-full px-2 py-0.5 text-[11px] font-medium',
            analysis.status === 'done' && 'bg-status-good/15 text-status-good',
            analysis.status === 'failed' && 'bg-status-critical/15 text-status-critical',
            (analysis.status === 'queued' || analysis.status === 'running') &&
              'bg-surface-3 text-ink-secondary',
          )}
        >
          {analysis.status}
        </span>
      </td>
      <td className="px-4 py-2 text-right">
        {analysis.status === 'done' && (
          <Link
            to={`/predictions/${analysis.predictionId}`}
            className="text-xs font-medium text-[var(--series-mortality)] hover:underline"
          >
            Report →
          </Link>
        )}
      </td>
    </tr>
  );
}
