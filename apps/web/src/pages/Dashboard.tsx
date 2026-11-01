import type { WardDashboard } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { apiErrorMessage, getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent, relativeTime } from '../lib/format.js';

import { ModalityBadges, RiskBadge, Sparkline } from '../components/risk.jsx';
import { Card, EmptyState, ErrorState, Kpi, Segmented, SkeletonRows } from '../components/ui.jsx';

/**
 * P4: ward dashboard — every active stay, highest risk first.
 *
 * This is the landing page, so it has to answer "who needs me now?" in one
 * glance: risk, trend, what evidence exists, and how many alerts are open.
 *
 * Every KPI is derived from the rows this page already loads. None of them
 * costs a request — if a number needed one, the card would not be here.
 */
export function DashboardPage() {
  const [ward, setWard] = useState<string>('');
  const [minRisk, setMinRisk] = useState<number>(0);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboard', { ward }],
    queryFn: () => getData<WardDashboard>('/dashboard/ward', ward ? { ward } : undefined),
    refetchInterval: 20_000,
  });

  const all = data?.rows ?? [];
  const rows = all.filter((r) => (r.mortality ?? 0) >= minRisk);

  const atRisk = all.filter((r) => (r.mortality ?? 0) >= 0.5 || (r.pneumonia ?? 0) >= 0.5).length;
  const openAlerts = all.reduce((sum, r) => sum + r.openAlerts, 0);
  const complete = all.filter(
    (r) => r.availability.ehr && r.availability.cxr && r.availability.notes,
  ).length;

  const wardOptions = [
    { value: '', label: 'All wards' },
    ...(data?.wards ?? []).map((w) => ({ value: w, label: w })),
  ];

  return (
    <div className="flex flex-col gap-[22px]">
      <header>
        <h1 className="text-[28px] font-bold tracking-[-0.02em] text-ink">Ward overview</h1>
        <p className="mt-1.5 text-ink-secondary">
          Every active stay, highest risk first. Scores refresh automatically.
        </p>
      </header>

      {!isLoading && !error && all.length > 0 && (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-4">
          <Kpi
            label="Patients monitored"
            value={all.length}
            hint={`${data?.wards.length ?? 0} wards`}
          />
          <Kpi label="At risk" value={atRisk} hint="Either task at 50% or above" />
          <Kpi label="Open alerts" value={openAlerts} hint="Across all shown stays" />
          <Kpi
            label="Complete inputs"
            value={`${complete} of ${all.length}`}
            hint="All three modalities present"
          />
        </div>
      )}

      <Card bodyClassName="p-0">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline px-5 py-4">
          <Segmented
            value={ward}
            onChange={setWard}
            options={wardOptions}
            ariaLabel="Filter by ward"
          />
          <select
            value={minRisk}
            onChange={(e) => setMinRisk(Number(e.target.value))}
            aria-label="Filter by minimum risk"
            className="field-input"
          >
            <option value={0}>Any risk</option>
            <option value={0.25}>Moderate and above</option>
            <option value={0.5}>High and above</option>
            <option value={0.7}>Critical only</option>
          </select>
        </div>

        {isLoading ? (
          <div className="p-5">
            <SkeletonRows rows={8} />
          </div>
        ) : error ? (
          <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon="◎"
            title={all.length ? 'No stays match these filters' : 'No active stays'}
          />
        ) : (
          <div className="w-full min-w-0 overflow-x-auto">
            <table className="w-full min-w-[880px] text-sm">
              <thead>
                <tr className="table-head">
                  {[
                    'Patient',
                    'Ward / bed',
                    'Mortality',
                    'Trend',
                    'Pneumonia',
                    'Modalities',
                    'Alerts',
                    'Scored',
                  ].map((heading) => (
                    <th key={heading}>{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.stayId} className="border-t border-hairline hover:bg-softer">
                    <td className="px-3.5 py-3 pl-5">
                      <Link
                        to={`/stays/${row.stayId}`}
                        className="font-semibold text-ink hover:underline"
                      >
                        {row.pseudoId}
                      </Link>
                      <div className="text-xs text-ink-secondary">
                        {row.age}y · {row.sex}
                      </div>
                    </td>
                    <td className="px-3.5 py-3 text-ink-secondary">
                      {row.ward}
                      <span className="text-ink-muted"> · {row.bedLabel}</span>
                    </td>
                    <td className="px-3.5 py-3">
                      <RiskBadge probability={row.mortality} />
                    </td>
                    <td className="px-3.5 py-3">
                      <Sparkline
                        values={row.sparkline}
                        ariaLabel={`Mortality trend for ${row.pseudoId}, latest ${percent(row.mortality)}`}
                      />
                    </td>
                    <td
                      className={cn(
                        'tnum px-3.5 py-3',
                        row.pneumonia === null ? 'text-ink-muted' : 'font-medium text-ink',
                      )}
                    >
                      {percent(row.pneumonia)}
                    </td>
                    <td className="px-3.5 py-3">
                      <ModalityBadges availability={row.availability} />
                    </td>
                    <td className="px-3.5 py-3">
                      {row.openAlerts > 0 ? (
                        <Link
                          to={`/alerts?stayId=${row.stayId}`}
                          className="chip-critical tnum inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold"
                        >
                          <span aria-hidden="true">■</span>
                          {row.openAlerts}
                        </Link>
                      ) : (
                        <span className="text-xs text-ink-muted">—</span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-3.5 py-3 pr-5 text-xs text-ink-secondary">
                      {relativeTime(row.lastScoredAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
