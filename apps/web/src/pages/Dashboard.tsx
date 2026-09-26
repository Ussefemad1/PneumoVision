import type { WardDashboard } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { apiErrorMessage, getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent, relativeTime } from '../lib/format.js';
import { riskBand } from '../lib/risk.js';
import { ModalityBadges, RiskBadge, Sparkline } from '../components/risk.jsx';
import { Card, EmptyState, ErrorState, SkeletonRows } from '../components/ui.jsx';

/**
 * P4: ward dashboard — every active stay, highest risk first.
 *
 * This is the landing page, so it has to answer "who needs me now?" in one
 * glance: risk, trend, what evidence exists, and how many alerts are open.
 */
export function DashboardPage() {
  const [ward, setWard] = useState<string>('');
  const [minRisk, setMinRisk] = useState<number>(0);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboard', { ward }],
    queryFn: () => getData<WardDashboard>('/dashboard/ward', ward ? { ward } : undefined),
    refetchInterval: 20_000,
  });

  const rows = (data?.rows ?? []).filter((r) => (r.mortality ?? 0) >= minRisk);

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-ink">Ward dashboard</h1>
          <p className="text-sm text-ink-muted">Active stays ranked by 48-hour mortality risk.</p>
        </div>

        {/* Filters sit in one row above the content. */}
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={ward}
            onChange={(e) => setWard(e.target.value)}
            aria-label="Filter by ward"
            className="rounded-md border border-hairline bg-surface-1 px-2.5 py-1.5 text-sm text-ink"
          >
            <option value="">All wards</option>
            {data?.wards.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>

          <select
            value={minRisk}
            onChange={(e) => setMinRisk(Number(e.target.value))}
            aria-label="Filter by minimum risk"
            className="rounded-md border border-hairline bg-surface-1 px-2.5 py-1.5 text-sm text-ink"
          >
            <option value={0}>Any risk</option>
            <option value={0.25}>Moderate and above</option>
            <option value={0.5}>High and above</option>
            <option value={0.7}>Critical only</option>
          </select>
        </div>
      </header>

      <Card bodyClassName="p-0">
        {isLoading ? (
          <div className="p-4">
            <SkeletonRows rows={8} />
          </div>
        ) : error ? (
          <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon="◎"
            title={data?.rows.length ? 'No stays match these filters' : 'No active stays'}
            description={
              data?.rows.length
                ? 'Widen the ward or risk filter to see more patients.'
                : 'Seeded demo data should appear here. Check the API logs if this stays empty.'
            }
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead>
                <tr className="border-b border-hairline bg-surface-3/60 text-left text-[11px] uppercase tracking-wide text-ink-muted">
                  <th className="px-4 py-2 font-medium">Patient</th>
                  <th className="px-4 py-2 font-medium">Ward / bed</th>
                  <th className="px-4 py-2 font-medium">Mortality</th>
                  <th className="px-4 py-2 font-medium">Trend</th>
                  <th className="px-4 py-2 font-medium">Pneumonia</th>
                  <th className="px-4 py-2 font-medium">Modalities</th>
                  <th className="px-4 py-2 font-medium">Alerts</th>
                  <th className="px-4 py-2 font-medium">Scored</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const band = riskBand(row.mortality);
                  return (
                    <tr
                      key={row.stayId}
                      className="border-b border-hairline last:border-0 hover:bg-surface-3/50"
                    >
                      <td className="px-4 py-2.5">
                        <Link
                          to={`/stays/${row.stayId}`}
                          className="font-medium text-ink hover:underline"
                        >
                          {row.pseudoId}
                        </Link>
                        <div className="text-[11px] text-ink-muted">
                          {row.age}y · {row.sex}
                        </div>
                      </td>
                      <td className="px-4 py-2.5 text-ink-secondary">
                        {row.ward}
                        <span className="text-ink-muted"> · {row.bedLabel}</span>
                      </td>
                      <td className="px-4 py-2.5">
                        <RiskBadge probability={row.mortality} />
                      </td>
                      <td className="px-4 py-2.5">
                        <Sparkline
                          values={row.sparkline}
                          ariaLabel={`Mortality trend for ${row.pseudoId}, latest ${percent(row.mortality)}`}
                        />
                      </td>
                      <td
                        className={cn(
                          'tnum px-4 py-2.5',
                          row.pneumonia === null && 'text-ink-muted',
                        )}
                      >
                        {percent(row.pneumonia)}
                      </td>
                      <td className="px-4 py-2.5">
                        <ModalityBadges availability={row.availability} />
                      </td>
                      <td className="px-4 py-2.5">
                        {row.openAlerts > 0 ? (
                          <Link
                            to={`/alerts?stayId=${row.stayId}`}
                            className="tnum inline-flex items-center gap-1 rounded-full bg-status-critical/15 px-2 py-0.5 text-xs font-medium text-status-critical"
                          >
                            <span aria-hidden="true">■</span>
                            {row.openAlerts}
                          </Link>
                        ) : (
                          <span className="text-xs text-ink-muted">—</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-ink-muted">
                        {relativeTime(row.lastScoredAt)}
                      </td>
                      <td className="sr-only">{band.label}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
