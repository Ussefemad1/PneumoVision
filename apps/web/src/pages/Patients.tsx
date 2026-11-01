import type { WardDashboard } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { apiErrorMessage, getData } from '../lib/api.js';
import { percent, relativeTime } from '../lib/format.js';
import { Icon } from '../components/icons.jsx';
import { ModalityBadges, RiskBadge } from '../components/risk.jsx';
import { Card, EmptyState, ErrorState, Segmented, SkeletonRows } from '../components/ui.jsx';

/**
 * Every stay, as a roster rather than a triage board.
 *
 * Deliberately the same query key and fetcher as the ward dashboard
 * (`['dashboard', { ward }]` → `/dashboard/ward`), so the two share a cache
 * entry and this page costs no extra request and no new endpoint. The
 * dashboard sorts by risk to answer "who needs me now?"; this one sorts by
 * patient id and adds search, to answer "where is this patient?".
 */
export function PatientsPage() {
  const [ward, setWard] = useState<string>('');
  const [query, setQuery] = useState('');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboard', { ward }],
    queryFn: () => getData<WardDashboard>('/dashboard/ward', ward ? { ward } : undefined),
    refetchInterval: 20_000,
  });

  const all = data?.rows ?? [];
  const needle = query.trim().toLowerCase();
  const rows = all
    .filter((r) =>
      needle === '' ? true : `${r.pseudoId} ${r.ward} ${r.bedLabel}`.toLowerCase().includes(needle),
    )
    .slice()
    .sort((a, b) => a.pseudoId.localeCompare(b.pseudoId));

  const wardOptions = [
    { value: '', label: 'All wards' },
    ...(data?.wards ?? []).map((w) => ({ value: w, label: w })),
  ];

  return (
    <div className="flex flex-col gap-[22px]">
      <header>
        <h1 className="text-[28px] font-bold tracking-[-0.02em] text-ink">Patients</h1>
        <p className="mt-1.5 text-ink-secondary">
          {all.length} {all.length === 1 ? 'patient' : 'patients'} with an active ICU stay.
        </p>
      </header>

      <Card bodyClassName="p-0">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline px-5 py-4">
          <Segmented
            value={ward}
            onChange={setWard}
            options={wardOptions}
            ariaLabel="Filter by ward"
          />
          <label className="field-input flex min-w-[240px] items-center gap-2 text-sm">
            <Icon name="search" size={16} className="flex-none text-ink-secondary" />
            <input
              type="search"
              aria-label="Search patients"
              placeholder="Search ID, ward or bed"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full border-0 bg-transparent p-0 text-ink outline-none"
            />
          </label>
        </div>

        {isLoading ? (
          <div className="p-5">
            <SkeletonRows rows={8} />
          </div>
        ) : error ? (
          <ErrorState message={apiErrorMessage(error)} onRetry={() => void refetch()} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon="○"
            title={all.length ? 'No patients match this search' : 'No active stays'}
            description={all.length ? 'Try a different id, ward or bed.' : undefined}
          />
        ) : (
          <div className="w-full min-w-0 overflow-x-auto">
            <table className="w-full min-w-[840px] text-sm">
              <thead>
                <tr className="table-head">
                  {[
                    'Patient',
                    'Age · Sex',
                    'Current stay',
                    'Mortality',
                    'Pneumonia',
                    'Sources',
                    'Last scored',
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
                    </td>
                    <td className="px-3.5 py-3 text-ink-secondary">
                      {row.age} · {row.sex}
                    </td>
                    <td className="px-3.5 py-3 text-ink-secondary">
                      {row.ward}
                      <span className="text-ink-muted"> · bed {row.bedLabel}</span>
                    </td>
                    <td className="px-3.5 py-3">
                      <RiskBadge probability={row.mortality} />
                    </td>
                    <td className="tnum px-3.5 py-3 text-ink">{percent(row.pneumonia)}</td>
                    <td className="px-3.5 py-3">
                      <ModalityBadges availability={row.availability} />
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
