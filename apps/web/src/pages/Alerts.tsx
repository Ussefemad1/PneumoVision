import type { AlertWithContext } from '@pneumovision/shared';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { apiErrorMessage, getData, patchData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent, relativeTime } from '../lib/format.js';
import { SEVERITY_INFO } from '../lib/risk.js';
import { getSocket } from '../lib/socket.js';
import {
  Button,
  Card,
  EmptyState,
  ErrorState,
  Segmented,
  SkeletonRows,
} from '../components/ui.jsx';

const STATUS_FILTERS = [
  { value: 'open', label: 'Open' },
  { value: 'acknowledged', label: 'Acknowledged' },
  { value: 'resolved', label: 'Resolved' },
  { value: '', label: 'Any' },
] as const;

/** P3: alerts centre — filter, acknowledge, resolve, live. */
export function AlertsPage() {
  const [params, setParams] = useSearchParams();
  const status = params.get('status') ?? 'open';
  const severity = params.get('severity') ?? '';
  const stayId = params.get('stayId') ?? '';
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ['alerts', { status, severity, stayId }],
    queryFn: () =>
      getData<AlertWithContext[]>('/alerts', {
        ...(status ? { status } : {}),
        ...(severity ? { severity } : {}),
        ...(stayId ? { stayId } : {}),
      }),
  });

  // New and updated alerts arrive over the socket; refresh rather than
  // patching the list by hand, so filters stay authoritative.
  useEffect(() => {
    const socket = getSocket();
    const refresh = () => void queryClient.invalidateQueries({ queryKey: ['alerts'] });
    socket.on('alert:new', refresh);
    socket.on('alert:updated', refresh);
    return () => {
      socket.off('alert:new', refresh);
      socket.off('alert:updated', refresh);
    };
  }, [queryClient]);

  /**
   * Acknowledgement is the one place we update optimistically: it is a
   * frequent, low-stakes, easily reversible action, and waiting on a round
   * trip makes clearing a list feel broken.
   */
  const acknowledge = useMutation({
    mutationFn: (id: string) => patchData(`/alerts/${id}/acknowledge`),
    onMutate: async (id) => {
      const key = ['alerts', { status, severity, stayId }];
      await queryClient.cancelQueries({ queryKey: key });
      const previous = queryClient.getQueryData<AlertWithContext[]>(key);
      queryClient.setQueryData<AlertWithContext[]>(key, (old) =>
        old?.map((a) => (a.id === id ? { ...a, status: 'acknowledged' } : a)),
      );
      return { previous, key };
    },
    onError: (_err, _id, context) => {
      if (context) queryClient.setQueryData(context.key, context.previous);
    },
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ['alerts'] }),
  });

  const resolve = useMutation({
    mutationFn: (id: string) => patchData(`/alerts/${id}/resolve`),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['alerts'] }),
  });

  const setFilter = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  };

  return (
    <div className="flex flex-col gap-[22px]">
      <header>
        <h1 className="text-[28px] font-bold tracking-[-0.02em] text-ink">Alerts</h1>
        <p className="mt-1.5 text-ink-secondary">
          Threshold crossings from the alert engine, live. Acknowledge to claim one.
        </p>
      </header>

      <div className="flex flex-wrap items-center justify-between gap-3">
        {/*
         * Segmented rather than the underline tabs in the design: these are
         * filters over one list, not tabpanel switchers, and the response is
         * already filtered server-side so per-status counts are not available
         * without extra requests.
         */}
        <Segmented
          value={status}
          onChange={(value) => setFilter('status', value)}
          options={STATUS_FILTERS}
          ariaLabel="Filter by status"
        />
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={severity}
            onChange={(e) => setFilter('severity', e.target.value)}
            aria-label="Filter by severity"
            className="field-input"
          >
            <option value="">Any severity</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
          </select>
          {stayId && (
            <Button size="sm" variant="ghost" onClick={() => setFilter('stayId', '')}>
              Clear patient filter
            </Button>
          )}
        </div>
      </div>

      {query.isLoading ? (
        <Card bodyClassName="p-5">
          <SkeletonRows rows={6} />
        </Card>
      ) : query.error ? (
        <Card>
          <ErrorState message={apiErrorMessage(query.error)} onRetry={() => void query.refetch()} />
        </Card>
      ) : (query.data?.length ?? 0) === 0 ? (
        <Card>
          <EmptyState
            icon="✓"
            title="Nothing to action"
            description="No alerts match these filters."
          />
        </Card>
      ) : (
        <ul className="flex flex-col gap-3">
          {query.data?.map((alert) => {
            const info = SEVERITY_INFO[alert.severity];
            return (
              <li key={alert.id}>
                <Card bodyClassName="p-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <span
                      className={cn(
                        'inline-flex h-7 items-center gap-1.5 rounded-full px-3 text-xs font-semibold',
                        info.chipClass,
                      )}
                    >
                      <span aria-hidden="true">{info.icon}</span>
                      {info.label}
                    </span>

                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-ink">{alert.rule}</div>
                      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-ink-secondary">
                        <Link
                          to={`/stays/${alert.stay.id}`}
                          className="font-medium text-ink-secondary hover:underline"
                        >
                          {alert.stay.pseudoId}
                        </Link>
                        <span>
                          {alert.stay.ward} · {alert.stay.bedLabel}
                        </span>
                        <span className="tnum">value {percent(alert.value)}</span>
                        <span>{relativeTime(alert.createdAt)}</span>
                        {alert.acknowledgedAt && (
                          <span>acknowledged {relativeTime(alert.acknowledgedAt)}</span>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          'inline-flex h-6 items-center rounded-full px-2.5 text-xs font-semibold capitalize',
                          alert.status === 'open' && 'chip-neutral',
                          alert.status === 'acknowledged' && 'chip-moderate',
                          alert.status === 'resolved' && 'chip-low',
                        )}
                      >
                        {alert.status}
                      </span>

                      {alert.status === 'open' && (
                        <Button size="sm" onClick={() => acknowledge.mutate(alert.id)}>
                          Acknowledge
                        </Button>
                      )}
                      {alert.status !== 'resolved' && (
                        <Button size="sm" variant="ghost" onClick={() => resolve.mutate(alert.id)}>
                          Resolve
                        </Button>
                      )}
                      {alert.predictionId && (
                        <Link
                          to={`/predictions/${alert.predictionId}`}
                          className="whitespace-nowrap text-xs font-semibold text-accent-ink hover:underline"
                        >
                          Report →
                        </Link>
                      )}
                    </div>
                  </div>
                </Card>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
