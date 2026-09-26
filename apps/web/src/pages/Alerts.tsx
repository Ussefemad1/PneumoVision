import type { AlertWithContext } from '@pneumovision/shared';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { apiErrorMessage, getData, patchData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { percent, relativeTime } from '../lib/format.js';
import { SEVERITY_INFO } from '../lib/risk.js';
import { getSocket } from '../lib/socket.js';
import { Button, Card, EmptyState, ErrorState, SkeletonRows } from '../components/ui.jsx';

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
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-ink">Alerts</h1>
          <p className="text-sm text-ink-muted">
            Raised when risk crosses a threshold or climbs sharply within a window.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            value={status}
            onChange={(e) => setFilter('status', e.target.value)}
            aria-label="Filter by status"
            className="rounded-md border border-hairline bg-surface-1 px-2.5 py-1.5 text-sm text-ink"
          >
            <option value="">Any status</option>
            <option value="open">Open</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="resolved">Resolved</option>
          </select>
          <select
            value={severity}
            onChange={(e) => setFilter('severity', e.target.value)}
            aria-label="Filter by severity"
            className="rounded-md border border-hairline bg-surface-1 px-2.5 py-1.5 text-sm text-ink"
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
      </header>

      {query.isLoading ? (
        <SkeletonRows rows={6} />
      ) : query.error ? (
        <ErrorState message={apiErrorMessage(query.error)} onRetry={() => void query.refetch()} />
      ) : (query.data?.length ?? 0) === 0 ? (
        <Card>
          <EmptyState
            icon="✓"
            title="Nothing to action"
            description={
              status === 'open'
                ? 'No open alerts. Deteriorating patients will appear here as soon as a prediction crosses a rule.'
                : 'No alerts match these filters.'
            }
          />
        </Card>
      ) : (
        <ul className="space-y-2">
          {query.data?.map((alert) => {
            const info = SEVERITY_INFO[alert.severity];
            return (
              <li key={alert.id}>
                <Card bodyClassName="p-3">
                  <div className="flex flex-wrap items-center gap-3">
                    <span
                      className={cn(
                        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium',
                        info.chipClass,
                      )}
                    >
                      <span aria-hidden="true">{info.icon}</span>
                      {info.label}
                    </span>

                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-ink">{alert.rule}</div>
                      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-ink-muted">
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
                          'rounded-full px-2 py-0.5 text-[11px] capitalize',
                          alert.status === 'open' && 'bg-surface-3 text-ink-secondary',
                          alert.status === 'acknowledged' &&
                            'bg-status-warning/15 text-[#8a5d00] dark:text-status-warning',
                          alert.status === 'resolved' && 'bg-status-good/15 text-status-good',
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
                          className="text-xs font-medium text-[var(--series-mortality)] hover:underline"
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
