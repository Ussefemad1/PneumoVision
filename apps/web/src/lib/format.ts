/**
 * Formatting helpers.
 *
 * House rule: every probability is shown as a percentage with exactly one
 * decimal, everywhere, so two numbers on screen are always comparable.
 */

export function percent(value: number | null | undefined, dash = '—'): string {
  if (value === null || value === undefined || Number.isNaN(value)) return dash;
  return `${(value * 100).toFixed(1)}%`;
}

/** Whole-percent form, for axis ticks and other tight spaces. */
export function percentTerse(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function shortDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function timeOnly(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

export function relativeTime(iso: string | null): string {
  if (!iso) return 'never';
  const deltaMs = Date.now() - new Date(iso).getTime();
  const minutes = Math.round(deltaMs / 60_000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function durationMs(ms: number | null): string {
  if (ms === null) return '—';
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
}
