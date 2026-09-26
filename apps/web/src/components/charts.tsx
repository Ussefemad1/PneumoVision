import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { cn } from '../lib/cn.js';
import { percent, percentTerse, shortDateTime } from '../lib/format.js';

/**
 * Chart layer.
 *
 * Shared rules: one y-axis only, recessive grid and axes, a legend whenever
 * two series are on screen, tabular figures, and a hover layer on everything
 * that plots. Series colours come from the validated categorical slots; the
 * status ramp is reserved for risk bands and never appears as a series.
 */

const AXIS_STYLE = { fontSize: 11, fill: 'var(--text-muted)' } as const;

function TooltipShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-hairline bg-surface-1 px-2.5 py-2 text-xs shadow-lg">
      {children}
    </div>
  );
}

export interface TrajectoryPoint {
  cutoffTime: string;
  probability: number;
}

export interface TrajectoryEvent {
  at: string;
  kind: 'cxr' | 'note';
  label: string;
}

/**
 * P2: risk trajectory.
 *
 * Both tasks share one 0–100% axis, so they are directly comparable — a dual
 * axis here would invite exactly the false comparison it looks like it
 * enables. Event markers show when new evidence arrived.
 */
export function RiskTrajectoryChart({
  mortality,
  pneumonia,
  events = [],
  threshold,
  height = 240,
}: {
  mortality: TrajectoryPoint[];
  pneumonia?: TrajectoryPoint[] | undefined;
  events?: TrajectoryEvent[] | undefined;
  threshold?: number | undefined;
  height?: number | undefined;
}) {
  const data = useMemo(() => {
    const byTime = new Map<number, { t: number; mortality?: number; pneumonia?: number }>();
    for (const p of mortality) {
      const t = new Date(p.cutoffTime).getTime();
      byTime.set(t, { ...(byTime.get(t) ?? { t }), t, mortality: p.probability });
    }
    for (const p of pneumonia ?? []) {
      const t = new Date(p.cutoffTime).getTime();
      byTime.set(t, { ...(byTime.get(t) ?? { t }), t, pneumonia: p.probability });
    }
    return [...byTime.values()].sort((a, b) => a.t - b.t);
  }, [mortality, pneumonia]);

  const hasPneumonia = (pneumonia?.length ?? 0) > 0;

  return (
    <div>
      {/* Legend is always present with two series; identity is never colour-alone. */}
      <div className="mb-2 flex flex-wrap items-center gap-4 text-xs text-ink-secondary">
        <LegendSwatch color="var(--series-mortality)" label="Mortality" />
        {hasPneumonia && <LegendSwatch color="var(--series-pneumonia)" label="Pneumonia" />}
        {threshold !== undefined && (
          <span className="flex items-center gap-1.5 text-ink-muted">
            <svg width="16" height="8" aria-hidden="true">
              <line
                x1="0"
                y1="4"
                x2="16"
                y2="4"
                stroke="var(--axis)"
                strokeWidth="2"
                strokeDasharray="4 3"
              />
            </svg>
            Alert threshold {percentTerse(threshold)}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(t: number) => shortDateTime(new Date(t).toISOString())}
            tick={AXIS_STYLE}
            stroke="var(--axis)"
            minTickGap={48}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={percentTerse}
            tick={AXIS_STYLE}
            stroke="var(--axis)"
            width={52}
          />
          {threshold !== undefined && (
            <ReferenceLine y={threshold} stroke="var(--axis)" strokeDasharray="4 3" />
          )}
          {events.map((event, i) => (
            <ReferenceLine
              key={`${event.at}-${i}`}
              x={new Date(event.at).getTime()}
              stroke="var(--grid)"
              strokeWidth={1}
              label={{
                value: event.kind === 'cxr' ? '▣' : '▤',
                position: 'top',
                fontSize: 10,
                fill: 'var(--text-muted)',
              }}
            />
          ))}
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              return (
                <TooltipShell>
                  <div className="mb-1 font-medium text-ink">
                    {shortDateTime(new Date(Number(label)).toISOString())}
                  </div>
                  {payload.map((entry) => (
                    <div key={entry.name} className="flex items-center gap-2">
                      <span
                        aria-hidden="true"
                        className="h-2 w-2 rounded-full"
                        style={{ background: entry.color }}
                      />
                      <span className="text-ink-secondary">{entry.name}</span>
                      <span className="tnum ml-auto font-medium text-ink">
                        {percent(entry.value as number)}
                      </span>
                    </div>
                  ))}
                </TooltipShell>
              );
            }}
          />
          <Line
            type="monotone"
            dataKey="mortality"
            name="Mortality"
            stroke="var(--series-mortality)"
            strokeWidth={2}
            dot={{ r: 3, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls
            isAnimationActive={false}
          />
          {hasPneumonia && (
            <Line
              type="monotone"
              dataKey="pneumonia"
              name="Pneumonia"
              stroke="var(--series-pneumonia)"
              strokeWidth={2}
              dot={{ r: 3, strokeWidth: 0 }}
              activeDot={{ r: 5 }}
              connectNulls
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span aria-hidden="true" className="h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
      {label}
    </span>
  );
}

/**
 * F6: modality contribution.
 *
 * A horizontal bar per branch, sorted by weight. The branch name is on the
 * axis, so colour carries no identity — it only separates the two *kinds* of
 * branch (fusion-stage vs per-modality), which the legend names.
 */
export function AlphaContributionChart({
  alphas,
}: {
  alphas: Record<string, number | null | undefined>;
}) {
  const LABELS: Record<string, string> = {
    high: 'High-confidence joint',
    low: 'Low-confidence joint',
    miss: 'Missingness',
    ehr: 'EHR (vitals)',
    cxr: 'Chest X-ray',
    rr: 'Radiology reports',
    dn: 'Discharge notes',
  };
  const FUSION_BRANCHES = new Set(['high', 'low', 'miss']);

  const rows = Object.entries(alphas)
    .filter(([, v]) => typeof v === 'number')
    .map(([key, v]) => ({ key, label: LABELS[key] ?? key, value: v as number }))
    .sort((a, b) => b.value - a.value);

  if (rows.length === 0) {
    return <p className="text-xs text-ink-muted">No fusion weights were returned.</p>;
  }

  const max = Math.max(...rows.map((r) => r.value));

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-4 text-xs text-ink-secondary">
        <LegendSwatch color="var(--series-mortality)" label="Fusion stage" />
        <LegendSwatch color="var(--series-pneumonia)" label="Single modality" />
      </div>

      <ul className="space-y-2">
        {rows.map((row) => (
          <li
            key={row.key}
            className="grid grid-cols-[minmax(0,9.5rem)_1fr_auto] items-center gap-3"
          >
            <span className="truncate text-xs text-ink-secondary">{row.label}</span>
            <span className="h-2.5 w-full overflow-hidden rounded-full bg-surface-3">
              <span
                className="block h-full rounded-full"
                style={{
                  width: `${Math.max(2, (row.value / max) * 100)}%`,
                  background: FUSION_BRANCHES.has(row.key)
                    ? 'var(--series-mortality)'
                    : 'var(--series-pneumonia)',
                }}
              />
            </span>
            {/* Direct label on every bar: there are few enough that a value
                axis would be wasted space. */}
            <span className="tnum w-12 text-right text-xs font-medium text-ink">
              {percent(row.value)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * F2 (CXR): ViT patch-confidence heatmap overlaid on the radiograph.
 *
 * Sequential single-hue ramp — magnitude, not identity. Opacity is
 * adjustable because an overlay that cannot be dimmed hides the very image a
 * radiologist needs to read.
 */
export function ConfidenceHeatmapOverlay({
  grid,
  opacity,
  className,
}: {
  grid: number[][];
  opacity: number;
  className?: string | undefined;
}) {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return null;

  return (
    <svg
      viewBox={`0 0 ${cols} ${rows}`}
      preserveAspectRatio="none"
      className={cn('pointer-events-none absolute inset-0 h-full w-full', className)}
      style={{ opacity }}
      aria-hidden="true"
    >
      {grid.map((row, y) =>
        row.map((value, x) => (
          <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill={sequentialBlue(value)} />
        )),
      )}
    </svg>
  );
}

/** Maps 0–1 onto the validated sequential blue ramp. */
export function sequentialBlue(value: number): string {
  const steps = [
    'var(--seq-100)',
    'var(--seq-250)',
    'var(--seq-400)',
    'var(--seq-550)',
    'var(--seq-700)',
  ];
  const index = Math.min(steps.length - 1, Math.max(0, Math.floor(value * steps.length)));
  return steps[index]!;
}

/** Shared legend for anything drawn on the sequential confidence ramp. */
export function ConfidenceScaleLegend({ theta }: { theta?: number | undefined }) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-muted">
      <span>Low confidence</span>
      <span className="flex h-2.5 w-24 overflow-hidden rounded-full">
        {[0.1, 0.3, 0.5, 0.7, 0.9].map((v) => (
          <span key={v} className="h-full flex-1" style={{ background: sequentialBlue(v) }} />
        ))}
      </span>
      <span>High</span>
      {theta !== undefined && <span className="ml-1">· θ = {theta.toFixed(2)}</span>}
    </div>
  );
}

/**
 * F2 (EHR): per-hour confidence band under the vitals timeline.
 *
 * Bars below θ are drawn hollow as well as pale, so the θ split is visible
 * without relying on the ramp alone.
 */
export function EhrConfidenceBand({
  values,
  theta,
  height = 44,
}: {
  values: number[];
  theta: number;
  height?: number | undefined;
}) {
  const [hover, setHover] = useState<number | null>(null);

  return (
    <div className="relative">
      <div className="flex items-end gap-px" style={{ height }}>
        {values.map((value, hour) => {
          const above = value >= theta;
          return (
            <div
              key={hour}
              className="relative flex-1 cursor-default rounded-sm"
              style={{
                height: `${Math.max(6, value * 100)}%`,
                background: above ? sequentialBlue(value) : 'transparent',
                border: above ? 'none' : `1px solid ${sequentialBlue(value)}`,
              }}
              onMouseEnter={() => setHover(hour)}
              onMouseLeave={() => setHover(null)}
            />
          );
        })}
      </div>
      {hover !== null && (
        <div className="pointer-events-none absolute -top-9 left-0 z-10 w-full">
          <div className="mx-auto w-fit rounded-md border border-hairline bg-surface-1 px-2 py-1 text-[11px] shadow-lg">
            <span className="text-ink-muted">Hour {hover - values.length + 1}</span>{' '}
            <span className="tnum font-medium text-ink">{percent(values[hover])}</span>{' '}
            <span className="text-ink-muted">
              {values[hover]! >= theta ? '· above θ' : '· below θ'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
