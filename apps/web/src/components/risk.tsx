import { percent } from '../lib/format.js';
import { riskBand } from '../lib/risk.js';
import { cn } from '../lib/cn.js';
import { Chip } from './ui.jsx';

/**
 * Risk presentation.
 *
 * Every component here pairs the status colour with a shape glyph and a word,
 * so the band survives greyscale printing and colour-vision deficiency.
 */

export function RiskBadge({
  probability,
  className,
}: {
  probability: number | null | undefined;
  className?: string | undefined;
}) {
  const band = riskBand(probability);
  return (
    <Chip className={cn(band.chipClass, 'tnum', className)}>
      <span aria-hidden="true">{band.icon}</span>
      <span>{percent(probability)}</span>
      <span className="opacity-80">{band.label}</span>
    </Chip>
  );
}

/**
 * Headline gauge: a semicircular arc plus the number and the band word.
 *
 * The arc is the magnitude channel and the label is the identity channel;
 * neither is load-bearing on its own.
 */
export function RiskGauge({
  probability,
  label,
  sublabel,
}: {
  probability: number | null;
  label: string;
  sublabel?: string | undefined;
}) {
  const band = riskBand(probability);
  const value = probability ?? 0;

  // Semicircle from 180° to 0°.
  const radius = 54;
  const circumference = Math.PI * radius;
  const dash = circumference * value;

  return (
    <figure className="flex flex-col items-center gap-1">
      <svg
        viewBox="0 0 140 82"
        className="h-[82px] w-[140px]"
        role="img"
        aria-label={`${label}: ${percent(probability)} (${band.label})`}
      >
        <path
          d="M 16 70 A 54 54 0 0 1 124 70"
          fill="none"
          stroke="var(--grid)"
          strokeWidth="11"
          strokeLinecap="round"
        />
        {probability !== null && (
          <path
            d="M 16 70 A 54 54 0 0 1 124 70"
            fill="none"
            stroke={band.cssVar}
            strokeWidth="11"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
          />
        )}
        <text
          x="70"
          y="60"
          textAnchor="middle"
          className="tnum fill-[var(--text-primary)] text-[22px] font-semibold"
        >
          {percent(probability)}
        </text>
      </svg>
      <figcaption className="text-center">
        <div className="text-sm font-medium text-ink">{label}</div>
        <div
          className="mt-0.5 flex items-center justify-center gap-1 text-xs"
          style={{ color: band.cssVar }}
        >
          <span aria-hidden="true">{band.icon}</span>
          <span className="font-medium">{probability === null ? 'Not scored' : band.label}</span>
        </div>
        {sublabel && <div className="mt-0.5 text-[11px] text-ink-muted">{sublabel}</div>}
      </figcaption>
    </figure>
  );
}

/**
 * F5: modality availability badges.
 *
 * A present modality is a filled chip with a tick; a missing one is outlined
 * with a dash, so absence reads without colour.
 */
export function ModalityBadges({
  availability,
  className,
}: {
  availability: { ehr: boolean; cxr: boolean; notes: boolean };
  className?: string | undefined;
}) {
  const items = [
    { key: 'ehr', label: 'EHR', present: availability.ehr },
    { key: 'cxr', label: 'CXR', present: availability.cxr },
    { key: 'notes', label: 'Notes', present: availability.notes },
  ];

  return (
    <div className={cn('flex flex-wrap gap-1', className)}>
      {items.map((item) => (
        <span
          key={item.key}
          title={`${item.label}: ${item.present ? 'available' : 'not available'}`}
          className={cn(
            'inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium',
            item.present
              ? 'bg-surface-3 text-ink-secondary ring-1 ring-hairline'
              : 'text-ink-muted ring-1 ring-dashed ring-hairline',
          )}
        >
          <span aria-hidden="true">{item.present ? '✓' : '–'}</span>
          {item.label}
        </span>
      ))}
    </div>
  );
}

/**
 * Sparkline: bare trend, no axes. One series, so no legend — the row it sits
 * in names it. Renders a dash when there is nothing to draw rather than an
 * empty box.
 */
export function Sparkline({
  values,
  width = 96,
  height = 26,
  ariaLabel,
}: {
  values: number[];
  width?: number | undefined;
  height?: number | undefined;
  ariaLabel?: string | undefined;
}) {
  if (values.length < 2) {
    return (
      <span className="text-xs text-ink-muted" title="Not enough scorings yet">
        —
      </span>
    );
  }

  const band = riskBand(values.at(-1));
  // Fixed 0–1 domain: a sparkline normalised per-row would make a stable
  // patient look as dramatic as a deteriorating one.
  const step = width / (values.length - 1);
  const y = (v: number) => height - 2 - v * (height - 4);
  const points = values.map((v, i) => `${i * step},${y(v)}`).join(' ');
  const last = values.at(-1)!;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={ariaLabel ?? `Trend, latest ${percent(last)}`}
      className="overflow-visible"
    >
      <polyline
        points={points}
        fill="none"
        stroke={band.cssVar}
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle
        cx={(values.length - 1) * step}
        cy={y(last)}
        r="3"
        fill={band.cssVar}
        stroke="var(--surface-1)"
        strokeWidth="1.5"
      />
    </svg>
  );
}
