/**
 * Risk banding.
 *
 * Colour is never the only signal: every band carries a distinct icon glyph
 * and a text label, so the four levels remain distinguishable in greyscale,
 * under any colour-vision deficiency, and in forced-colours mode. The hexes
 * come from the reserved status ramp and are not used for any data series.
 */

export type RiskBand = 'low' | 'moderate' | 'high' | 'critical';

export interface RiskBandInfo {
  band: RiskBand;
  label: string;
  /** Distinct shape per band — the non-colour channel. */
  icon: string;
  /** Tailwind classes for a filled chip. */
  chipClass: string;
  /** The status token, for chart marks and gauges. */
  cssVar: string;
}

const BANDS: Record<RiskBand, RiskBandInfo> = {
  low: {
    band: 'low',
    label: 'Low',
    icon: '●',
    chipClass: 'bg-status-good/15 text-status-good ring-1 ring-status-good/40',
    cssVar: 'var(--status-good)',
  },
  moderate: {
    band: 'moderate',
    label: 'Moderate',
    icon: '◆',
    chipClass:
      'bg-status-warning/20 text-[#8a5d00] dark:text-status-warning ring-1 ring-status-warning/50',
    cssVar: 'var(--status-warning)',
  },
  high: {
    band: 'high',
    label: 'High',
    icon: '▲',
    chipClass:
      'bg-status-serious/20 text-[#a2482a] dark:text-status-serious ring-1 ring-status-serious/50',
    cssVar: 'var(--status-serious)',
  },
  critical: {
    band: 'critical',
    label: 'Critical',
    icon: '■',
    chipClass: 'bg-status-critical/15 text-status-critical ring-1 ring-status-critical/50',
    cssVar: 'var(--status-critical)',
  },
};

/** Thresholds mirror the alert rules in the API (services/alertEngine.ts). */
export function riskBand(probability: number | null | undefined): RiskBandInfo {
  if (probability === null || probability === undefined) return BANDS.low;
  if (probability >= 0.7) return BANDS.critical;
  if (probability >= 0.5) return BANDS.high;
  if (probability >= 0.25) return BANDS.moderate;
  return BANDS.low;
}

export const SEVERITY_INFO = {
  info: { label: 'Info', icon: '●', cssVar: 'var(--status-good)', chipClass: BANDS.low.chipClass },
  warning: {
    label: 'Warning',
    icon: '◆',
    cssVar: 'var(--status-warning)',
    chipClass: BANDS.moderate.chipClass,
  },
  critical: {
    label: 'Critical',
    icon: '■',
    cssVar: 'var(--status-critical)',
    chipClass: BANDS.critical.chipClass,
  },
} as const;
