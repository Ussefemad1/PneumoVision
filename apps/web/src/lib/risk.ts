/**
 * Risk banding.
 *
 * Colour is never the only signal: every band carries a distinct icon glyph
 * and a text label, so the four levels remain distinguishable in greyscale,
 * under any colour-vision deficiency, and in forced-colours mode. The hexes
 * come from the reserved status ramp and are not used for any data series.
 *
 * Two colours per band, and they are not interchangeable. `cssVar` is the
 * validated *fill* ramp — for strokes, gauge arcs and chart marks only. It is
 * deliberately sub-3:1 on a light surface, so it must never colour type;
 * `inkVar` is the text-safe companion. See index.css.
 */

export type RiskBand = 'low' | 'moderate' | 'high' | 'critical';

export interface RiskBandInfo {
  band: RiskBand;
  label: string;
  /** Distinct shape per band — the non-colour channel. */
  icon: string;
  /** Colour-only chip class: fill, text and hairline, theme-aware in CSS. */
  chipClass: string;
  /** The status token, for chart marks and gauges. Strokes and fills only. */
  cssVar: string;
  /** Text-safe companion of cssVar — use this wherever the band colours type. */
  inkVar: string;
}

const BANDS: Record<RiskBand, RiskBandInfo> = {
  low: {
    band: 'low',
    label: 'Low',
    icon: '●',
    chipClass: 'chip-low',
    cssVar: 'var(--status-good)',
    inkVar: 'var(--status-good-ink)',
  },
  moderate: {
    band: 'moderate',
    label: 'Moderate',
    icon: '◆',
    chipClass: 'chip-moderate',
    cssVar: 'var(--status-warning)',
    inkVar: 'var(--status-warning-ink)',
  },
  high: {
    band: 'high',
    label: 'High',
    icon: '▲',
    chipClass: 'chip-high',
    cssVar: 'var(--status-serious)',
    inkVar: 'var(--status-serious-ink)',
  },
  critical: {
    band: 'critical',
    label: 'Critical',
    icon: '■',
    chipClass: 'chip-critical',
    cssVar: 'var(--status-critical)',
    inkVar: 'var(--status-critical-ink)',
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
  info: {
    label: 'Info',
    icon: '●',
    cssVar: 'var(--status-good)',
    inkVar: 'var(--status-good-ink)',
    chipClass: BANDS.low.chipClass,
  },
  warning: {
    label: 'Warning',
    icon: '◆',
    cssVar: 'var(--status-warning)',
    inkVar: 'var(--status-warning-ink)',
    chipClass: BANDS.moderate.chipClass,
  },
  critical: {
    label: 'Critical',
    icon: '■',
    cssVar: 'var(--status-critical)',
    inkVar: 'var(--status-critical-ink)',
    chipClass: BANDS.critical.chipClass,
  },
} as const;
