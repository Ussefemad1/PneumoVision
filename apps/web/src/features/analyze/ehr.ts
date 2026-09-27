import {
  ANALYZE_LIMITS,
  EHR_VARIABLES,
  validateVitalsValues,
  type EhrIssue,
  type EhrVariable,
} from '@pneumovision/shared';

/**
 * Client-side EHR state for the Analyze page.
 *
 * The grid holds raw strings exactly as typed; blank cells are gaps and stay
 * gaps. Rows are ordered oldest → newest and the newest row is the hour ending
 * at submission (offset 0), so what is sent is an hour offset, never a
 * client-clock timestamp.
 */

export type GridCells = Partial<Record<EhrVariable, string>>;

export interface GridRow {
  id: string;
  cells: GridCells;
}

let nextId = 0;
export const newRowId = () => `row-${++nextId}`;

/** Hour offset of row `index` in a grid of `count` rows: newest is 0. */
export const hourOffset = (index: number, count: number) => index - (count - 1);

export function rowErrors(row: GridRow): EhrIssue[] {
  const result = validateVitalsValues(row.cells);
  return result.ok ? [] : result.errors;
}

/** True when a single cell's value would be rejected. */
export function cellInvalid(variable: EhrVariable, value: string | undefined): boolean {
  if (!value?.trim()) return false;
  return !validateVitalsValues({ [variable]: value }).ok;
}

export const rowHasData = (row: GridRow) =>
  Object.values(row.cells).some((v) => v !== undefined && v.trim() !== '');

/** The `ehrJson` payload, or null when the grid carries no data at all. */
export function gridPayload(rows: GridRow[]) {
  const out = rows
    .map((row, i) => ({
      hour: hourOffset(i, rows.length),
      values: Object.fromEntries(
        Object.entries(row.cells).filter(([, v]) => v !== undefined && v.trim() !== ''),
      ),
    }))
    .filter((row) => Object.keys(row.values).length > 0);
  return out.length > 0 ? { rows: out } : null;
}

// ── Synthetic presets ────────────────────────────────────────────────────────

export type PresetName = 'stable' | 'deteriorating' | 'borderline';

export const PRESETS: { name: PresetName; label: string; description: string }[] = [
  { name: 'stable', label: 'Stable', description: 'Normal, steady vitals over 12 hours' },
  {
    name: 'deteriorating',
    label: 'Deteriorating',
    description: 'Rising heart and respiratory rate, falling SpO₂ on more oxygen',
  },
  {
    name: 'borderline',
    label: 'Borderline',
    description: 'Mildly abnormal and drifting, but not clearly worsening',
  },
];

const PRESET_HOURS = 12;

/** start → end over the window, rounded to `dp` decimals. */
function ramp(start: number, end: number, i: number, dp = 0): string {
  const value = start + ((end - start) * i) / (PRESET_HOURS - 1);
  return value.toFixed(dp);
}

/**
 * Fabricated, deterministic vitals — never derived from any real record.
 * Some variables are charted only every few hours, so the grid shows real
 * gaps the way a chart does.
 */
export function presetRows(name: PresetName): GridRow[] {
  const shape = {
    stable: {
      hr: [76, 80],
      rr: [15, 16],
      spo2: [98, 97],
      fio2: [0.21, 0.21],
      sbp: [124, 120],
      temp: [36.7, 36.9],
      gcs: ['15', '15'],
    },
    deteriorating: {
      hr: [88, 128],
      rr: [18, 32],
      spo2: [96, 86],
      fio2: [0.21, 0.6],
      sbp: [118, 92],
      temp: [37.2, 38.7],
      gcs: ['15', '13'],
    },
    borderline: {
      hr: [96, 104],
      rr: [20, 23],
      spo2: [94, 93],
      fio2: [0.28, 0.35],
      sbp: [108, 104],
      temp: [37.6, 37.9],
      gcs: ['15', '14'],
    },
  }[name];

  return Array.from({ length: PRESET_HOURS }, (_, i): GridRow => {
    const sbp = Number(ramp(shape.sbp[0]!, shape.sbp[1]!, i));
    const dbp = Math.round(sbp * 0.6);
    const cells: GridCells = {
      'Heart Rate': ramp(shape.hr[0]!, shape.hr[1]!, i),
      'Respiratory rate': ramp(shape.rr[0]!, shape.rr[1]!, i),
      'Oxygen saturation': ramp(shape.spo2[0]!, shape.spo2[1]!, i),
      'Fraction inspired oxygen': ramp(shape.fio2[0]!, shape.fio2[1]!, i, 2),
      'Systolic blood pressure': String(sbp),
      'Diastolic blood pressure': String(dbp),
      'Mean blood pressure': String(Math.round((sbp + 2 * dbp) / 3)),
    };
    if (i % 2 === 0) cells.Temperature = ramp(shape.temp[0]!, shape.temp[1]!, i, 1);
    if (i % 4 === 0) {
      cells['Glascow coma scale total'] = i < PRESET_HOURS / 2 ? shape.gcs[0]! : shape.gcs[1]!;
      cells.Glucose = String(118 + ((i * 7) % 30));
    }
    if (i === 0) {
      cells.Weight = '78';
      cells.Height = '172';
    }
    return { id: newRowId(), cells };
  });
}

export const MAX_ROWS = ANALYZE_LIMITS.ehrRows;
export const VARIABLES = EHR_VARIABLES;
