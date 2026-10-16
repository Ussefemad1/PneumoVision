import {
  EHR_CATEGORICAL_VALUES,
  EHR_CATEGORICAL_VARIABLES,
  EHR_VARIABLES,
  EHR_WINDOW_HOURS,
  type EhrVariable,
} from './generated/ehr-variables.js';

/**
 * EHR input validation shared by the API and the Analyze page.
 *
 * One implementation, so the CSV preview in the browser reports exactly the
 * row errors the server would reject with, and nothing on either side
 * silently drops a column it does not recognise.
 */

export type EhrIssueCode =
  | 'EHR_UNKNOWN_VARIABLE'
  | 'EHR_INVALID_VALUE'
  | 'EHR_INVALID_TIMESTAMP'
  | 'EHR_FUTURE_TIMESTAMP'
  | 'EHR_OUTSIDE_WINDOW'
  | 'EHR_DUPLICATE_HOUR'
  | 'EHR_MISSING_TIME_COLUMN'
  | 'EHR_TOO_MANY_ROWS';

export interface EhrIssue {
  code: EhrIssueCode;
  message: string;
}

/** Charted values for one hour. Absent keys are gaps; nothing is imputed here. */
export type VitalsInput = Partial<Record<EhrVariable, number | string>>;

export type VitalsValidation =
  { ok: true; values: VitalsInput } | { ok: false; errors: EhrIssue[] };

const KNOWN = new Set<string>(EHR_VARIABLES);
const CATEGORICAL = new Set<string>(EHR_CATEGORICAL_VARIABLES);
const NUMERIC_TEXT = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function isBlank(value: unknown): boolean {
  return value === undefined || value === null || (typeof value === 'string' && !value.trim());
}

/** Resolves a categorical cell to its verbatim MIMIC string, or null if not allowed. */
function categoricalValue(variable: string, raw: number | string): string | null {
  const allowed = EHR_CATEGORICAL_VALUES[variable] ?? [];
  const text = String(raw).trim();
  if (allowed.includes(text)) return text;
  // `1` for capillary refill means '1.0'; only purely numeric codes qualify.
  if (NUMERIC_TEXT.test(text)) {
    const match = allowed.find((v) => NUMERIC_TEXT.test(v) && Number(v) === Number(text));
    if (match !== undefined) return match;
  }
  return null;
}

/**
 * Validates one hour of the 17 raw variables. Unknown names are errors, not
 * ignored; blank cells are gaps and are omitted from the result.
 */
export function validateVitalsValues(values: Record<string, unknown>): VitalsValidation {
  const out: VitalsInput = {};
  const errors: EhrIssue[] = [];

  for (const [variable, raw] of Object.entries(values)) {
    if (!KNOWN.has(variable)) {
      errors.push({ code: 'EHR_UNKNOWN_VARIABLE', message: `Unknown EHR variable "${variable}"` });
      continue;
    }
    if (isBlank(raw)) continue;
    const key = variable as EhrVariable;

    if (typeof raw !== 'number' && typeof raw !== 'string') {
      errors.push({ code: 'EHR_INVALID_VALUE', message: `Invalid value for ${variable}` });
      continue;
    }

    if (CATEGORICAL.has(variable)) {
      const value = categoricalValue(variable, raw);
      if (value === null) {
        errors.push({
          code: 'EHR_INVALID_VALUE',
          message: `"${String(raw)}" is not an allowed value for ${variable}`,
        });
      } else {
        out[key] = value;
      }
      continue;
    }

    const text = String(raw).trim();
    const number = typeof raw === 'number' ? raw : NUMERIC_TEXT.test(text) ? Number(text) : NaN;
    if (Number.isFinite(number)) {
      out[key] = number;
    } else {
      errors.push({
        code: 'EHR_INVALID_VALUE',
        message: `"${text}" is not a number (${variable})`,
      });
    }
  }

  return errors.length > 0 ? { ok: false, errors } : { ok: true, values: out };
}

/**
 * When a row was charted: either an hour offset from submission (0 = the
 * hour ending at submission, -47 = the earliest in the window) or an
 * absolute timestamp. Offsets are what the page sends, so a client clock
 * that runs ahead of the server's cannot turn a row into "future" data.
 */
export type EhrRowTime = { hour: number } | { ts: string };

export const EHR_EARLIEST_HOUR = -(EHR_WINDOW_HOURS - 1);

export function validateEhrHour(hour: number): EhrIssue | null {
  if (!Number.isInteger(hour)) {
    return { code: 'EHR_INVALID_TIMESTAMP', message: `Hour offset ${hour} is not a whole number` };
  }
  if (hour > 0) {
    return { code: 'EHR_FUTURE_TIMESTAMP', message: `Hour offset ${hour} is after submission` };
  }
  if (hour < EHR_EARLIEST_HOUR) {
    return {
      code: 'EHR_OUTSIDE_WINDOW',
      message: `Hour offset ${hour} is outside the ${EHR_WINDOW_HOURS}-hour window`,
    };
  }
  return null;
}

// ── CSV ──────────────────────────────────────────────────────────────────────

export interface ParsedEhrRow {
  /** 1-based line number in the file, for error messages. */
  line: number;
  time: EhrRowTime | null;
  values: VitalsInput;
  errors: EhrIssue[];
}

export interface ParsedEhrCsv {
  headerError: EhrIssue | null;
  rows: ParsedEhrRow[];
}

/** Splits one CSV line, honouring double quotes and `""` escapes. */
function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      cells.push(cell);
      cell = '';
    } else {
      cell += ch;
    }
  }
  cells.push(cell);
  return cells.map((c) => c.trim());
}

const TIME_COLUMNS: Record<string, 'hour' | 'ts'> = { hour: 'hour', ts: 'ts', timestamp: 'ts' };

/**
 * Parses an EHR CSV: a header row with a time column (`hour` offset, or
 * `ts`/`timestamp`) and any of the 17 variable names, then one row per hour.
 */
/** First column of an extracted MIMIC `*_timeseries.csv`: hours since ICU admission. */
const MIMIC_TIME_COLUMN = 'Hours';

const csvCell = (value: string) => (/[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);

/**
 * An extracted MIMIC timeseries as hourly rows with `hour` offsets.
 *
 * Bins exactly as the training Discretizer does (timestep 1, start time
 * zero): a reading at t falls in bin `trunc(t - 1e-6)`, readings after 48 h
 * are dropped, and within a bin the last value of each variable wins. Bin b
 * becomes `hour = b - 47`, so the window ends at "now".
 */
function mimicTimeseriesToHourly(header: string[], rawRows: string[]): string {
  const variables = header.slice(1).map((h) => h.trim());
  const bins = new Map<number, Map<string, string>>();
  for (const raw of rawRows) {
    const cells = splitCsvLine(raw);
    const t = Number(cells[0]);
    if (!cells[0]?.trim() || Number.isNaN(t) || t > EHR_WINDOW_HOURS + 1e-6) continue;
    const bin = Math.max(0, Math.trunc(t - 1e-6));
    const values = bins.get(bin) ?? new Map<string, string>();
    variables.forEach((name, i) => {
      const value = (cells[i + 1] ?? '').trim();
      if (value !== '') values.set(name, value);
    });
    bins.set(bin, values);
  }
  const out = [['hour', ...variables].map(csvCell).join(',')];
  for (const bin of [...bins.keys()].sort((a, b) => a - b)) {
    const values = bins.get(bin)!;
    if (bin >= EHR_WINDOW_HOURS || values.size === 0) continue;
    out.push(
      [String(bin - (EHR_WINDOW_HOURS - 1)), ...variables.map((v) => values.get(v) ?? '')]
        .map(csvCell)
        .join(','),
    );
  }
  return out.join('\n');
}

export function parseEhrCsv(text: string): ParsedEhrCsv {
  const lines = text
    .replace(/^\uFEFF/, '') // byte-order mark
    .split(/\r?\n/)
    .map((raw, i) => ({ raw, line: i + 1 }))
    .filter(({ raw }) => raw.trim() !== '');

  const header = lines[0] ? splitCsvLine(lines[0].raw) : [];
  if (header[0]?.trim() === MIMIC_TIME_COLUMN) {
    return parseEhrCsv(mimicTimeseriesToHourly(header, lines.slice(1).map((l) => l.raw)));
  }
  const timeIndex = header.findIndex((h) => h.toLowerCase() in TIME_COLUMNS);
  if (timeIndex < 0) {
    return {
      headerError: {
        code: 'EHR_MISSING_TIME_COLUMN',
        message: 'The header needs an "hour" column (offset, 0 = now) or a "ts" column',
      },
      rows: [],
    };
  }
  const unknown = header.filter((h, i) => i !== timeIndex && !KNOWN.has(h));
  if (unknown.length > 0) {
    return {
      headerError: {
        code: 'EHR_UNKNOWN_VARIABLE',
        message: `Unknown column${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}`,
      },
      rows: [],
    };
  }
  const timeKind = TIME_COLUMNS[header[timeIndex]!.toLowerCase()]!;

  const rows = lines.slice(1).map(({ raw, line }): ParsedEhrRow => {
    const cells = splitCsvLine(raw);
    const errors: EhrIssue[] = [];
    const cellTime = cells[timeIndex] ?? '';

    let time: EhrRowTime | null = null;
    if (timeKind === 'hour') {
      const hour = NUMERIC_TEXT.test(cellTime) ? Number(cellTime) : NaN;
      const issue: EhrIssue | null = Number.isNaN(hour)
        ? { code: 'EHR_INVALID_TIMESTAMP', message: `"${cellTime}" is not an hour offset` }
        : validateEhrHour(hour);
      if (issue) errors.push(issue);
      else time = { hour };
    } else {
      const ts = new Date(cellTime);
      if (!cellTime || Number.isNaN(ts.getTime())) {
        errors.push({ code: 'EHR_INVALID_TIMESTAMP', message: `"${cellTime}" is not a timestamp` });
      } else {
        time = { ts: ts.toISOString() };
      }
    }

    const record: Record<string, string> = {};
    header.forEach((name, i) => {
      if (i !== timeIndex) record[name] = cells[i] ?? '';
    });
    const validated = validateVitalsValues(record);
    if (!validated.ok) errors.push(...validated.errors);

    return { line, time, values: validated.ok ? validated.values : {}, errors };
  });

  return { headerError: null, rows };
}
