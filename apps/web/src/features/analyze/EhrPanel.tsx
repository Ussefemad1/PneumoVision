import {
  ANALYZE_LIMITS,
  EHR_CATEGORICAL_VALUES,
  EHR_CATEGORICAL_VARIABLES,
  parseEhrCsv,
  type EhrVariable,
  type ParsedEhrCsv,
} from '@pneumovision/shared';
import { useId, useState } from 'react';

import { cn } from '../../lib/cn.js';
import { readText } from '../../lib/files.js';
import { Button } from '../../components/ui.jsx';
import {
  MAX_ROWS,
  PRESETS,
  VARIABLES,
  cellInvalid,
  hourOffset,
  newRowId,
  presetRows,
  rowErrors,
  type GridRow,
} from './ehr.js';

export type EhrTab = 'grid' | 'csv' | 'synthetic';

export interface CsvState {
  file: File;
  parsed: ParsedEhrCsv | null;
  /** File-level problem: too big, unreadable, too many rows. */
  error: string | null;
}

/** Short column headers; the full name is in each cell's accessible label. */
const SHORT: Partial<Record<EhrVariable, string>> = {
  'Capillary refill rate': 'Cap refill',
  'Diastolic blood pressure': 'DBP',
  'Fraction inspired oxygen': 'FiO₂',
  'Glascow coma scale eye opening': 'GCS eye',
  'Glascow coma scale motor response': 'GCS motor',
  'Glascow coma scale total': 'GCS total',
  'Glascow coma scale verbal response': 'GCS verbal',
  'Heart Rate': 'HR',
  'Mean blood pressure': 'MBP',
  'Oxygen saturation': 'SpO₂',
  'Respiratory rate': 'RR',
  'Systolic blood pressure': 'SBP',
  Temperature: 'Temp',
};

const hourLabel = (offset: number) => (offset === 0 ? 'now' : `${offset} h`);

export function EhrPanel({
  tab,
  onTab,
  rows,
  onRows,
  csv,
  onCsv,
}: {
  tab: EhrTab;
  onTab: (tab: EhrTab) => void;
  rows: GridRow[];
  onRows: (rows: GridRow[]) => void;
  csv: CsvState | null;
  onCsv: (csv: CsvState | null) => void;
}) {
  const baseId = useId();
  const tabs: { key: EhrTab; label: string }[] = [
    { key: 'grid', label: 'Grid' },
    { key: 'csv', label: 'CSV upload' },
    { key: 'synthetic', label: 'Synthetic example' },
  ];

  return (
    <div>
      <div role="tablist" aria-label="EHR input method" className="flex flex-wrap gap-1">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            id={`${baseId}-${t.key}`}
            aria-selected={tab === t.key}
            aria-controls={`${baseId}-panel`}
            onClick={() => onTab(t.key)}
            className={cn(
              'rounded-md px-3 py-1.5 text-xs font-medium',
              tab === t.key
                ? 'bg-surface-3 text-ink'
                : 'text-ink-secondary hover:bg-surface-2 hover:text-ink',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-ink-muted">
        {tab === 'csv'
          ? 'The uploaded CSV is sent; the grid is ignored while this tab is open.'
          : 'The grid is sent. Empty cells are gaps — nothing is filled in for you.'}
      </p>

      <div
        role="tabpanel"
        id={`${baseId}-panel`}
        aria-labelledby={`${baseId}-${tab}`}
        className="mt-3"
      >
        {tab === 'grid' && <EhrGrid rows={rows} onRows={onRows} />}
        {tab === 'csv' && <EhrCsv csv={csv} onCsv={onCsv} />}
        {tab === 'synthetic' && (
          <div className="grid gap-2 sm:grid-cols-3">
            {PRESETS.map((preset) => (
              <button
                key={preset.name}
                type="button"
                onClick={() => {
                  onRows(presetRows(preset.name));
                  onTab('grid');
                }}
                className="rounded-lg border border-hairline p-3 text-left hover:bg-surface-2"
              >
                <span className="block text-sm font-medium text-ink">{preset.label}</span>
                <span className="mt-1 block text-[11px] text-ink-muted">{preset.description}</span>
              </button>
            ))}
            <p className="text-[11px] text-ink-muted sm:col-span-3">
              Fabricated values for demonstration. Loading a preset replaces the grid.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Grid ─────────────────────────────────────────────────────────────────────

function EhrGrid({ rows, onRows }: { rows: GridRow[]; onRows: (rows: GridRow[]) => void }) {
  const setCell = (rowId: string, variable: EhrVariable, value: string) =>
    onRows(
      rows.map((r) => (r.id === rowId ? { ...r, cells: { ...r.cells, [variable]: value } } : r)),
    );

  const addEarlier = () => onRows([{ id: newRowId(), cells: {} }, ...rows]);

  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-hairline p-4 text-center">
        <p className="text-sm text-ink-secondary">No EHR rows — the EHR branch will be missing.</p>
        <Button size="sm" className="mt-3" onClick={addEarlier}>
          Add an hour
        </Button>
      </div>
    );
  }

  return (
    <div>
      <div className="max-h-96 overflow-auto rounded-lg border border-hairline">
        <table aria-label="EHR grid" className="min-w-max border-collapse text-xs">
          <thead className="sticky top-0 bg-surface-2">
            <tr>
              <th scope="col" className="sticky left-0 bg-surface-2 px-2 py-1.5 text-left">
                Hour
              </th>
              {VARIABLES.map((v) => (
                <th
                  key={v}
                  scope="col"
                  title={v}
                  className="px-1.5 py-1.5 text-left font-medium text-ink-secondary"
                >
                  {SHORT[v] ?? v}
                </th>
              ))}
              <th scope="col" className="px-2 py-1.5">
                <span className="sr-only">Remove</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const offset = hourOffset(i, rows.length);
              const errors = rowErrors(row);
              return (
                <tr key={row.id} className="border-t border-hairline">
                  <th
                    scope="row"
                    className={cn(
                      'sticky left-0 bg-surface-1 px-2 py-1 text-left font-medium',
                      errors.length > 0 ? 'text-status-critical' : 'text-ink',
                    )}
                    title={errors.map((e) => e.message).join('\n') || undefined}
                  >
                    {errors.length > 0 && <span aria-hidden="true">✕ </span>}
                    {hourLabel(offset)}
                  </th>
                  {VARIABLES.map((variable) => {
                    const value = row.cells[variable] ?? '';
                    const invalid = cellInvalid(variable, value);
                    const label = `${variable} at ${hourLabel(offset)}`;
                    const common = cn(
                      'w-20 rounded border bg-surface-1 px-1 py-0.5 text-xs text-ink',
                      invalid ? 'border-status-critical' : 'border-hairline',
                    );
                    return (
                      <td key={variable} className="px-1 py-1">
                        {EHR_CATEGORICAL_VARIABLES.includes(variable) ? (
                          <select
                            aria-label={label}
                            aria-invalid={invalid || undefined}
                            value={value}
                            onChange={(e) => setCell(row.id, variable, e.target.value)}
                            className={cn(common, 'w-28')}
                          >
                            <option value="">—</option>
                            {(EHR_CATEGORICAL_VALUES[variable] ?? []).map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            aria-label={label}
                            aria-invalid={invalid || undefined}
                            inputMode="decimal"
                            value={value}
                            onChange={(e) => setCell(row.id, variable, e.target.value)}
                            className={common}
                          />
                        )}
                      </td>
                    );
                  })}
                  <td className="px-1 py-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      aria-label={`Remove hour ${hourLabel(offset)}`}
                      onClick={() => onRows(rows.filter((r) => r.id !== row.id))}
                    >
                      ✕
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={addEarlier} disabled={rows.length >= MAX_ROWS}>
          Add earlier hour
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onRows([])}>
          Clear EHR
        </Button>
        <span className="text-[11px] text-ink-muted">
          {rows.length} of {MAX_ROWS} hours · newest row is the hour ending now
        </span>
      </div>
    </div>
  );
}

// ── CSV ──────────────────────────────────────────────────────────────────────

export async function readCsvFile(file: File): Promise<CsvState> {
  if (file.size > ANALYZE_LIMITS.ehrCsvBytes) {
    return { file, parsed: null, error: 'The CSV is larger than 1 MB.' };
  }
  const parsed = parseEhrCsv(await readText(file));
  const error =
    parsed.rows.length > ANALYZE_LIMITS.ehrRows
      ? `The CSV has ${parsed.rows.length} rows; at most ${ANALYZE_LIMITS.ehrRows} are accepted.`
      : null;
  return { file, parsed, error };
}

export function csvHasErrors(csv: CsvState | null): boolean {
  if (!csv) return false;
  return (
    csv.error !== null ||
    !csv.parsed ||
    csv.parsed.headerError !== null ||
    csv.parsed.rows.some((r) => r.errors.length > 0)
  );
}

function EhrCsv({ csv, onCsv }: { csv: CsvState | null; onCsv: (csv: CsvState | null) => void }) {
  const [reading, setReading] = useState(false);

  return (
    <div>
      <label className="block">
        <span className="text-xs text-ink-secondary">
          Header: <code>hour</code> (0 = now, −1 = an hour earlier …) or <code>ts</code> (ISO time),
          then any of the 17 variable names. At most 48 rows, 1 MB.
        </span>
        <input
          aria-label="EHR CSV file"
          type="file"
          accept=".csv,text/csv"
          className="mt-2 block w-full text-xs"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (!file) return;
            setReading(true);
            void readCsvFile(file)
              .then(onCsv)
              .finally(() => setReading(false));
          }}
        />
      </label>

      {reading && <p className="mt-2 text-xs text-ink-muted">Reading…</p>}

      {csv && (
        <div className="mt-3">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-xs text-ink-secondary">{csv.file.name}</span>
            <Button size="sm" variant="ghost" onClick={() => onCsv(null)}>
              Remove CSV
            </Button>
          </div>
          {csv.error && <p className="mt-1 text-xs text-status-critical">✕ {csv.error}</p>}
          {csv.parsed?.headerError && (
            <p className="mt-1 text-xs text-status-critical">✕ {csv.parsed.headerError.message}</p>
          )}
          {csv.parsed && csv.parsed.rows.length > 0 && (
            <div className="mt-2 max-h-64 overflow-auto rounded-lg border border-hairline">
              <table aria-label="CSV preview" className="w-full text-xs">
                <thead className="sticky top-0 bg-surface-2 text-left">
                  <tr>
                    <th scope="col" className="px-2 py-1">
                      Line
                    </th>
                    <th scope="col" className="px-2 py-1">
                      Time
                    </th>
                    <th scope="col" className="px-2 py-1">
                      Values
                    </th>
                    <th scope="col" className="px-2 py-1">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {csv.parsed.rows.map((row) => (
                    <tr key={row.line} className="border-t border-hairline align-top">
                      <td className="px-2 py-1">Line {row.line}</td>
                      <td className="px-2 py-1">
                        {row.time === null
                          ? '—'
                          : 'hour' in row.time
                            ? hourLabel(row.time.hour)
                            : row.time.ts}
                      </td>
                      <td className="px-2 py-1">{Object.keys(row.values).length}</td>
                      <td className="px-2 py-1">
                        {row.errors.length === 0 ? (
                          <span className="text-status-good">
                            <span aria-hidden="true">✓</span> OK
                          </span>
                        ) : (
                          <ul className="text-status-critical">
                            {row.errors.map((err, i) => (
                              <li key={i}>
                                <span aria-hidden="true">✕ </span>
                                {err.message}
                              </li>
                            ))}
                          </ul>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
