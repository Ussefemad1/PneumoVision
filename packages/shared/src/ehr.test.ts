import { describe, expect, it } from 'vitest';

import { parseEhrCsv, validateVitalsValues } from './ehr.js';

describe('validateVitalsValues', () => {
  it('parses numbers, keeps allowed categorical strings, and leaves gaps empty', () => {
    const result = validateVitalsValues({
      'Heart Rate': '96',
      'Glascow coma scale total': '15',
      Temperature: '',
      'Oxygen saturation': null,
    });
    expect(result).toEqual({
      ok: true,
      values: { 'Heart Rate': 96, 'Glascow coma scale total': '15' },
    });
  });

  it('reports unknown variables instead of ignoring them', () => {
    const result = validateVitalsValues({ 'Heart rate': 90 });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors[0]).toMatchObject({ code: 'EHR_UNKNOWN_VARIABLE' });
  });

  it('rejects categorical values outside the allowed list and non-numeric numbers', () => {
    const result = validateVitalsValues({ 'Glascow coma scale total': '99', 'Heart Rate': 'fast' });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.map((e) => e.code)).toEqual(['EHR_INVALID_VALUE', 'EHR_INVALID_VALUE']);
    }
  });

  it('accepts a numeric categorical code written as a number', () => {
    expect(validateVitalsValues({ 'Capillary refill rate': 1 })).toEqual({
      ok: true,
      values: { 'Capillary refill rate': '1.0' },
    });
  });
});

describe('parseEhrCsv', () => {
  it('parses hour-offset rows and reports row-level errors with line numbers', () => {
    const csv = [
      'hour,Heart Rate,Glascow coma scale total',
      '-2,96,15',
      '-1,abc,15',
      '0,101,',
    ].join('\n');
    const result = parseEhrCsv(csv);
    expect(result.headerError).toBeNull();
    expect(result.rows).toHaveLength(3);
    expect(result.rows[0]).toMatchObject({ line: 2, time: { hour: -2 }, errors: [] });
    expect(result.rows[1]!.errors[0]).toMatchObject({ code: 'EHR_INVALID_VALUE' });
    expect(result.rows[2]!.values).toEqual({ 'Heart Rate': 101 });
  });

  it('accepts absolute timestamps', () => {
    const result = parseEhrCsv('ts,Heart Rate\n2026-01-01T10:00:00Z,90\n');
    expect(result.rows[0]!.time).toEqual({ ts: '2026-01-01T10:00:00.000Z' });
  });

  it('flags a missing time column and unknown columns in the header', () => {
    expect(parseEhrCsv('Heart Rate\n90').headerError?.code).toBe('EHR_MISSING_TIME_COLUMN');
    expect(parseEhrCsv('hour,Pulse\n0,90').headerError?.code).toBe('EHR_UNKNOWN_VARIABLE');
  });

  it('flags hours outside the 48-hour window', () => {
    const result = parseEhrCsv('hour,Heart Rate\n-48,90\n1,90\n');
    expect(result.rows.map((r) => r.errors[0]?.code)).toEqual([
      'EHR_OUTSIDE_WINDOW',
      'EHR_FUTURE_TIMESTAMP',
    ]);
  });

  it('handles quoted cells and CRLF', () => {
    const result = parseEhrCsv('hour,"Glascow coma scale eye opening"\r\n0,"4 Spontaneously"\r\n');
    expect(result.rows[0]!.values).toEqual({ 'Glascow coma scale eye opening': '4 Spontaneously' });
  });
});
