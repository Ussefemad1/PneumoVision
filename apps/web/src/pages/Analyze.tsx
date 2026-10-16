import type { AnalyzeResponse, StayDetail, Task } from '@pneumovision/shared';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';

import { api, apiErrorCode, apiErrorMessage, getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { getSocket } from '../lib/socket.js';
import { Button, Card } from '../components/ui.jsx';
import { CxrDrop, cxrProblem } from '../features/analyze/CxrDrop.jsx';
import { gridPayload, rowErrors, type GridRow } from '../features/analyze/ehr.js';
import {
  EhrPanel,
  csvHasErrors,
  type CsvState,
  type EhrTab,
} from '../features/analyze/EhrPanel.jsx';
import {
  NotesEditor,
  noteTooLong,
  usableNotes,
  type DraftNote,
} from '../features/analyze/NotesEditor.jsx';

const TASKS: { key: Task; title: string }[] = [
  { key: 'pneumonia', title: 'Pneumonia' },
  { key: 'mortality', title: '48-hour mortality' },
];

type Phase = 'idle' | 'uploading' | 'running';

interface SubmitError {
  code: string | null;
  message: string;
}

/**
 * Ad-hoc analysis: submit any mix of chest X-ray, EHR and notes and open the
 * resulting prediction report. The server is authoritative on validation and
 * F8 leakage control; the checks here only give earlier feedback.
 */
export function AnalyzePage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const stayId = params.get('stayId');

  const [task, setTask] = useState<Task>('pneumonia');
  const [cxr, setCxr] = useState<File | null>(null);
  const [ehrTab, setEhrTab] = useState<EhrTab>('grid');
  const [rows, setRows] = useState<GridRow[]>([]);
  const [csv, setCsv] = useState<CsvState | null>(null);
  const [notes, setNotes] = useState<DraftNote[]>([]);

  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<SubmitError | null>(null);

  const stay = useQuery({
    queryKey: ['stay', stayId],
    queryFn: () => getData<StayDetail>(`/stays/${stayId}`),
    enabled: Boolean(stayId),
  });

  // The API emits prediction:* to this user's room; follow the run while
  // submitting. (In-process today, so the HTTP response usually wins.)
  useEffect(() => {
    const socket = getSocket();
    const onRunning = () => setPhase((p) => (p === 'idle' ? p : 'running'));
    socket.on('prediction:queued', onRunning);
    socket.on('prediction:running', onRunning);
    return () => {
      socket.off('prediction:queued', onRunning);
      socket.off('prediction:running', onRunning);
    };
  }, []);

  // ── What would be sent ────────────────────────────────────────────────────
  const ehrFromCsv = ehrTab === 'csv';
  const grid = ehrFromCsv ? null : gridPayload(rows);
  const csvRows = ehrFromCsv && csv?.parsed ? csv.parsed.rows.length : 0;
  const hasEhr = ehrFromCsv ? csvRows > 0 : grid !== null;
  const hasCxr = cxr !== null;
  const sendable = usableNotes(task, notes);

  const problems: string[] = [];
  if (cxr && cxrProblem(cxr)) problems.push('Fix or remove the chest X-ray.');
  if (!ehrFromCsv && rows.some((r) => rowErrors(r).length > 0)) {
    problems.push('Fix the highlighted EHR cells.');
  }
  if (ehrFromCsv && csvHasErrors(csv)) problems.push('Fix the EHR CSV errors or remove it.');
  if (notes.some((n) => n.error || noteTooLong(n))) problems.push('Fix the note errors.');
  if (stayId && stay.isError) problems.push('The requested stay could not be loaded.');

  const noModality = !hasEhr && !hasCxr && sendable.length === 0;
  const busy = phase !== 'idle';
  const canSubmit = !busy && !noModality && problems.length === 0;

  async function submit() {
    setError(null);
    setPhase('uploading');
    setProgress(0);

    const form = new FormData();
    form.append('task', task);
    if (stayId) form.append('stayId', stayId);
    if (cxr) form.append('cxr', cxr);
    if (ehrFromCsv && csv) form.append('ehr', csv.file);
    if (grid) form.append('ehrJson', JSON.stringify(grid));

    // Every non-empty note goes as a file with its type alongside. Excluded
    // discharge notes are sent too: the server withholds them and records
    // the exclusion, so the report can show it.
    const withText = notes.filter((n) => n.text.trim() !== '');
    withText.forEach((note, i) => {
      form.append('notes', new Blob([note.text], { type: 'text/plain' }), `note-${i + 1}.txt`);
    });
    if (withText.length > 0) {
      form.append('noteTypes', JSON.stringify(withText.map((n) => n.type)));
    }

    try {
      const res = await api.post<{ data: AnalyzeResponse }>('/analyze', form, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(e.loaded / e.total);
          if (e.total && e.loaded >= e.total) setPhase('running');
        },
      });
      void navigate(`/predictions/${res.data.data.predictionId}`);
    } catch (err) {
      // Every input stays exactly as it was, so the user can fix and retry.
      setError({
        code: apiErrorCode(err),
        message: apiErrorMessage(err, 'The analysis could not be submitted.'),
      });
      setPhase('idle');
      setProgress(null);
    }
  }

  const statusText =
    phase === 'uploading' && progress !== null ? `${Math.round(progress * 100)}%` : '';

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div>
        <Link to="/analyze" className="text-xs text-ink-muted hover:text-ink hover:underline">
          ← Back to analyses
        </Link>
        <h1 className="mt-2 text-2xl font-semibold text-ink">New analysis</h1>
      </div>

      {stayId && (
        <div
          role="status"
          className="rounded-lg border border-hairline bg-surface-2 px-4 py-3 text-sm"
        >
          {stay.isLoading && 'Loading stay…'}
          {stay.isError && (
            <span className="text-status-critical">{apiErrorMessage(stay.error)}</span>
          )}
          {stay.data && (
            <>
              Patient <span className="font-medium">{stay.data.patient.pseudoId}</span> ·{' '}
              {stay.data.ward} · bed {stay.data.bedLabel}{' '}
              <Link to="/analyze/new" className="ml-2 underline">
                Remove
              </Link>
            </>
          )}
        </div>
      )}

      <Card title="Task">
        <div role="radiogroup" aria-label="Prediction task" className="grid gap-3 sm:grid-cols-2">
          {TASKS.map((t) => (
            <label
              key={t.key}
              className={cn(
                'cursor-pointer rounded-lg border p-4 transition',
                task === t.key
                  ? 'border-[var(--series-mortality)] bg-surface-3'
                  : 'border-hairline hover:bg-surface-2',
              )}
            >
              <span className="flex items-center gap-2">
                <input
                  type="radio"
                  name="task"
                  value={t.key}
                  checked={task === t.key}
                  onChange={() => setTask(t.key)}
                  disabled={busy}
                />
                <span className="text-sm font-medium text-ink">{t.title}</span>
              </span>
            </label>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[1fr_2fr]">
        <Card title="Chest X-ray">
          <CxrDrop file={cxr} onFile={setCxr} />
        </Card>

        <Card title="Clinical notes">
          <NotesEditor task={task} notes={notes} onNotes={setNotes} />
        </Card>
      </div>

      <Card title="Vitals">
        <EhrPanel
          tab={ehrTab}
          onTab={setEhrTab}
          rows={rows}
          onRows={setRows}
          csv={csv}
          onCsv={setCsv}
        />
      </Card>

      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" disabled={!canSubmit} onClick={() => void submit()}>
            {busy ? 'Analyzing…' : 'Run analysis'}
          </Button>
          <span role="status" className="text-xs text-ink-muted">
            {statusText}
          </span>
          {phase === 'uploading' && progress !== null && (
            <progress
              value={progress}
              max={1}
              className="h-1.5 w-40"
              aria-label="Upload progress"
            />
          )}
        </div>
        {problems.length > 0 && !busy && (
          <ul className="mt-2 text-xs text-status-critical">
            {problems.map((p) => (
              <li key={p}>✕ {p}</li>
            ))}
          </ul>
        )}
        {error && (
          <div
            role="alert"
            className="mt-3 rounded-md border border-status-critical/40 bg-status-critical/10 px-3 py-2 text-sm"
          >
            <span aria-hidden="true">✕ </span>
            {error.message}
          </div>
        )}
      </Card>
    </div>
  );
}
