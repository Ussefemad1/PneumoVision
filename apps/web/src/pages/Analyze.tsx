import type { Prediction } from '@pneumovision/shared';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, apiErrorMessage } from '../lib/api.js';
import { getSocket } from '../lib/socket.js';
import { Button, Card, ErrorState } from '../components/ui.jsx';

type Task = 'mortality' | 'pneumonia';

const SAMPLE_EHR = {
  rows: [
    { ts: new Date(Date.now() - 2 * 3_600_000).toISOString(), values: { 'Heart Rate': 96, 'Respiratory rate': 22, 'Oxygen saturation': 94, 'Systolic blood pressure': 116, 'Diastolic blood pressure': 68, Temperature: 37.2 } },
    { ts: new Date(Date.now() - 3_600_000).toISOString(), values: { 'Heart Rate': 102, 'Respiratory rate': 24, 'Oxygen saturation': 92, 'Systolic blood pressure': 110, 'Diastolic blood pressure': 64, Temperature: 37.6 } },
  ],
};

export function AnalyzePage() {
  const navigate = useNavigate();
  const [task, setTask] = useState<Task>('pneumonia');
  const [cxr, setCxr] = useState<File | null>(null);
  const [notes, setNotes] = useState('');
  const [includeEhr, setIncludeEhr] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState('Ready to analyze');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const socket = getSocket();
    const onQueued = () => setStatus('Prediction queued…');
    const onRunning = () => setStatus('Running multimodal inference…');
    socket.on('prediction:queued', onQueued);
    socket.on('prediction:running', onRunning);
    return () => {
      socket.off('prediction:queued', onQueued);
      socket.off('prediction:running', onRunning);
    };
  }, []);

  async function submit() {
    if (!cxr && !notes.trim() && !includeEhr) {
      setError('Select at least one modality');
      return;
    }

    setSubmitting(true);
    setError(null);
    setStatus('Uploading modalities…');

    const form = new FormData();
    form.append('task', task);
    if (cxr) form.append('cxr', cxr);
    if (notes.trim()) {
      form.append('notesJson', JSON.stringify([{ type: 'radiology', text: notes.trim() }]));
    }
    if (includeEhr) form.append('ehrJson', JSON.stringify(SAMPLE_EHR));

    try {
      const response = await api.post<{ data: { predictionId: string; stayId: string; status: string } }>(
        '/analyze',
        form,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      );
      const { predictionId } = response.data.data;
      setStatus('Prediction complete — opening report…');
      navigate(`/predictions/${predictionId}`);
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not analyze the submitted modalities'));
      setStatus('Analysis failed');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div>
        <p className="text-xs uppercase tracking-wide text-ink-muted">Demo workflow</p>
        <h1 className="mt-1 text-2xl font-semibold text-ink">Analyze multimodal evidence</h1>
        <p className="mt-1 max-w-2xl text-sm text-ink-secondary">
          Submit any combination of EHR, chest radiograph, and clinical notes. Missing modalities are handled by the same prediction and missingness pipeline used by the stay view.
        </p>
      </div>

      <Card title="1 · Prediction task" subtitle="Choose the report you want the fusion model to produce">
        <div className="grid gap-3 sm:grid-cols-2">
          {(['pneumonia', 'mortality'] as Task[]).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setTask(value)}
              aria-pressed={task === value}
              className={`rounded-lg border p-4 text-left transition ${task === value ? 'border-[var(--series-mortality)] bg-surface-3' : 'border-hairline hover:bg-surface-2'}`}
            >
              <div className="text-sm font-medium capitalize text-ink">{value}</div>
              <div className="mt-1 text-xs text-ink-muted">
                {value === 'pneumonia' ? 'EHR + CXR + radiology/discharge notes' : 'EHR + CXR + radiology/progress/nursing notes'}
              </div>
            </button>
          ))}
        </div>
      </Card>

      <Card title="2 · Submit modalities" subtitle="Every field is optional; the server validates the combination and enforces leakage control">
        <div className="grid gap-4 md:grid-cols-3">
          <label className="rounded-lg border border-hairline p-4">
            <span className="block text-sm font-medium text-ink">Chest X-ray</span>
            <span className="mt-1 block text-xs text-ink-muted">PNG or JPEG</span>
            <input
              aria-label="Chest X-ray"
              type="file"
              accept="image/png,image/jpeg"
              className="mt-3 w-full text-xs"
              onChange={(event) => setCxr(event.target.files?.[0] ?? null)}
            />
            {cxr && <span className="mt-2 block truncate text-xs text-ink-secondary">{cxr.name}</span>}
          </label>

          <label className="rounded-lg border border-hairline p-4">
            <span className="block text-sm font-medium text-ink">Clinical notes</span>
            <span className="mt-1 block text-xs text-ink-muted">Radiology note for the demo</span>
            <textarea
              aria-label="Clinical notes"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              rows={7}
              className="mt-3 w-full rounded-md border border-hairline bg-surface-2 p-2 text-sm text-ink"
              placeholder="Paste a clinical note…"
            />
          </label>

          <div className="rounded-lg border border-hairline p-4">
            <div className="text-sm font-medium text-ink">EHR</div>
            <div className="mt-1 text-xs text-ink-muted">Synthetic hourly vitals for the demo</div>
            <button
              type="button"
              role="switch"
              aria-checked={includeEhr}
              onClick={() => setIncludeEhr((value) => !value)}
              className={`mt-4 rounded-md px-3 py-2 text-xs ${includeEhr ? 'bg-[var(--series-mortality)] text-white' : 'bg-surface-3 text-ink-secondary'}`}
            >
              {includeEhr ? 'EHR included' : 'EHR omitted'}
            </button>
          </div>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-hairline pt-4">
          <Button variant="primary" disabled={submitting} onClick={() => void submit()}>
            {submitting ? 'Analyzing…' : 'Run analysis'}
          </Button>
          <span role="status" className="text-xs text-ink-muted">{status}</span>
          {error && <span role="alert" className="text-xs text-status-critical">{error}</span>}
        </div>
      </Card>

      <Card title="3 · Result" subtitle="The completed prediction opens in the existing prediction-report view">
        <p className="text-sm text-ink-secondary">
          The report automatically shows a reduced-input banner when EHR or another modality is absent, and renders the same F1–F6 evidence, confidence, missingness, contribution, and fused-decision sections.
        </p>
      </Card>
    </div>
  );
}
