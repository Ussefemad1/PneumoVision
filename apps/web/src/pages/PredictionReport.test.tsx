import type { Prediction } from '@pneumovision/shared';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import type * as ApiModule from '../lib/api.js';
import { PredictionReportPage } from './PredictionReport.js';

const { getData } = vi.hoisted(() => ({ getData: vi.fn() }));

vi.mock('../lib/api.js', async (importOriginal) => ({
  ...(await importOriginal<typeof ApiModule>()),
  getData,
}));

/** A synthetic pneumonia prediction as the fusion mode returns it. */
const prediction: Prediction = {
  id: '64b000000000000000000001',
  stayId: '64b000000000000000000002',
  task: 'pneumonia',
  cutoffTime: '2026-10-17T10:00:00.000Z',
  status: 'done',
  modelVersion: 'medpatch-r3:r3-fusion-001@a241114adabb,r3-fusion-002@294ad302c8ee',
  inputHash: 'x',
  latencyMs: 4170,
  requestedBy: null,
  inputsUsed: ['ehr', 'rr', 'dn'],
  excludedInputs: [],
  error: null,
  createdAt: '2026-10-17T10:00:05.000Z',
  result: {
    probability: 0.4123,
    unimodal: { ehr: 0.3, cxr: null, rr: 0.5, dn: 0.6 },
    joint: { high: 0.45, low: 0.35 },
    missingness: { vector: { ehr: true, cxr: false, rr: true, dn: true }, prediction: 0.2 },
    alphas: { high: 0.3, low: 0.2, miss: 0.1, ehr: 0.15, cxr: null, rr: 0.15, dn: 0.1 },
    confidence: {
      theta: 0.75,
      fractionAbove: { ehr: 0.5, cxr: null, rr: 0.6, dn: 0.7 },
      ehrTimesteps: null,
      cxrPatchGrid: null,
      noteSpans: null,
    },
    received: {
      ehr: { hours: 48, variablesPresent: ['Heart Rate'] },
      cxr: null,
      notes: [],
      mode: 'model',
    },
    sources: {
      fusion: 'trained',
      ehr: {
        kind: 'trained',
        model: 'EHR reader (LSTM), inside r3-fusion-002',
        testAuroc: null,
        confidence: null,
      },
      cxr: null,
      rr: { kind: 'trained', model: 'RR reader', testAuroc: null, confidence: null },
      dn: { kind: 'trained', model: 'DN reader', testAuroc: null, confidence: null },
    },
    fusionModel: {
      runId: 'r3-fusion-002',
      sha256: '294ad302c8ee',
      head: 'late',
      heads: { high: 0.45, low: 0.35, late: 0.4123, combined: 0.404, less_combined: 0.431 },
      bertModelName: 'dmis-lab/biobert-v1.1',
      theta: 0.75,
      modalities: ['EHR', 'CXR', 'RR', 'DN'],
      calibrated: false,
      notes: ['Probability is the raw sigmoid of the late head: an uncalibrated probability.'],
    },
  },
};

function renderReport() {
  getData.mockImplementation((url: string) =>
    url.startsWith('/predictions/') ? Promise.resolve(prediction) : new Promise(() => {}),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/predictions/${prediction.id}`]}>
        <Routes>
          <Route path="/predictions/:predictionId" element={<PredictionReportPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('PredictionReportPage in fusion mode', () => {
  it('names the trained MedPatch run and calls the probability uncalibrated', async () => {
    renderReport();
    expect(await screen.findByText(/MedPatch fusion \(Round 3, run r3-fusion-002\)/)).toBeVisible();
    expect(screen.getAllByText('Uncalibrated probability').length).toBeGreaterThan(0);
    expect(screen.queryByText('Calibrated')).toBeNull();
  });

  it('keeps every head and the run in a collapsible "Model details"', async () => {
    renderReport();
    const summary = await screen.findByText('Model details');
    fireEvent.click(summary);
    expect(screen.getByText(/Late fusion \(chosen on validation\)/)).toBeInTheDocument();
    expect(screen.getByText(/r3-fusion-002 · sha256 294ad302c8ee/)).toBeInTheDocument();
    expect(screen.getByText('dmis-lab/biobert-v1.1')).toBeInTheDocument();
    expect(screen.getByText('Mean of high and late')).toBeInTheDocument();
  });

  it('still labels the token-confidence maps as simulated', async () => {
    renderReport();
    await screen.findByText('Model details');
    expect(screen.getAllByText('Simulated').length).toBeGreaterThan(0);
  });
});
