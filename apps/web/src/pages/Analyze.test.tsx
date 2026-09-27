import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { AxiosError, AxiosHeaders } from 'axios';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as ApiModule from '../lib/api.js';
import { readText } from '../lib/files.js';
import { AnalyzePage } from './Analyze.js';

const { post } = vi.hoisted(() => ({ post: vi.fn() }));

vi.mock('../lib/api.js', async (importOriginal) => ({
  ...(await importOriginal<typeof ApiModule>()),
  api: { post },
  getData: vi.fn(),
}));

vi.mock('../lib/socket.js', () => ({
  getSocket: () => ({ on: vi.fn(), off: vi.fn() }),
}));

function renderPage(path = '/analyze') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/predictions/:id" element={<p>report page</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const runButton = () => screen.getByRole('button', { name: /run analysis/i });

function addNote(text: string, type?: string) {
  fireEvent.click(screen.getByRole('button', { name: /add note/i }));
  const index = screen.getAllByLabelText(/^Note \d+ text$/).length;
  fireEvent.change(screen.getByLabelText(`Note ${index} text`), { target: { value: text } });
  if (type) {
    fireEvent.change(screen.getByLabelText(`Note ${index} type`), { target: { value: type } });
  }
  return index;
}

beforeEach(() => {
  post.mockReset();
});

describe('AnalyzePage', () => {
  it('disables submit until at least one modality is provided', () => {
    renderPage();
    expect(runButton()).toBeDisabled();

    addNote('Synthetic radiology note: right basal opacity.');
    expect(runButton()).toBeEnabled();
  });

  it('greys out discharge notes for mortality and does not count them as a modality', () => {
    renderPage();
    fireEvent.click(screen.getByRole('radio', { name: /mortality/i }));
    const index = addNote('Synthetic discharge summary.', 'discharge');

    const text = screen.getByLabelText(`Note ${index} text`);
    expect(text).toBeDisabled();
    expect(
      screen.getByText(/excluded to prevent outcome leakage — discharge notes/i),
    ).toBeVisible();
    expect(runButton()).toBeDisabled();

    fireEvent.click(screen.getByRole('radio', { name: /pneumonia/i }));
    expect(screen.getByLabelText(`Note ${index} text`)).toBeEnabled();
    expect(runButton()).toBeEnabled();
  });

  it('keeps every input after a failed submit and shows the API error code', async () => {
    post.mockRejectedValueOnce(
      new AxiosError('bad', 'ERR_BAD_REQUEST', undefined, undefined, {
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: { headers: new AxiosHeaders() },
        data: {
          error: { code: 'EHR_INVALID_VALUE', message: 'Heart Rate is wrong', requestId: 'r1' },
        },
      }),
    );
    renderPage();
    fireEvent.click(screen.getByRole('radio', { name: /mortality/i }));
    addNote('Synthetic nursing note: comfortable overnight.', 'nursing');

    fireEvent.click(runButton());

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('EHR_INVALID_VALUE');
    expect(alert).toHaveTextContent('Heart Rate is wrong');
    expect(screen.getByLabelText('Note 1 text')).toHaveValue(
      'Synthetic nursing note: comfortable overnight.',
    );
    expect(screen.getByLabelText('Note 1 type')).toHaveValue('nursing');
    expect(screen.getByRole('radio', { name: /mortality/i })).toBeChecked();
    expect(runButton()).toBeEnabled();
  });

  it('submits the chosen task and notes, then opens the report', async () => {
    post.mockResolvedValueOnce({
      data: {
        data: {
          predictionId: 'a'.repeat(24),
          stayId: 'b'.repeat(24),
          stayStatus: 'adhoc',
          status: 'done',
          excluded: [],
        },
      },
    });
    renderPage();
    addNote('Synthetic radiology note.', 'radiology');
    fireEvent.click(runButton());

    expect(await screen.findByText('report page')).toBeInTheDocument();
    const form = post.mock.calls[0]![1] as FormData;
    expect(form.get('task')).toBe('pneumonia');
    expect(JSON.parse(form.get('noteTypes') as string)).toEqual(['radiology']);
    const files = form.getAll('notes') as File[];
    expect(files).toHaveLength(1);
    expect(await readText(files[0]!)).toBe('Synthetic radiology note.');
  });

  it('previews an EHR CSV with row-level errors and blocks submit', async () => {
    renderPage();
    fireEvent.click(screen.getByRole('tab', { name: /csv upload/i }));
    const file = new File(['hour,Heart Rate\n-1,90\n0,abc\n'], 'ehr.csv', { type: 'text/csv' });
    fireEvent.change(screen.getByLabelText('EHR CSV file'), { target: { files: [file] } });

    const preview = await screen.findByRole('table', { name: /csv preview/i });
    const errorRow = within(preview).getByText(/"abc" is not a number/);
    expect(errorRow).toBeInTheDocument();
    expect(within(preview).getByText('Line 3')).toBeInTheDocument();
    await waitFor(() => expect(runButton()).toBeDisabled());
  });

  it('loads a synthetic EHR preset into the grid', () => {
    renderPage();
    fireEvent.click(screen.getByRole('tab', { name: /synthetic example/i }));
    fireEvent.click(screen.getByRole('button', { name: /deteriorating/i }));

    const grid = screen.getByRole('table', { name: /ehr grid/i });
    expect(within(grid).getAllByRole('row').length).toBeGreaterThan(2);
    expect(runButton()).toBeEnabled();
  });
});
