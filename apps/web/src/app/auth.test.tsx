import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { AxiosError, AxiosHeaders } from 'axios';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as ApiModule from '../lib/api.js';
import { LoginPage } from '../pages/Login.js';
import { RequireAuth } from './App.js';
import { AuthProvider } from './auth.js';
import { Layout } from './Layout.js';

const { getData, postData } = vi.hoisted(() => ({ getData: vi.fn(), postData: vi.fn() }));

vi.mock('../lib/api.js', async (importOriginal) => ({
  ...(await importOriginal<typeof ApiModule>()),
  getData,
  postData,
}));

vi.mock('../lib/socket.js', () => ({ disconnectSocket: vi.fn() }));

const USER = { id: 'u1', email: 'c@pv.local', name: 'Dr Demo', role: 'clinician' };

const unauthorized = () =>
  new AxiosError('401', 'ERR_BAD_REQUEST', undefined, undefined, {
    status: 401,
    statusText: 'Unauthorized',
    headers: {},
    config: { headers: new AxiosHeaders() },
    data: { error: { code: 'UNAUTHENTICATED', message: 'Not signed in' } },
  });

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/']}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route element={<Layout />}>
                <Route index element={<p>ward page</p>} />
              </Route>
            </Route>
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('sign out', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.matchMedia = vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
    let signedIn = true;
    getData.mockImplementation((url: string) => {
      if (url === '/auth/me') return signedIn ? Promise.resolve(USER) : Promise.reject(unauthorized());
      return signedIn ? Promise.resolve([]) : Promise.reject(unauthorized());
    });
    postData.mockImplementation((url: string) => {
      if (url === '/auth/logout') signedIn = false;
      return Promise.resolve({ ok: true });
    });
  });

  it('lands on the login page without a refresh', async () => {
    renderApp();
    expect(await screen.findByText('ward page')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /sign out/i }));

    expect(await screen.findByRole('button', { name: /^sign in$/i })).toBeInTheDocument();
    expect(screen.queryByText('ward page')).not.toBeInTheDocument();
  });

  it('still signs out locally when the logout request fails', async () => {
    postData.mockRejectedValueOnce(new Error('network down'));
    renderApp();
    await screen.findByText('ward page');

    fireEvent.click(screen.getByRole('button', { name: /sign out/i }));

    expect(await screen.findByRole('button', { name: /^sign in$/i })).toBeInTheDocument();
  });
});
