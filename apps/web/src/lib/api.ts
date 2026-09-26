import axios, { AxiosError } from 'axios';

/**
 * API client.
 *
 * `withCredentials` matters: auth is an httpOnly cookie, so every request has
 * to carry it. In the demo the Vite dev server proxies /api to the gateway, so
 * a relative baseURL works in both the demo and the Docker stack.
 */
export const api = axios.create({
  baseURL: '/api/v1',
  withCredentials: true,
  timeout: 130_000,
});

interface ErrorBody {
  error?: { code?: string; message?: string; requestId?: string };
}

/** Pulls the API's error envelope out of an axios failure. */
export function apiErrorMessage(err: unknown, fallback = 'Something went wrong'): string {
  if (err instanceof AxiosError) {
    const body = err.response?.data as ErrorBody | undefined;
    if (body?.error?.message) return body.error.message;
    if (err.code === 'ECONNABORTED') return 'The request timed out.';
    if (!err.response) return 'Cannot reach the API. Is it running?';
  }
  return fallback;
}

export function isUnauthorized(err: unknown): boolean {
  return err instanceof AxiosError && err.response?.status === 401;
}

/** Unwraps `{ data }` so callers work with the payload directly. */
export async function getData<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const res = await api.get<{ data: T }>(url, params ? { params } : undefined);
  return res.data.data;
}

export async function postData<T>(url: string, body?: unknown): Promise<T> {
  const res = await api.post<{ data: T }>(url, body ?? {});
  return res.data.data;
}

export async function patchData<T>(url: string, body?: unknown): Promise<T> {
  const res = await api.patch<{ data: T }>(url, body ?? {});
  return res.data.data;
}
