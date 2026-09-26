import { useState, type FormEvent } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';

import { useAuth } from '../app/auth.jsx';
import { apiErrorMessage } from '../lib/api.js';
import { Button } from '../components/ui.jsx';
import { Disclaimer } from '../components/Disclaimer.jsx';

export function LoginPage() {
  const { user, isLoading, login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!isLoading && user) return <Navigate to="/" replace />;

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    login(email, password)
      .then(() => navigate('/', { replace: true }))
      .catch((err: unknown) => setError(apiErrorMessage(err, 'Sign-in failed')))
      .finally(() => setSubmitting(false));
  };

  return (
    <div className="grid min-h-screen place-items-center px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6 text-center">
          <div
            aria-hidden="true"
            className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-xl bg-[var(--series-mortality)] text-lg font-semibold text-white"
          >
            P
          </div>
          <h1 className="text-xl font-semibold text-ink">PneumoVision</h1>
          <p className="mt-1 text-sm text-ink-muted">Multimodal clinical decision support</p>
        </div>

        <form onSubmit={onSubmit} className="card space-y-4 p-5">
          <div>
            <label htmlFor="email" className="mb-1 block text-xs font-medium text-ink-secondary">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="username"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-hairline bg-surface-2 px-3 py-2 text-sm text-ink outline-none focus:ring-2 focus:ring-[var(--series-mortality)]"
              placeholder="clinician@pneumovision.local"
            />
          </div>

          <div>
            <label htmlFor="password" className="mb-1 block text-xs font-medium text-ink-secondary">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-hairline bg-surface-2 px-3 py-2 text-sm text-ink outline-none focus:ring-2 focus:ring-[var(--series-mortality)]"
            />
          </div>

          {error && (
            <p
              role="alert"
              className="flex items-start gap-1.5 rounded-md bg-status-critical/10 px-3 py-2 text-xs text-status-critical"
            >
              <span aria-hidden="true">■</span>
              {error}
            </p>
          )}

          <Button type="submit" variant="primary" className="w-full" disabled={submitting}>
            {submitting ? 'Signing in…' : 'Sign in'}
          </Button>

          {/* TODO(phase-3): TOTP step at /login/mfa, mandatory for admin. */}
          <p className="text-center text-[11px] text-ink-muted">
            Demo accounts are printed by <code>npm run gen:secrets -- --write</code>
          </p>
        </form>

        <Disclaimer className="mt-4" />
      </div>
    </div>
  );
}
