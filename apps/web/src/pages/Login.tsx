import { useState, type FormEvent } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';

import { useAuth } from '../app/auth.jsx';
import { apiErrorMessage } from '../lib/api.js';
import { Button } from '../components/ui.jsx';
import { Icon } from '../components/icons.jsx';
import { LogoMark, Wordmark } from '../components/Logo.jsx';

/**
 * Sign-in. The lungs artwork is purely decorative (alt="") and sits behind a
 * radial glow; the heading block gets its own halo so the wordmark stays
 * readable over the artwork in both themes.
 *
 * The two artwork variants are swapped with Tailwind's `dark:` variant — the
 * one place it is still the right tool, because this is an asset swap, not a
 * colour. It covers both theme paths: lib/theme.ts mirrors a `.dark` class as
 * well as stamping `data-theme`.
 */
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
    <div
      className="relative grid min-h-screen place-items-center overflow-hidden px-6 py-10"
      style={{
        background:
          'radial-gradient(ellipse 45% 55% at 50% 50%, var(--glow) 0%, transparent 100%), var(--surface-2)',
      }}
    >
      <img
        src="/brand/lungs-light.png"
        alt=""
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-1/2 w-[1000px] max-w-none -translate-x-1/2 -translate-y-1/2 opacity-70 dark:hidden"
      />
      <img
        src="/brand/lungs-dark.png"
        alt=""
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-1/2 hidden w-[1000px] max-w-none -translate-x-1/2 -translate-y-1/2 opacity-45 dark:block"
      />

      <div className="relative flex w-full max-w-[440px] flex-col items-center gap-7">
        <div
          className="flex flex-col items-center gap-3.5 rounded-[28px] px-6 py-7 text-center min-[560px]:px-10"
          style={{
            background:
              'radial-gradient(ellipse 60% 62% at 50% 50%, var(--halo) 45%, transparent 100%)',
          }}
        >
          {/* Brand tile: fixed hexes, like LogoMark — the mark is brand, not theme. */}
          <span
            aria-hidden="true"
            className="grid h-[68px] w-[68px] place-items-center rounded-[18px] bg-[#2f4156] dark:bg-[#1e2b39]"
          >
            <LogoMark size={44} onDark />
          </span>
          <h1 className="text-[46px] font-extrabold leading-none tracking-[-0.035em] text-ink min-[560px]:text-[64px]">
            <Wordmark />
          </h1>
          <p className="text-[17px] font-medium text-ink min-[560px]:text-[19px]">
            See the reasoning behind every risk score.
          </p>
        </div>

        <form
          onSubmit={onSubmit}
          className="card flex w-full flex-col gap-[18px] rounded-[20px] p-7 min-[560px]:px-8"
        >
          <label className="flex flex-col gap-2 text-[13px] font-semibold text-ink" htmlFor="email">
            Email
            <input
              id="email"
              type="email"
              autoComplete="username"
              placeholder="name@hospital.org"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="field-input w-full text-sm font-normal"
            />
          </label>

          <label
            className="flex flex-col gap-2 text-[13px] font-semibold text-ink"
            htmlFor="password"
          >
            Password
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="field-input w-full text-sm font-normal"
            />
          </label>

          {error && (
            <p
              role="alert"
              className="note-critical flex items-start gap-2 px-3.5 py-3 text-[13px]"
            >
              <span aria-hidden="true">■</span>
              {error}
            </p>
          )}

          <Button type="submit" variant="primary" className="h-[50px] w-full" disabled={submitting}>
            {submitting ? 'Signing in…' : 'Sign in'}
          </Button>

          {/* TODO(phase-3): TOTP step at /login/mfa, mandatory for admin. */}

          <p className="flex items-center justify-center gap-2 text-[11px] text-ink-muted">
            <Icon name="shield" size={14} className="flex-none" />
            Research prototype · not a medical device
          </p>
        </form>
      </div>
    </div>
  );
}
