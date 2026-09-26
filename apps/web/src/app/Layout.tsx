import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';

import { getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { useTheme, type ThemeChoice } from '../lib/theme.js';
import { Button } from '../components/ui.jsx';
import { useAuth } from './auth.jsx';

const NAV: { to: string; label: string; end?: boolean | undefined }[] = [
  { to: '/', label: 'Ward', end: true },
  { to: '/alerts', label: 'Alerts' },
  { to: '/replay', label: 'ICU Replay' },
];

export function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  // Badge the Alerts tab so an unacknowledged alert is visible from any page.
  const { data: openAlerts } = useQuery({
    queryKey: ['alerts', { status: 'open' }],
    queryFn: () => getData<unknown[]>('/alerts', { status: 'open' }),
    refetchInterval: 30_000,
  });

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-hairline bg-surface-1/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-4">
          <button
            onClick={() => void navigate('/')}
            className="flex items-center gap-2 text-sm font-semibold text-ink"
          >
            <span
              aria-hidden="true"
              className="grid h-6 w-6 place-items-center rounded bg-[var(--series-mortality)] text-[13px] text-white"
            >
              P
            </span>
            PneumoVision
          </button>

          <nav className="flex items-center gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end ?? false}
                className={({ isActive }) =>
                  cn(
                    'rounded-md px-3 py-1.5 text-sm transition',
                    isActive
                      ? 'bg-surface-3 font-medium text-ink'
                      : 'text-ink-secondary hover:bg-surface-3 hover:text-ink',
                  )
                }
              >
                {item.label}
                {item.to === '/alerts' && (openAlerts?.length ?? 0) > 0 && (
                  <span className="tnum ml-1.5 rounded-full bg-status-critical px-1.5 py-0.5 text-[10px] font-semibold text-white">
                    {openAlerts?.length}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <ThemeToggle />
            <div className="hidden text-right sm:block">
              <div className="text-xs font-medium text-ink">{user?.name}</div>
              <div className="text-[11px] capitalize text-ink-muted">{user?.role}</div>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                void logout().then(() => navigate('/login'));
              }}
            >
              Sign out
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-hairline px-4 py-3 text-center text-[11px] text-ink-muted">
        Research prototype · decision support only · not for clinical use · synthetic data
      </footer>
    </div>
  );
}

function ThemeToggle() {
  const { choice, setChoice } = useTheme();
  const options: { value: ThemeChoice; icon: string; label: string }[] = [
    { value: 'light', icon: '☀', label: 'Light' },
    { value: 'dark', icon: '☾', label: 'Dark' },
    { value: 'system', icon: '◐', label: 'System' },
  ];

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="flex items-center rounded-md border border-hairline p-0.5"
    >
      {options.map((option) => (
        <button
          key={option.value}
          role="radio"
          aria-checked={choice === option.value}
          title={option.label}
          onClick={() => setChoice(option.value)}
          className={cn(
            'rounded px-1.5 py-0.5 text-xs transition',
            choice === option.value
              ? 'bg-surface-3 text-ink'
              : 'text-ink-muted hover:text-ink-secondary',
          )}
        >
          <span aria-hidden="true">{option.icon}</span>
          <span className="sr-only">{option.label}</span>
        </button>
      ))}
    </div>
  );
}
