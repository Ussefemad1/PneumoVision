import { useEffect, useRef, useState } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';

import { getData } from '../lib/api.js';
import { cn } from '../lib/cn.js';
import { useTheme, type ThemeChoice } from '../lib/theme.js';
import { Icon, type IconName } from '../components/icons.jsx';
import { LogoMark, Wordmark } from '../components/Logo.jsx';
import { useAuth } from './auth.jsx';

/**
 * App shell: a navy sidebar beside the content column.
 *
 * Below 900px the sidebar becomes a wrapping top bar rather than a drawer —
 * there is no hamburger and nothing is hidden behind a toggle, so every
 * destination stays one tap away and nothing a test asserts on can be
 * responsively hidden.
 */

const NAV: { to: string; label: string; icon: IconName; end?: boolean | undefined }[] = [
  { to: '/', label: 'Overview', icon: 'grid', end: true },
  { to: '/analyze', label: 'Analyses', icon: 'history' },
  { to: '/alerts', label: 'Alerts', icon: 'bell' },
];

/** Up to two initials, for the avatar. Falls back to a dash, never empty. */
function initials(name: string | undefined): string {
  const parts = (name ?? '').trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return '—';
  return parts
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? '')
    .join('');
}

export function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const { data: openAlerts } = useQuery({
    queryKey: ['alerts', { status: 'open' }],
    queryFn: () => getData<unknown[]>('/alerts', { status: 'open' }),
    refetchInterval: 30_000,
  });
  const alertCount = openAlerts?.length ?? 0;

  const navClass = ({ isActive }: { isActive: boolean }) =>
    cn(
      'flex h-[42px] items-center gap-3 rounded-control px-3 text-sm transition',
      isActive
        ? 'bg-sidebar-active font-semibold text-sidebar-ink shadow-[inset_3px_0_0_var(--soft)]'
        : 'font-medium text-sidebar-muted hover:bg-sidebar-active hover:text-sidebar-ink',
    );

  return (
    <div className="flex min-h-screen flex-col min-[900px]:flex-row">
      <nav
        aria-label="Main"
        className={cn(
          'flex flex-wrap items-center gap-2 border-b border-white/[0.08] bg-sidebar p-3',
          'min-[900px]:sticky min-[900px]:top-0 min-[900px]:h-screen min-[900px]:w-[248px]',
          'min-[900px]:border-b-0 min-[900px]:border-r min-[900px]:border-r-white/[0.08]',
          'min-[900px]:flex-none min-[900px]:flex-col min-[900px]:flex-nowrap',
          'min-[900px]:items-stretch min-[900px]:gap-0 min-[900px]:overflow-y-auto min-[900px]:px-4 min-[900px]:py-5',
        )}
      >
        <button
          onClick={() => void navigate('/')}
          className="flex items-center gap-2.5 rounded-control px-1 text-left min-[900px]:px-2 min-[900px]:pb-5 min-[900px]:pt-1"
        >
          <span
            aria-hidden="true"
            className="grid h-10 w-10 flex-none place-items-center rounded-[11px] bg-white/[0.08]"
          >
            <LogoMark size={30} onDark />
          </span>
          <span className="min-w-0">
            <span className="block text-[17px] font-extrabold tracking-[-0.02em] text-sidebar-ink">
              <Wordmark tone="muted" />
            </span>
            <span className="block text-[11px] text-sidebar-muted">Clinical decision support</span>
          </span>
        </button>

        <NavLink
          to="/analyze/new"
          className="flex h-11 items-center justify-center gap-2 rounded-control bg-cta px-4 text-sm font-semibold text-cta-ink transition hover:brightness-105 min-[900px]:mb-5 min-[900px]:w-full"
        >
          <Icon name="plus" size={18} strokeWidth={2} />
          New analysis
        </NavLink>

        <div className="eyebrow hidden px-3 pb-2 text-sidebar-muted min-[900px]:block">
          Workspace
        </div>

        <div className="flex flex-wrap gap-0.5 min-[900px]:flex-col">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end ?? false} className={navClass}>
              <Icon name={item.icon} />
              <span className="min-[900px]:flex-1">{item.label}</span>
              {item.to === '/alerts' && alertCount > 0 && (
                <span className="tnum flex h-5 min-w-[22px] items-center justify-center rounded-full bg-status-critical px-1.5 text-[11px] font-bold text-white">
                  {alertCount}
                </span>
              )}
            </NavLink>
          ))}
        </div>

        <div
          aria-hidden="true"
          className="hidden min-[900px]:block min-[900px]:min-h-6 min-[900px]:flex-1"
        />
        <div
          aria-hidden="true"
          className="hidden h-px bg-white/[0.12] min-[900px]:mx-1 min-[900px]:mb-3 min-[900px]:mt-2 min-[900px]:block"
        />

        <div className="flex min-[900px]:flex-col">
          <NavLink to="/replay" className={navClass}>
            <Icon name="play" />
            <span className="min-[900px]:flex-1">ICU Replay</span>
            <span className="rounded-md bg-sidebar-active px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.06em] text-sidebar-ink">
              Demo
            </span>
          </NavLink>
        </div>

        <UserCard
          name={user?.name}
          role={user?.role}
          onSignOut={() => void logout()}
          className="ml-auto min-[900px]:ml-0 min-[900px]:mt-3"
        />
      </nav>

      <main className="flex w-full min-w-0 max-w-[1440px] flex-1 flex-col gap-[22px] px-4 py-6 min-[900px]:px-10 min-[900px]:py-7">
        <div className="min-w-0 flex-1">
          <Outlet />
        </div>
        {/*
         * Deliberately NOT role="note": the Disclaimer component owns that
         * role, and a bare getByRole('note') must stay unambiguous.
         */}
        <div className="note-info flex items-center gap-2.5 px-4 py-3 text-xs">
          <Icon name="info" size={16} className="flex-none" />
          Research prototype · decision support only · not for clinical use · synthetic data
        </div>
      </main>
    </div>
  );
}

/**
 * User card plus the theme popover.
 *
 * Sign out is its own always-visible icon button, never inside the popover:
 * app/auth.test.tsx clicks it directly without opening any menu.
 */
function UserCard({
  name,
  role,
  onSignOut,
  className,
}: {
  name: string | undefined;
  role: string | undefined;
  onSignOut: () => void;
  className?: string | undefined;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  // Plain document listeners: no ResizeObserver, IntersectionObserver,
  // <dialog> or the Popover API, none of which exist in jsdom — and
  // auth.test.tsx mounts this whole shell.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div ref={wrap} className={cn('relative', className)}>
      {open && (
        <div className="card absolute right-0 top-full z-20 mt-2 w-60 p-3 min-[900px]:bottom-full min-[900px]:left-0 min-[900px]:top-auto min-[900px]:mb-2 min-[900px]:w-auto">
          <ThemeChoices />
        </div>
      )}
      <div className="flex items-center gap-1.5 rounded-xl border border-white/[0.14] p-1.5">
        <button
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-1 py-0.5 text-left transition hover:bg-sidebar-active"
        >
          <span
            aria-hidden="true"
            className="grid h-[34px] w-[34px] flex-none place-items-center rounded-full bg-sidebar-active text-[13px] font-semibold text-sidebar-ink"
          >
            {initials(name)}
          </span>
          <span className="hidden min-w-0 min-[900px]:block">
            <span className="block truncate text-[13px] font-semibold text-sidebar-ink">
              {name}
            </span>
            <span className="block truncate text-[11px] capitalize text-sidebar-muted">{role}</span>
          </span>
          <Icon
            name="up"
            size={16}
            className="hidden flex-none text-sidebar-muted min-[900px]:block"
          />
        </button>
        {/* No navigate here: RequireAuth redirects once the session is cleared. */}
        <button
          aria-label="Sign out"
          onClick={onSignOut}
          className="grid h-9 w-9 flex-none place-items-center rounded-lg text-sidebar-muted transition hover:bg-sidebar-active hover:text-sidebar-ink"
        >
          <Icon name="out" />
        </button>
      </div>
    </div>
  );
}

function ThemeChoices() {
  const { choice, setChoice } = useTheme();
  const options: { value: ThemeChoice; label: string }[] = [
    { value: 'light', label: 'Light' },
    { value: 'dark', label: 'Dark' },
    { value: 'system', label: 'System' },
  ];
  return (
    <>
      <div className="eyebrow mb-2">Theme</div>
      <div
        role="radiogroup"
        aria-label="Colour theme"
        className="flex gap-1 rounded-control bg-surface-3 p-1"
      >
        {options.map((option) => (
          <button
            key={option.value}
            role="radio"
            aria-checked={choice === option.value}
            onClick={() => setChoice(option.value)}
            className={cn(
              'h-8 flex-1 rounded-lg px-2.5 text-xs transition',
              choice === option.value
                ? 'bg-surface-1 font-semibold text-ink shadow-sm'
                : 'font-medium text-ink-secondary hover:text-ink',
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </>
  );
}
