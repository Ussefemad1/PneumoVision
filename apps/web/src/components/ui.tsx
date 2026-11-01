import type { ReactNode } from 'react';

import { cn } from '../lib/cn.js';

/** Card shell used by every panel, so radius/border/padding stay consistent. */
export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode | undefined;
  subtitle?: ReactNode | undefined;
  actions?: ReactNode | undefined;
  children: ReactNode;
  className?: string | undefined;
  bodyClassName?: string | undefined;
}) {
  return (
    <section className={cn('card', className)}>
      {(title ?? actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-hairline px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-muted">{subtitle}</p>}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </header>
      )}
      <div className={cn('p-4', bodyClassName)}>{children}</div>
    </section>
  );
}

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | undefined;
  size?: 'sm' | 'md' | undefined;
}) {
  return (
    <button
      {...props}
      className={cn(
        'inline-flex items-center justify-center gap-1.5 rounded-control font-medium transition',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2',
        'focus-visible:outline-accent',
        'disabled:cursor-not-allowed disabled:opacity-50',
        size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-2 text-sm',
        variant === 'primary' &&
          'bg-accent text-accent-on hover:brightness-110 active:brightness-95',
        variant === 'secondary' &&
          'border border-hairline bg-surface-1 text-ink hover:bg-surface-3',
        variant === 'ghost' && 'text-ink-secondary hover:bg-surface-3 hover:text-ink',
        variant === 'danger' &&
          'bg-status-critical text-white hover:brightness-110 active:brightness-95',
        className,
      )}
    />
  );
}

/** Loading placeholder. Mirrors the shape of what is coming, not a spinner. */
export function Skeleton({ className }: { className?: string | undefined }) {
  return <div className={cn('skeleton rounded-md', className)} aria-hidden="true" />;
}

export function SkeletonRows({
  rows = 5,
  className,
}: {
  rows?: number | undefined;
  className?: string | undefined;
}) {
  return (
    <div className={cn('space-y-2', className)} role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}

/**
 * Empty state. Always says *what* is missing and what would fill it — an
 * empty panel with no explanation is the most common way a demo looks broken.
 */
export function EmptyState({
  icon = '○',
  title,
  description,
  action,
}: {
  icon?: string | undefined;
  title: string;
  description?: string | undefined;
  action?: ReactNode | undefined;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
      <div aria-hidden="true" className="text-2xl text-ink-muted">
        {icon}
      </div>
      <p className="text-sm font-medium text-ink">{title}</p>
      {description && <p className="max-w-sm text-xs text-ink-muted">{description}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: (() => void) | undefined;
}) {
  return (
    <div role="alert" className="flex flex-col items-center gap-2 px-4 py-10 text-center">
      <div aria-hidden="true" className="text-2xl text-status-critical">
        ■
      </div>
      <p className="text-sm font-medium text-ink">Could not load this</p>
      <p className="max-w-sm text-xs text-ink-muted">{message}</p>
      {onRetry && (
        <Button size="sm" onClick={onRetry} className="mt-2">
          Try again
        </Button>
      )}
    </div>
  );
}

export function Chip({
  children,
  className,
  title,
}: {
  children: ReactNode;
  className?: string | undefined;
  title?: string | undefined;
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
        className,
      )}
    >
      {children}
    </span>
  );
}

/** Label/value pair used across the report and stay headers. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</dt>
      <dd className="mt-0.5 truncate text-sm text-ink">{children}</dd>
    </div>
  );
}

/**
 * Single headline number. A plain <div>, not a <section>: only Card renders a
 * <section>, because the e2e suite scopes panels by section + heading and a
 * second one would widen or duplicate that match.
 */
export function Kpi({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode | undefined;
}) {
  return (
    <div className="card flex flex-col gap-2.5 p-5">
      <div className="eyebrow">{label}</div>
      <div className="tnum text-[32px] font-bold leading-none tracking-[-0.02em] text-ink">
        {value}
      </div>
      {hint && <div className="text-xs text-ink-secondary">{hint}</div>}
    </div>
  );
}

/** Segmented single-choice control. Real radios, so arrow keys and SRs work. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
  className,
}: {
  value: T;
  onChange: (value: T) => void;
  options: readonly { value: T; label: string }[];
  ariaLabel: string;
  className?: string | undefined;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn('inline-flex flex-wrap gap-1 rounded-control bg-surface-3 p-1', className)}
    >
      {options.map((option) => (
        <button
          key={option.value}
          role="radio"
          aria-checked={value === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            'h-[34px] rounded-lg px-3.5 text-[13px] transition',
            value === option.value
              ? 'bg-surface-1 font-semibold text-ink shadow-sm'
              : 'font-medium text-ink-secondary hover:text-ink',
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
