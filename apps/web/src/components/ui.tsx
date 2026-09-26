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
        'inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2',
        'focus-visible:outline-[var(--series-mortality)]',
        'disabled:cursor-not-allowed disabled:opacity-50',
        size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-2 text-sm',
        variant === 'primary' &&
          'bg-[var(--series-mortality)] text-white hover:brightness-110 active:brightness-95',
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
