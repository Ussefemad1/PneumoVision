import { DISCLAIMER } from '@pneumovision/shared';

import { cn } from '../lib/cn.js';

/**
 * P7: shown on every surface that displays a prediction. Text comes from the
 * shared package so it cannot drift between pages or exported reports.
 */
export function Disclaimer({ className }: { className?: string }) {
  return (
    <p
      role="note"
      className={cn(
        'rounded-md border border-amber-500/40 bg-amber-50 px-3 py-2 text-xs',
        'text-amber-900 dark:bg-amber-950/40 dark:text-amber-200',
        className,
      )}
    >
      {DISCLAIMER}
    </p>
  );
}
