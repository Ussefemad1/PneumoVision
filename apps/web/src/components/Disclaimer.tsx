import { DISCLAIMER } from '@pneumovision/shared';

import { cn } from '../lib/cn.js';

/**
 * P7: shown on every surface that displays a prediction. Text comes from the
 * shared package so it cannot drift between pages or exported reports.
 */
export function Disclaimer({ className }: { className?: string | undefined }) {
  return (
    <p role="note" className={cn('note-warning px-3 py-2 text-xs', className)}>
      {DISCLAIMER}
    </p>
  );
}
