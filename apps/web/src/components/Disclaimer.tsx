import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { DISCLAIMER } from '@pneumovision/shared';

import { cn } from '../lib/cn.js';

/**
 * P7: shown on every surface that displays a prediction. Text comes from the
 * shared package so it cannot drift between pages or exported reports.
 */

/**
 * Lets the app shell know a page is already showing the full disclaimer, so
 * its own standing footer line can stand down and the two do not stack.
 *
 * The context is optional on purpose: Disclaimer is rendered bare in its own
 * unit test, with no provider, and must not throw there.
 */
const DisclaimerRegistry = createContext<{
  register: () => void;
  unregister: () => void;
} | null>(null);

export function DisclaimerScope({
  children,
}: {
  children: (hasPageDisclaimer: boolean) => ReactNode;
}) {
  const [count, setCount] = useState(0);
  const [value] = useState(() => ({
    register: () => setCount((n) => n + 1),
    unregister: () => setCount((n) => Math.max(0, n - 1)),
  }));

  return (
    <DisclaimerRegistry.Provider value={value}>{children(count > 0)}</DisclaimerRegistry.Provider>
  );
}

export function Disclaimer({ className }: { className?: string | undefined }) {
  const registry = useContext(DisclaimerRegistry);

  useEffect(() => {
    if (!registry) return;
    registry.register();
    return registry.unregister;
  }, [registry]);

  return (
    <p role="note" className={cn('note-warning px-3 py-2 text-xs', className)}>
      {DISCLAIMER}
    </p>
  );
}
