import { useEffect, useState } from 'react';

/**
 * Theme handling.
 *
 * Three states: an explicit 'light' or 'dark' choice stamps `data-theme` on
 * the root element and wins over the OS; 'system' removes the stamp and lets
 * `prefers-color-scheme` decide. The choice is a per-browser convenience, so
 * localStorage is the right home for it — and every access is guarded,
 * because a private window or blocked site data makes it throw.
 */
export type ThemeChoice = 'light' | 'dark' | 'system';

const STORAGE_KEY = 'pv.theme';

function readStored(): ThemeChoice {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (value === 'light' || value === 'dark' || value === 'system') return value;
  } catch {
    /* storage unavailable — fall through to the default */
  }
  return 'system';
}

function apply(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', choice);
  // Tailwind's class strategy needs `.dark` to mirror the resolved theme.
  const dark =
    choice === 'dark' ||
    (choice === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  root.classList.toggle('dark', dark);
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readStored);

  useEffect(() => {
    apply(choice);
    try {
      localStorage.setItem(STORAGE_KEY, choice);
    } catch {
      /* non-fatal: the theme still applies for this session */
    }
  }, [choice]);

  // Follow the OS while the user has not made an explicit choice.
  useEffect(() => {
    if (choice !== 'system') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => apply('system');
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [choice]);

  return { choice, setChoice };
}

/** Applied before React mounts, so there is no flash of the wrong theme. */
export function initTheme(): void {
  apply(readStored());
}
