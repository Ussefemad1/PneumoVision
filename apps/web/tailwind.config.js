/** @type {import('tailwindcss').Config} */
export default {
  // Class-based so the in-app toggle wins over the OS setting in both
  // directions; the root element carries data-theme, mirrored to .dark.
  darkMode: ['class', '[data-theme="dark"]'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          1: 'var(--surface-1)',
          2: 'var(--surface-2)',
          3: 'var(--surface-3)',
        },
        ink: {
          DEFAULT: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          muted: 'var(--text-muted)',
        },
        hairline: 'var(--border)',
        grid: 'var(--grid)',
        // Reserved status ramp — risk bands and alert severity only, never a
        // series colour, and never the sole signal (always icon + label).
        status: {
          good: 'var(--status-good)',
          warning: 'var(--status-warning)',
          serious: 'var(--status-serious)',
          critical: 'var(--status-critical)',
        },
        series: {
          mortality: 'var(--series-mortality)',
          pneumonia: 'var(--series-pneumonia)',
        },
      },
      borderColor: { DEFAULT: 'var(--border)' },
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
