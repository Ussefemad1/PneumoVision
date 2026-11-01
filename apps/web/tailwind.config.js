/** @type {import('tailwindcss').Config} */
export default {
  // Class-based so the in-app toggle wins over the OS setting in both
  // directions; the root element carries data-theme, mirrored to .dark.
  darkMode: ['class', '[data-theme="dark"]'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // Every colour here is a bare var(). Tailwind resolves alpha modifiers
      // at build time with parseColor(), which cannot parse a var() — so
      // `bg-surface-1/90` or `ring-status-good/40` emits NOTHING AT ALL
      // rather than a translucent colour. Never put a /alpha on these. Where a
      // tint is needed, add a token (see --chip-*-bg, --softer in index.css).
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
        axis: 'var(--axis)',
        // UI chrome. Keeps the brand off the CVD-validated series slots.
        accent: {
          DEFAULT: 'var(--accent)',
          ink: 'var(--accent-ink)',
          on: 'var(--on-accent)',
        },
        soft: 'var(--soft)',
        softer: 'var(--softer)',
        sidebar: {
          DEFAULT: 'var(--sidebar)',
          ink: 'var(--sidebar-ink)',
          muted: 'var(--sidebar-muted)',
          active: 'var(--sidebar-active)',
        },
        // Reserved status ramp — risk bands and alert severity only, never a
        // series colour, and never the sole signal (always icon + label).
        status: {
          good: 'var(--status-good)',
          warning: 'var(--status-warning)',
          serious: 'var(--status-serious)',
          critical: 'var(--status-critical)',
          // Text-safe companions; the ramp above is sub-3:1 on light and must
          // never colour type. -> text-status-ink-warning, etc.
          ink: {
            good: 'var(--status-good-ink)',
            warning: 'var(--status-warning-ink)',
            serious: 'var(--status-serious-ink)',
            critical: 'var(--status-critical-ink)',
          },
        },
        series: {
          mortality: 'var(--series-mortality)',
          pneumonia: 'var(--series-pneumonia)',
        },
        seq: {
          100: 'var(--seq-100)',
          250: 'var(--seq-250)',
          400: 'var(--seq-400)',
          550: 'var(--seq-550)',
          700: 'var(--seq-700)',
        },
      },
      borderColor: { DEFAULT: 'var(--border)' },
      // Added as new keys rather than remapping md/lg/xl: remapping would
      // silently reshape every existing rounded-lg well and badge, where 16px
      // is far too round, and the diff would be invisible in review.
      borderRadius: {
        card: 'var(--radius-card)', // 16px — card-level surfaces
        control: 'var(--radius-control)', // 10px — buttons, inputs, pills
      },
      // A per-theme shadow is not expressible here (the config is static), so
      // it comes from a variable that dark mode redefines. That value is a
      // transparent shadow rather than `none`, because `none` inside the comma
      // list Tailwind builds would invalidate the declaration and any ring.
      boxShadow: { card: 'var(--shadow-card)' },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
