/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Colour-blind-safe risk palette (Okabe–Ito). Never the sole signal —
        // every risk band is also carried by an icon and a text label.
        risk: {
          low: '#0072B2',
          moderate: '#E69F00',
          high: '#D55E00',
          critical: '#CC79A7',
        },
      },
    },
  },
  plugins: [],
};
