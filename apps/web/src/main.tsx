// Self-hosted Inter (variable, 100-900). Bundled rather than fetched from
// Google Fonts so type is correct with no network — the demo may be presented
// offline. Registers the family as 'Inter Variable'.
import '@fontsource-variable/inter';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './app/App.js';
import { initTheme } from './lib/theme.js';
import './styles/index.css';

// Applied before the first paint so there is no flash of the wrong theme.
initTheme();

const container = document.getElementById('root');
if (!container) throw new Error('#root not found');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
