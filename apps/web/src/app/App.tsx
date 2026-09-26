import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

import { Disclaimer } from '../components/Disclaimer.js';

/**
 * Application shell. Routing, the auth guard and the page tree land in later
 * phases; this is the provider skeleton they mount into.
 */
export function App() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 30_000, retry: 1, refetchOnWindowFocus: false },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>
      <main className="min-h-screen bg-white p-8 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
        <h1 className="text-2xl font-semibold">PneumoVision</h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Multimodal clinical decision support — scaffold.
        </p>
        <Disclaimer className="mt-6 max-w-2xl" />
      </main>
    </QueryClientProvider>
  );
}
