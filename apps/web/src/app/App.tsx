import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { Navigate, Outlet, Route, BrowserRouter, Routes } from 'react-router-dom';

import { AlertsPage } from '../pages/Alerts.jsx';
import { DashboardPage } from '../pages/Dashboard.jsx';
import { LoginPage } from '../pages/Login.jsx';
import { PredictionReportPage } from '../pages/PredictionReport.jsx';
import { ReplayPage } from '../pages/Replay.jsx';
import { StayPage } from '../pages/Stay.jsx';
import { Skeleton } from '../components/ui.jsx';
import { AuthProvider, useAuth } from './auth.jsx';
import { Layout } from './Layout.jsx';

export function App() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 15_000, retry: 1, refetchOnWindowFocus: false },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route element={<Layout />}>
                <Route index element={<DashboardPage />} />
                <Route path="stays/:stayId" element={<StayPage />} />
                <Route path="predictions/:predictionId" element={<PredictionReportPage />} />
                <Route path="alerts" element={<AlertsPage />} />
                <Route path="replay" element={<ReplayPage />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

/** Auth guard: unauthenticated users never reach a patient route. */
function RequireAuth() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="mx-auto max-w-[1400px] space-y-4 p-6">
        <Skeleton className="h-14 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  return user ? <Outlet /> : <Navigate to="/login" replace />;
}
