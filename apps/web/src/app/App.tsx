import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { Navigate, Outlet, Route, BrowserRouter, Routes, useSearchParams } from 'react-router-dom';

import { AlertsPage } from '../pages/Alerts.jsx';
import { AnalysesPage } from '../pages/Analyses.jsx';
import { AnalyzePage } from '../pages/Analyze.jsx';
import { DashboardPage } from '../pages/Dashboard.jsx';
import { ModelsPage } from '../pages/Models.jsx';
import { PatientsPage } from '../pages/Patients.jsx';
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
                <Route path="/analyze" element={<AnalyzeIndex />} />
                <Route path="/analyze/new" element={<AnalyzePage />} />
                <Route path="/analyses" element={<Navigate to="/analyze" replace />} />
                <Route path="/stays/:stayId" element={<StayPage />} />
                <Route path="/predictions/:predictionId" element={<PredictionReportPage />} />
                <Route path="/patients" element={<PatientsPage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/replay" element={<ReplayPage />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

/**
 * `/analyze` is the history list; the form lives at `/analyze/new`. Links of
 * the form `/analyze?stayId=…` predate the split and mean "open the form for
 * this stay", so they are forwarded there.
 */
function AnalyzeIndex() {
  const [params] = useSearchParams();
  return params.has('stayId') ? (
    <Navigate to={`/analyze/new?${params.toString()}`} replace />
  ) : (
    <AnalysesPage />
  );
}

export function RequireAuth() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      // Mirrors Layout's shell so the content does not shift once it resolves.
      <div className="flex min-h-screen flex-col min-[900px]:flex-row">
        <div
          aria-hidden="true"
          className="h-16 bg-sidebar min-[900px]:h-screen min-[900px]:w-[248px] min-[900px]:flex-none"
        />
        <div className="flex w-full min-w-0 max-w-[1440px] flex-1 flex-col gap-[22px] px-4 py-6 min-[900px]:px-10 min-[900px]:py-7">
          <Skeleton className="h-14 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  return user ? <Outlet /> : <Navigate to="/login" replace />;
}
