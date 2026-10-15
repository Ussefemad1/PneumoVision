import type { CurrentUser } from '@pneumovision/shared';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { createContext, useContext, type ReactNode } from 'react';

import { getData, isUnauthorized, postData } from '../lib/api.js';
import { disconnectSocket } from '../lib/socket.js';

interface AuthState {
  user: CurrentUser | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<CurrentUser>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ['auth', 'me'],
    queryFn: () => getData<CurrentUser>('/auth/me'),
    // A 401 here is the normal signed-out state, not a failure to retry.
    retry: (count, error) => !isUnauthorized(error) && count < 1,
    staleTime: 5 * 60_000,
  });

  const loginMutation = useMutation({
    mutationFn: (vars: { email: string; password: string }) =>
      postData<CurrentUser>('/auth/login', vars),
    onSuccess: (user) => queryClient.setQueryData(['auth', 'me'], user),
  });

  const logout = async () => {
    await postData('/auth/logout').catch(() => {
      // Signed out locally even if the request failed: the user asked to leave.
    });
    disconnectSocket();
    // Mark the session gone first, on the query the AuthProvider observes.
    // `queryClient.clear()` used to run before this, which detached that
    // observer, so `user` stayed stale and /login bounced back to the
    // dashboard, whose refetch then showed "Not signed in".
    queryClient.setQueryData(['auth', 'me'], null);
    // Then drop every cached patient record, not just the session.
    queryClient.removeQueries({ predicate: (query) => query.queryKey[0] !== 'auth' });
  };

  return (
    <AuthContext.Provider
      value={{
        user: data ?? null,
        isLoading,
        login: (email, password) => loginMutation.mutateAsync({ email, password }),
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
