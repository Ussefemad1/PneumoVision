import { io, type Socket } from 'socket.io-client';

/**
 * Socket.IO client.
 *
 * The handshake authenticates with the same httpOnly cookie as the REST API,
 * so there is no token to pass — `withCredentials` is what makes it work.
 * One shared connection; components subscribe to the rooms they care about.
 */
let socket: Socket | undefined;

export function getSocket(): Socket {
  socket ??= io({
    path: '/socket.io',
    withCredentials: true,
    autoConnect: true,
    transports: ['websocket', 'polling'],
  });
  return socket;
}

export function disconnectSocket(): void {
  socket?.disconnect();
  socket = undefined;
}

export interface PredictionEvent {
  predictionId: string;
  stayId: string;
  task: 'mortality' | 'pneumonia';
  status: string;
  probability: number | null;
  cutoffTime: string;
}

export interface AlertEvent {
  id: string;
  stayId: string;
  task: 'mortality' | 'pneumonia';
  severity: 'info' | 'warning' | 'critical';
  rule: string;
  value: number;
  status: string;
  createdAt: string;
}

export interface VitalsEvent {
  stayId: string;
  ts: string;
  values: Record<string, number | string | null>;
}

export interface SimulationEvent {
  stayId: string;
  running: boolean;
  speed: number;
  cursorHour: number;
  totalHours: number;
}
