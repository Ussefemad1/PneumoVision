import type { Server as HttpServer } from 'node:http';

import { Server as SocketServer } from 'socket.io';

import type { Env } from '../config/env.js';
import { ACCESS_COOKIE, verifyAccessToken, type TokenKeys } from '../lib/tokens.js';
import type { Logger } from '../lib/logger.js';
import { stayRoom, wardRoom } from './bus.js';

/** What we attach to each authenticated socket. */
interface SocketData {
  userId: string;
  role: string;
}

/**
 * Socket.IO server.
 *
 * Authenticated with the same httpOnly access cookie as the REST API — the
 * handshake is rejected outright if it is missing or invalid, so an
 * unauthenticated client cannot subscribe to a patient room.
 */
export function createSocketServer(
  http: HttpServer,
  env: Env,
  keys: TokenKeys,
  logger: Logger,
): SocketServer {
  const io = new SocketServer<
    Record<string, (...args: never[]) => void>,
    Record<string, (...args: never[]) => void>,
    Record<string, (...args: never[]) => void>,
    SocketData
  >(http, {
    cors: { origin: env.WEB_ORIGIN, credentials: true },
    path: '/socket.io',
  });

  io.use((socket, next) => {
    const raw = socket.handshake.headers.cookie ?? '';
    const token = raw
      .split(';')
      .map((part) => part.trim())
      .find((part) => part.startsWith(`${ACCESS_COOKIE}=`))
      ?.slice(ACCESS_COOKIE.length + 1);

    if (!token) {
      next(new Error('unauthenticated'));
      return;
    }

    verifyAccessToken(decodeURIComponent(token), keys, env)
      .then((claims) => {
        socket.data.userId = claims.sub;
        socket.data.role = claims.role;
        next();
      })
      .catch(() => next(new Error('unauthenticated')));
  });

  io.on('connection', (socket) => {
    logger.debug({ socketId: socket.id, userId: socket.data.userId }, 'socket connected');

    // Clients subscribe to the stay they are viewing and the ward they are
    // watching; everything else is filtered server-side by room.
    socket.on('subscribe:stay', (stayId: unknown) => {
      if (typeof stayId === 'string' && /^[0-9a-f]{24}$/i.test(stayId)) {
        void socket.join(stayRoom(stayId));
      }
    });

    socket.on('unsubscribe:stay', (stayId: unknown) => {
      if (typeof stayId === 'string') void socket.leave(stayRoom(stayId));
    });

    socket.on('subscribe:ward', (ward: unknown) => {
      if (typeof ward === 'string' && ward.length <= 40) void socket.join(wardRoom(ward));
    });
  });

  return io;
}
