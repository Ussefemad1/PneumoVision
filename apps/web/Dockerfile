# ── build ────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS build
WORKDIR /repo

COPY package.json package-lock.json tsconfig.base.json ./
COPY packages/shared/package.json packages/shared/
COPY apps/web/package.json apps/web/
RUN npm ci --workspace @pneumovision/shared --workspace @pneumovision/web --include-workspace-root

COPY packages/shared packages/shared
COPY apps/web apps/web
RUN npm run build --workspace @pneumovision/shared \
 && npm run build --workspace @pneumovision/web

# ── runtime ──────────────────────────────────────────────────────────────────
# Static assets only. All API and socket traffic is routed by the edge Nginx.
FROM nginx:1.27-alpine AS runtime
COPY --from=build /repo/apps/web/dist /usr/share/nginx/html
COPY infra/nginx/web.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1
