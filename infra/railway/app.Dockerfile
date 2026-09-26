# PneumoVision — Railway "app" service (web + API in one container).
#
# Build from the repo root:
#   docker build -f infra/railway/app.Dockerfile -t pneumovision-app .
#
# This service runs in DEMO_MODE with the in-process MongoDB that
# mongodb-memory-server provides — there is no database service attached, and
# MONGO_URI is left empty. Everything it stores is synthetic seed data,
# recreated on every boot, so a restart is a clean slate by design.
#
# The inference service is a separate Railway service, built from
# services/inference/Dockerfile. Point INFERENCE_URL at its internal address.

# ── build ────────────────────────────────────────────────────────────────────
# Debian, not Alpine. mongodb-memory-server downloads an official mongod
# build, and those are linked against glibc — on musl they fail at exec with
# "no such file or directory", which is a famously unhelpful way to learn this.
FROM node:22-bookworm-slim AS build

WORKDIR /repo

# Workspace manifests first, so dependency installs cache independently of src.
COPY package.json package-lock.json tsconfig.base.json ./
COPY packages/shared/package.json packages/shared/
COPY apps/api/package.json apps/api/
COPY apps/web/package.json apps/web/
RUN npm ci

COPY packages/shared packages/shared
COPY apps/api apps/api
COPY apps/web apps/web

RUN npm run build --workspace @pneumovision/shared \
 && npm run build --workspace @pneumovision/api \
 && npm run build --workspace @pneumovision/web

# ── mongod binary, fetched at build time ─────────────────────────────────────
# Without this the first boot downloads ~600 MB before the app can answer a
# single request, on every cold start and every redeploy. Baking it in trades
# image size for a container that starts in seconds.
#
# The version is read from the compiled app rather than repeated here, so the
# cached binary is by construction the exact build the server will ask for. A
# literal would be free to drift, and the failure mode of drift is silent:
# the binary is simply re-downloaded at runtime and the cache does nothing.
ENV MONGOMS_DOWNLOAD_DIR=/opt/mongodb-binaries
RUN mkdir -p "$MONGOMS_DOWNLOAD_DIR" \
 && node --input-type=module -e "\
      const { MONGO_BINARY_VERSION } = await import('/repo/apps/api/dist/config/mongoBinary.js'); \
      const { MongoMemoryServer } = await import('mongodb-memory-server'); \
      console.log('pre-caching mongod ' + MONGO_BINARY_VERSION + ' into ' + process.env.MONGOMS_DOWNLOAD_DIR); \
      const server = await MongoMemoryServer.create({ binary: { version: MONGO_BINARY_VERSION } }); \
      await server.stop(); \
      console.log('mongod cached'); \
    " \
 && ls -lh "$MONGOMS_DOWNLOAD_DIR"

# Drop dev dependencies from the tree we copy forward. mongodb-memory-server
# is a runtime dependency of this image (it *is* the database), so it lives in
# `dependencies` in apps/api/package.json and survives this prune. If it ever
# moves back to devDependencies, the container starts and then dies looking
# for a module that is not there.
RUN npm prune --omit=dev \
 && node -e "require.resolve('mongodb-memory-server'); console.log('mongodb-memory-server survived prune')"

# ── runtime ──────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS runtime

# mongod links against libcurl and OpenSSL; node:22-bookworm-slim ships
# neither. Missing them surfaces as a mongod that exits immediately with a
# shared-object error, which mongodb-memory-server reports as a generic
# startup timeout — so install them explicitly rather than debug that later.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libcurl4 openssl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

ENV NODE_ENV=production \
    DEMO_MODE=true \
    MOCK_MODE=true \
    MONGOMS_DOWNLOAD_DIR=/opt/mongodb-binaries \
    WEB_DIST_DIR=/repo/apps/web/dist

# The pre-cached mongod, and the pruned production tree.
COPY --from=build --chown=node:node /opt/mongodb-binaries /opt/mongodb-binaries
COPY --from=build --chown=node:node /repo/node_modules ./node_modules
COPY --from=build --chown=node:node /repo/package.json ./package.json
COPY --from=build --chown=node:node /repo/packages/shared/dist ./packages/shared/dist
COPY --from=build --chown=node:node /repo/packages/shared/package.json ./packages/shared/package.json
COPY --from=build --chown=node:node /repo/apps/api/dist ./apps/api/dist
COPY --from=build --chown=node:node /repo/apps/api/package.json ./apps/api/package.json
COPY --from=build --chown=node:node /repo/apps/web/dist ./apps/web/dist

# The seed writes synthetic X-ray PNGs relative to the working directory, so
# that path has to be writable by the unprivileged user.
RUN mkdir -p /repo/.demo-data/images && chown -R node:node /repo/.demo-data

USER node
EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||4000)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

# Railway injects PORT; the app reads API_PORT. Bridge the two here rather
# than teach the config two names for one thing.
CMD ["sh", "-c", "API_PORT=${PORT:-4000} exec node apps/api/dist/server.js"]
