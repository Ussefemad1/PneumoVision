# Railway deployment

Two services, both built from this repo.

| Service     | Dockerfile                      | Notes                                   |
| ----------- | ------------------------------- | --------------------------------------- |
| `app`       | `infra/railway/app.Dockerfile`  | Web + API in one container. Public.     |
| `inference` | `services/inference/Dockerfile` | MOCK_MODE. **Private — do not expose.** |

Set the build context to the repo root for both, and the Dockerfile path per
the table.

## Why there is no database service

The `app` service runs with `DEMO_MODE=true` and an empty `MONGO_URI`, which
starts an in-process MongoDB via `mongodb-memory-server`. Consequences worth
being explicit about:

- **Storage is ephemeral.** Every restart, redeploy and sleep/wake gives a
  fresh database, re-seeded from scratch. That is fine for a demo of
  synthetic data and wrong for anything else.
- The mongod binary is baked into the image at build time
  (`MONGOMS_DOWNLOAD_DIR=/opt/mongodb-binaries`), so a cold start does not
  pull ~600 MB. It makes the image large; that is the trade.
- The image is Debian-based, not Alpine, because the official mongod builds
  are linked against glibc.

Moving to a real database later is a two-variable change — set `MONGO_URI` to
a Railway MongoDB plugin and `DEMO_MODE=false` — but note that turning off
demo mode makes `REDIS_URL`, the S3 variables, `FIELD_ENCRYPTION_KEY_HEX` and
`CSRF_SECRET` mandatory, because those code paths become live.

## Environment variables

Generate the secrets with `npm run gen:secrets -- --demo` and paste them in.
Never reuse the values from a local `.env.demo` that has been shared.

### `app`

| Variable                  | Value                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `DEMO_MODE`               | `true` (already set in the image)                                                      |
| `MOCK_MODE`               | `true` (already set in the image)                                                      |
| `MONGO_URI`               | _leave empty_ — empty means in-process MongoDB                                         |
| `WEB_ORIGIN`              | the service's public URL, e.g. `https://pneumovision.up.railway.app`                   |
| `JWT_PRIVATE_KEY_B64`     | from `gen:secrets`                                                                     |
| `JWT_PUBLIC_KEY_B64`      | from `gen:secrets`                                                                     |
| `INFERENCE_HMAC_SECRET`   | from `gen:secrets` — **must match the inference service**                              |
| `INFERENCE_URL`           | the inference service's private address, e.g. `http://inference.railway.internal:8000` |
| `SEED_ADMIN_EMAIL`        | `admin@pneumovision.local`                                                             |
| `SEED_ADMIN_PASSWORD`     | from `gen:secrets`                                                                     |
| `SEED_CLINICIAN_EMAIL`    | `clinician@pneumovision.local`                                                         |
| `SEED_CLINICIAN_PASSWORD` | from `gen:secrets` — this is the demo login                                            |
| `LOG_LEVEL`               | `info`                                                                                 |

`PORT` is injected by Railway and bridged to `API_PORT` by the container's
command; do not set `API_PORT` yourself.

`WEB_ORIGIN` must be the real public URL. It is the CORS origin and the
Socket.IO origin, so a wrong value shows up as a dashboard that loads but
never receives live updates.

### `inference`

| Variable                | Value                                                           |
| ----------------------- | --------------------------------------------------------------- |
| `MOCK_MODE`             | `true` — no trained checkpoints exist; `false` refuses to start |
| `INFERENCE_HMAC_SECRET` | the same value as the `app` service                             |
| `LOG_LEVEL`             | `warning`                                                       |

## First boot

Expect roughly 30–60 s before the first response: the container starts
mongod, seeds twelve synthetic stays, and backfills 108 predictions through
the real prediction pipeline so the dashboard is not empty. The healthcheck
allows 90 s for this.

Sign in with `SEED_CLINICIAN_EMAIL` / `SEED_CLINICIAN_PASSWORD`.

## Known limits of this deployment

- Ephemeral data, as above.
- ICU Replay timers live in process memory, so a restart stops any running
  replay (`TODO(phase-8)` moves them to a queue).
- Predictions run in-process rather than on a BullMQ worker
  (`TODO(phase-6)`), so a slow inference call occupies a request.
- Everything the app displays is synthetic. No MIMIC record, image or note is
  present in the image or the database.
