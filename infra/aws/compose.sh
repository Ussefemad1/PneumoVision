#!/usr/bin/env bash
# docker compose for the AWS stack, from any directory:
#
#   infra/aws/compose.sh ps
#   infra/aws/compose.sh logs -f --tail=100 inference
#   infra/aws/compose.sh exec api node apps/api/dist/cli/users.js create ...
#
# Pins the project directory to the repo root (so ./checkpoints, ./infra/...
# in the compose file resolve there) and reads the secrets from the repo's
# .env (written by bootstrap.sh, mode 600).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec docker compose \
  --project-directory "$REPO" \
  --env-file "$REPO/.env" \
  -f "$REPO/infra/aws/docker-compose.aws.yml" \
  "$@"
