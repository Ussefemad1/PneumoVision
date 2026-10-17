#!/usr/bin/env bash
# Deploy (or update) PneumoVision on the server. Run as `ubuntu`:
#
#   ~/PneumoVision/infra/aws/deploy.sh            pull, check, build, start, verify
#   ~/PneumoVision/infra/aws/deploy.sh --no-pull  same, without git pull
#
# Steps, any failure stops with a non-zero exit and says why:
#   1. git pull --ff-only
#   2. data-safety guard (data-guard.sh): no MIMIC-derived file on this server
#   3. checkpoints: every file the configured INFERENCE_MODE loads is present;
#      in fusion mode each checkpoint's sha256 equals its run.json manifest row
#   4. docker compose build, then up -d
#   5. wait until every service is healthy (the models take minutes on CPU)
#   6. HTTPS certificate (Let's Encrypt on first deploy, self-signed fallback)
#   7. smoke tests: /api/v1/health through nginx, and the inference
#      service's /v1/health from inside the Docker network
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$REPO/infra/aws/compose.sh"
cd "$REPO"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mdeploy FAILED:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f .env ]] || fail "no .env -- run infra/aws/bootstrap.sh first"
env_value() { grep "^$1=" .env | cut -d= -f2- || true; }
MODE="$(env_value INFERENCE_MODE)"
MODE="${MODE:-fusion}"
HOST="$(env_value PUBLIC_HOST)"

if [[ "${1:-}" != "--no-pull" ]]; then
  step "git pull --ff-only"
  git pull --ff-only || fail "git pull --ff-only failed (local changes on the server? see git status)"
fi

step "Data-safety guard"
bash infra/aws/data-guard.sh "$REPO" || fail "data-safety guard refused (see the list above)"

step "Checkpoints (INFERENCE_MODE=$MODE)"
case "$MODE" in
  fusion)
    for task in mortality phenotyping; do
      dir="checkpoints/round3/$task"
      [[ -f "$dir/run.json" ]] || fail "missing $dir/run.json (infra/aws/copy-checkpoints.md)"
      mapfile -t ckpts < <(find "$dir" -maxdepth 1 -name '*.pth.tar' -type f)
      ((${#ckpts[@]} == 1)) || fail "$dir must hold exactly one *.pth.tar (found ${#ckpts[@]})"
      expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_row"]["sha256"])' "$dir/run.json")" \
        || fail "$dir/run.json has no manifest_row.sha256"
      run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$dir/run.json")"
      actual="$(sha256sum "${ckpts[0]}" | cut -d' ' -f1)"
      [[ "$actual" == "$expected" ]] \
        || fail "${ckpts[0]}: sha256 $actual does not match $run_id's manifest row ($expected). Copy the file again."
      printf '  %-12s %-14s %s  %s  sha256 %s OK\n' "$task" "$run_id" \
        "$(du -h "${ckpts[0]}" | cut -f1)" "$(basename "${ckpts[0]}")" "${actual:0:12}"
    done
    heads=checkpoints/round3/reader_heads.pt
    if [[ -f "$heads" ]]; then
      [[ -f checkpoints/round3/reader_heads.json ]] || fail "$heads has no reader_heads.json next to it"
      expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' checkpoints/round3/reader_heads.json)"         || fail "checkpoints/round3/reader_heads.json has no sha256"
      actual="$(sha256sum "$heads" | cut -d' ' -f1)"
      [[ "$actual" == "$expected" ]] || fail "$heads: sha256 $actual does not match reader_heads.json ($expected). Copy both files again."
      printf '  %-12s %-14s %s  %s  sha256 %s OK\n' readers "Round 1 heads" "$(du -h "$heads" | cut -f1)" reader_heads.pt "${actual:0:12}"
    else
      echo "  reader_heads.pt not installed: the report's 'By source' will say 'Standalone readers not installed'"
      echo "  (infra/aws/copy-checkpoints.md, 'Standalone readers')"
    fi ;;
  readers)
    missing="$(python3 - <<'EOF'
import re, pathlib
src = pathlib.Path("services/inference/app/readers.py").read_text(encoding="utf-8")
# Every checkpoint path readers.py loads (relative to checkpoints/).
paths = re.findall(r'"((?:mortality|phenotyping)/[^"]+\.pth\.tar)"', src)
paths += ["".join(p) for p in re.findall(r'checkpoint="([^"]+)"\s*"([^"]+)"', src)]
missing = sorted({p for p in paths if not (pathlib.Path("checkpoints") / p).is_file()})
print("\n".join(missing))
EOF
)"
    [[ -z "$missing" ]] || fail "INFERENCE_MODE=readers needs these files under checkpoints/:
$missing" ;;
  mock)
    echo "  mock mode: no checkpoints needed" ;;
  *)
    fail "INFERENCE_MODE=$MODE in .env is not one of fusion, readers, mock" ;;
esac

step "Build images (on this server)"
"$COMPOSE" build || fail "docker compose build failed"

state_of() {
  local id
  id="$("$COMPOSE" ps -aq "$1" 2>/dev/null | head -1 || true)"
  [[ -n "$id" ]] || { echo absent; return; }
  # restarting / exited / dead win over a stale health status.
  docker inspect -f '{{if ne .State.Status "running"}}{{.State.Status}}{{else if .State.Health}}{{.State.Health.Status}}{{else}}healthy{{end}}' "$id"
}
inference_health() {
  "$COMPOSE" exec -T inference python -c "import urllib.request, urllib.error
try: print(urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4).read().decode())
except urllib.error.HTTPError as e: print(e.read().decode())
except Exception as e: print('not answering yet:', e)" 2>&1 || true
}
show_logs() {
  echo "---- $1: $("$COMPOSE" ps -a "$1" 2>&1 | tail -1)" >&2
  "$COMPOSE" logs --no-color --tail=50 "$1" >&2 || true
}
# Wait until every named service is healthy. Fail fast:
#   - inference reporting itself degraded (missing / mismatched checkpoint);
#   - any other service restarting, exited or unhealthy for 3 minutes
#     (inference alone may take its 10-minute start period to load models).
# On failure the culprit's last 50 log lines are printed.
wait_healthy() {
  local deadline=$((SECONDS + 1200)) pending s state h
  declare -A bad_since=()
  while :; do
    pending=()
    for s in "$@"; do
      state="$(state_of "$s")"
      if [[ "$state" == "healthy" ]]; then unset "bad_since[$s]"; continue; fi
      pending+=("$s:$state")
      if [[ "$s" != inference && "$state" =~ ^(restarting|exited|dead|unhealthy)$ ]]; then
        : "${bad_since[$s]:=$SECONDS}"
        if ((SECONDS - bad_since[$s] >= 180)); then
          show_logs "$s"
          fail "$s has been $state for 3 minutes (its last 50 log lines are above)"
        fi
      elif [[ "$s" == inference && "$state" =~ ^(restarting|exited|dead)$ ]]; then
        show_logs inference
        fail "the inference container is $state (its last 50 log lines are above)"
      fi
    done
    ((${#pending[@]} == 0)) && return 0
    if [[ " $* " == *" inference "* && " ${pending[*]} " == *" inference:"* ]]; then
      h="$(inference_health)"
      if [[ "$h" == *'"status":"degraded"'* ]]; then
        show_logs inference
        fail "the inference service is degraded: $h"
      fi
    fi
    if ((SECONDS >= deadline)); then
      for s in "${pending[@]}"; do show_logs "${s%%:*}"; done
      fail "not healthy after 20 minutes: ${pending[*]}"
    fi
    echo "  waiting: ${pending[*]}"
    sleep 15
  done
}

step "Start the data stores and the inference service"
"$COMPOSE" up -d --remove-orphans mongo redis inference \
  || fail "docker compose up failed"
wait_healthy mongo redis inference
echo "  healthy; inference: $(inference_health)"

step "Start the API, the web app and nginx"
"$COMPOSE" up -d --remove-orphans || fail "docker compose up failed"
wait_healthy mongo redis inference api web nginx
echo "  all healthy"

step "HTTPS certificate"
bash infra/aws/cert.sh ensure || echo "  (continuing with the self-signed certificate)"

step "Smoke tests"
api="$(curl -fsS -k --resolve "$HOST:443:127.0.0.1" "https://$HOST/api/v1/health")" \
  || fail "https://$HOST/api/v1/health through nginx did not answer 200"
echo "  nginx -> api:  $api"
inf="$("$COMPOSE" exec -T api node -e "fetch('http://inference:8000/v1/health').then(async r => { console.log(await r.text()); process.exit(r.ok ? 0 : 1) }).catch(e => { console.error(e.message); process.exit(1) })")" \
  || fail "inference /v1/health from the api container failed: $inf"
echo "  api -> inference: $inf"
[[ "$MODE" == "mock" || "$inf" == *'"modelsLoaded":true'* ]] || fail "inference reports no models loaded: $inf"

step "Deployed"
echo "  https://$HOST"
"$COMPOSE" ps --format 'table {{.Service}}\t{{.Status}}'
