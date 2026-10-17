#!/usr/bin/env bash
# Back up the database (and the seeded/uploaded X-rays) to ~/backups.
#
#   ~/PneumoVision/infra/aws/backup.sh
#
# Writes ~/backups/pneumovision-<UTC time>.mongo.gz (mongodump --archive --gzip)
# and ~/backups/pneumovision-<UTC time>.images.tgz, keeps the newest 10 of
# each, and prints the scp command to pull them to your PC.
#
# Run it, and pull the files, BEFORE the AWS credits run out: on the Free plan
# the account is closed then (not billed), and the instance and its disk go
# with it.
#
# Restore a database backup (replaces the current data):
#   ~/PneumoVision/infra/aws/backup.sh restore ~/backups/pneumovision-<time>.mongo.gz
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$REPO/infra/aws/compose.sh"

if [[ "${1:-}" == "restore" ]]; then
  file="${2:?usage: backup.sh restore <file>.mongo.gz}"
  [[ -s "$file" ]] || { echo "backup: no such backup: $file" >&2; exit 1; }
  "$COMPOSE" exec -T mongo sh -c 'mongorestore --quiet --drop --archive --gzip \
    -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" \
    --authenticationDatabase admin' <"$file"
  echo "restored $file"
  exit 0
fi
OUT="$HOME/backups"
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
mkdir -p "$OUT"
umask 077

db="$OUT/pneumovision-$STAMP.mongo.gz"
"$COMPOSE" exec -T mongo sh -c 'mongodump --quiet --archive --gzip \
  -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin --db pneumovision' >"$db"
[[ -s "$db" ]] || { echo "backup: mongodump wrote nothing" >&2; rm -f "$db"; exit 1; }

images="$OUT/pneumovision-$STAMP.images.tgz"
api="$("$COMPOSE" ps -q api)"
if [[ -n "$api" ]]; then
  docker run --rm --volumes-from "$api":ro -v "$OUT:/out" alpine:3.20 \
    tar czf "/out/$(basename "$images")" -C /repo .demo-data
  docker run --rm -v "$OUT:/out" alpine:3.20 chown "$(id -u):$(id -g)" "/out/$(basename "$images")"
fi

# Keep the newest 10 of each kind.
for kind in mongo.gz images.tgz; do
  ls -1t "$OUT"/pneumovision-*."$kind" 2>/dev/null | tail -n +11 | xargs -r rm -f
done

ls -lh "$db" ${api:+"$images"}
host="$(grep '^PUBLIC_HOST=' "$REPO/.env" | cut -d= -f2-)"
ip="${host%%.sslip.io}"
ip="${ip//-/.}"
cat <<EOF

Pull them to your PC (PowerShell, \$Key as in infra/aws/README.md):
  scp -i \$Key "ubuntu@${ip}:~/backups/pneumovision-$STAMP.*" "\$HOME\\Downloads\\"
EOF
