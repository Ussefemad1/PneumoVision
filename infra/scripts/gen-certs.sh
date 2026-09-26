#!/usr/bin/env bash
# Generates a self-signed TLS certificate for local development.
# Never use the output anywhere but a developer machine.
set -euo pipefail

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/nginx/certs"
mkdir -p "$CERT_DIR"

if [[ -f "$CERT_DIR/server.crt" && -f "$CERT_DIR/server.key" ]]; then
  echo "certs already present in $CERT_DIR — delete them to regenerate"
  exit 0
fi

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.crt" \
  -days 365 \
  -subj "/C=EG/ST=Alexandria/L=Alexandria/O=AASTMT/OU=PneumoVision/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:api,DNS:web,IP:127.0.0.1"

chmod 600 "$CERT_DIR/server.key"
echo "wrote $CERT_DIR/server.crt and server.key (self-signed, 365 days)"
echo "Your browser will warn on first visit — expected for a self-signed cert."
