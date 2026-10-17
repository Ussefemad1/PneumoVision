#!/usr/bin/env bash
# HTTPS certificate for <ip-with-dashes>.sslip.io from Let's Encrypt.
#
#   infra/aws/cert.sh ensure   issue once if no Let's Encrypt cert yet (deploy.sh)
#   infra/aws/cert.sh issue    request / re-request it now
#   infra/aws/cert.sh renew    renew when due (the systemd timer, twice a day)
#
# HTTP-01 through the running nginx: certbot (in its own container) writes the
# challenge into infra/aws/certbot/www, which nginx serves on port 80 under
# /.well-known/acme-challenge/. The certificate is then copied into
# infra/nginx/certs/server.{crt,key} -- the names nginx.conf expects -- and
# nginx reloads. If Let's Encrypt cannot issue (port 80 closed, rate limit,
# sslip.io unreachable), the self-signed pair from gen-certs.sh stays in place
# and the site still works over TLS with a browser warning.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CERTBOT_IMAGE="certbot/certbot:v2.11.0"
CONF="$REPO/infra/aws/certbot/conf"
WWW="$REPO/infra/aws/certbot/www"
CERTS="$REPO/infra/nginx/certs"
COMPOSE="$REPO/infra/aws/compose.sh"

env_value() { grep "^$1=" "$REPO/.env" 2>/dev/null | cut -d= -f2- || true; }
HOST="$(env_value PUBLIC_HOST)"
EMAIL="$(env_value LETSENCRYPT_EMAIL)"
[[ -n "$HOST" ]] || { echo "cert: PUBLIC_HOST is not set in $REPO/.env (run bootstrap.sh)" >&2; exit 1; }
mkdir -p "$CONF" "$WWW/.well-known/acme-challenge" "$CERTS"

certbot() {
  docker run --rm \
    -v "$CONF:/etc/letsencrypt" -v "$WWW:/var/www/certbot" \
    "$CERTBOT_IMAGE" "$@"
}

have_le_cert() {
  docker run --rm -v "$CONF:/etc/letsencrypt:ro" --entrypoint sh "$CERTBOT_IMAGE" \
    -c "test -s /etc/letsencrypt/live/$HOST/fullchain.pem"
}

# Copy the live certificate to the names nginx reads, owned by this user
# (certbot's own files are root-only), and reload nginx when it changed.
install_cert() {
  local before after
  before="$(sha256sum "$CERTS/server.crt" 2>/dev/null | cut -d' ' -f1 || true)"
  docker run --rm -v "$CONF:/etc/letsencrypt:ro" -v "$CERTS:/out" --entrypoint sh "$CERTBOT_IMAGE" -c "
    cp -L /etc/letsencrypt/live/$HOST/fullchain.pem /out/server.crt.new &&
    cp -L /etc/letsencrypt/live/$HOST/privkey.pem /out/server.key.new &&
    chown $(id -u):$(id -g) /out/server.crt.new /out/server.key.new &&
    chmod 644 /out/server.crt.new && chmod 600 /out/server.key.new &&
    mv /out/server.crt.new /out/server.crt && mv /out/server.key.new /out/server.key"
  after="$(sha256sum "$CERTS/server.crt" | cut -d' ' -f1)"
  if [[ "$before" != "$after" ]]; then
    "$COMPOSE" exec -T nginx nginx -s reload >/dev/null && echo "cert: nginx reloaded with the new certificate"
  fi
}

fallback() {
  cat >&2 <<EOF
cert: Let's Encrypt could not issue a certificate for $HOST.
      Keeping the self-signed certificate: the site works over HTTPS, but the
      browser shows a warning you have to click through. Usual causes: port 80
      is not open to 0.0.0.0/0 in the security group, the instance's public IP
      is not the one in $HOST, or the Let's Encrypt rate limit (5 per week per
      host name). Fix it and run: infra/aws/cert.sh issue
EOF
  if [[ ! -s "$CERTS/server.crt" ]]; then
    rm -f "$CERTS/server.key"
    CERT_CN="$HOST" bash "$REPO/infra/scripts/gen-certs.sh" >/dev/null
    "$COMPOSE" exec -T nginx nginx -s reload >/dev/null 2>&1 || true
  fi
}

issue() {
  local who=(--register-unsafely-without-email)
  [[ -z "$EMAIL" ]] || who=(--email "$EMAIL")
  if certbot certonly --webroot -w /var/www/certbot -d "$HOST" \
      --agree-tos --non-interactive --keep-until-expiring "${who[@]}"; then
    install_cert
    echo "cert: Let's Encrypt certificate installed for $HOST"
  else
    fallback
    return 1
  fi
}

case "${1:-ensure}" in
  ensure)
    if have_le_cert; then install_cert; echo "cert: Let's Encrypt certificate present for $HOST"; else issue; fi ;;
  issue)
    issue ;;
  renew)
    if have_le_cert; then
      certbot renew --webroot -w /var/www/certbot --quiet
      install_cert
    else
      echo "cert: no Let's Encrypt certificate yet; trying to issue one"
      issue || true
    fi ;;
  *)
    echo "usage: $0 ensure|issue|renew" >&2; exit 2 ;;
esac
