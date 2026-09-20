#!/usr/bin/env bash
# Deploy Sentinel AI to the shared Hostinger VPS, alongside the other tenants.
#
#   bash deploy/vps/deploy.sh eagle-eye.microeagle.online
#
# What it does, in order:
#   1. builds the frontend locally (so the VPS does not need node)
#   2. uploads only what the image needs to /opt/stacks/sentinel/src
#   3. writes /opt/stacks/sentinel/.env with generated secrets (kept locally too)
#   4. builds + starts the container, published on 127.0.0.1:8090
#   5. adds ONE new Traefik file-provider router for this host
#   6. verifies health over public HTTPS
#
# Never touches another project's stack or Traefik file. Re-running is safe:
# the data volume and the .env secrets are preserved.
set -euo pipefail

HOST="${1:?usage: deploy.sh <hostname>   e.g. eagle-eye.microeagle.online}"
VPS="${VPS:-187.127.191.13}"
KEY="${KEY:-$HOME/.ssh/nexus_vps}"
STACK="/opt/stacks/sentinel"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECRETS="$ROOT/deploy/vps/.secrets.env"          # git-ignored, local copy
# This ISP's link to the VPS drops connections now and then, so retry.
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25
     -o ConnectionAttempts=4 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$VPS")

# getent does not exist in Git Bash; fall back to nslookup, then python.
resolve() {
  local host="$1" ip=""
  ip="$(getent hosts "$host" 2>/dev/null | awk '{print $1}' | head -1)"
  [ -z "$ip" ] && ip="$(nslookup "$host" 2>/dev/null | awk '/^Address: /{a=$2} END{print a}')"
  [ -z "$ip" ] && ip="$(python -c "import socket,sys;print(socket.gethostbyname(sys.argv[1]))" "$host" 2>/dev/null)"
  printf '%s' "$ip"
}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

case "$HOST" in
  *_*) echo "ERROR: '$HOST' contains an underscore. Let's Encrypt will refuse a"
       echo "certificate for it (underscores are not legal in hostnames)."
       echo "Add an A record with a hyphen instead, e.g. eagle-eye.microeagle.online"
       exit 2 ;;
esac

say "0/6 checks"
"${SSH[@]}" "test -d /opt/stacks/traefik/dynamic" \
  || { echo "cannot reach the VPS, or Traefik is not where expected"; exit 1; }
resolved="$(resolve "$HOST")"
[ -n "$resolved" ] || { echo "ERROR: $HOST does not resolve yet - add the A record first"; exit 2; }
echo "  $HOST -> $resolved (VPS is $VPS)"
[ "$resolved" = "$VPS" ] || echo "  WARNING: that is not the VPS IP; the certificate will fail"

say "1/6 build frontend"
(cd "$ROOT/frontend" && npm run build >/dev/null) && echo "  dist built"

say "2/6 upload source"
"${SSH[@]}" "mkdir -p $STACK/src/docker"
tar -czf - -C "$ROOT" \
    --exclude='backend/.venv' --exclude='backend/storage' --exclude='backend/__pycache__' \
    --exclude='**/__pycache__' --exclude='backend/.env' --exclude='backend/tests' \
    backend frontend/dist portal docker/Dockerfile.vps \
  | "${SSH[@]}" "tar -xzf - -C $STACK/src"
echo "  uploaded"

say "3/6 secrets + compose"
if [ ! -f "$SECRETS" ]; then
  umask 077
  {
    echo "SENTINEL_JWT_SECRET=$(openssl rand -hex 32)"
    echo "SENTINEL_ADMIN_USERNAME=admin"
    echo "SENTINEL_ADMIN_PASSWORD=$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
  } > "$SECRETS"
  echo "  generated new secrets -> $SECRETS (keep this file)"
else
  echo "  reusing $SECRETS"
fi
{
  cat "$SECRETS"
  echo "SENTINEL_CORS_ORIGINS=https://$HOST"
  # The portal's identity service: the Sentinel_Auth workflow in n8n.
  echo "SENTINEL_PORTAL_AUTH_BASE_URL=https://n8n-kmar.srv1804200.hstgr.cloud/webhook/sentinel"
  echo "SENTINEL_PORTAL_ADMIN_EMAIL=${PORTAL_ADMIN_EMAIL:-thakurvansh75610@gmail.com}"
  echo "SENTINEL_PORTAL_NOTIFY_KEY=$(cat "$ROOT/deploy/n8n/notify_key.txt" 2>/dev/null)"
} | "${SSH[@]}" "cat > $STACK/.env && chmod 600 $STACK/.env"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new \
    "$ROOT/deploy/vps/docker-compose.yml" "root@$VPS:$STACK/docker-compose.yml" >/dev/null

say "4/6 build + start"
"${SSH[@]}" "cd $STACK && docker compose up -d --build" 2>&1 | tail -5

say "5/6 Traefik router"
sed "s/HOSTNAME_PLACEHOLDER/$HOST/" "$ROOT/deploy/vps/traefik-sentinel.yml" \
  | "${SSH[@]}" "cat > /opt/stacks/traefik/dynamic/sentinel.yml"
echo "  /opt/stacks/traefik/dynamic/sentinel.yml written (Traefik hot-reloads)"

say "6/6 verify"
"${SSH[@]}" "docker ps --filter name=sentinel-app --format '  container: {{.Status}}'"
for i in $(seq 1 30); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "https://$HOST/api/system/health" || true)"
  [ "$code" = "200" ] && break
  sleep 10
done
echo "  https://$HOST/api/system/health -> ${code:-no answer}"
[ "${code:-}" = "200" ] || { echo "not healthy yet; check: ssh root@$VPS 'docker logs sentinel-app --tail 50'"; exit 1; }
curl -s "https://$HOST/api/system/health" | head -c 300; echo
echo
echo "Dashboard:  https://$HOST"
echo "Admin user: see $SECRETS"
