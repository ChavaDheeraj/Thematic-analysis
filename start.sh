#!/usr/bin/env sh
# Start the ThematicML web application on http://127.0.0.1:${PORT:-8000}
# NETWORK=1 THEMATICML_PASSWORD=... ./start.sh   also lets other devices on the same network in, behind that password
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
[ -f frontend/dist/index.html ] || (cd frontend && npm install && npm run build)
HOST=127.0.0.1
if [ "${NETWORK:-0}" = "1" ]; then
    [ -n "${THEMATICML_PASSWORD:-}" ] || { echo "Set THEMATICML_PASSWORD before letting other devices connect." >&2; exit 1; }
    export THEMATICML_REQUIRE_PASSWORD=1
    HOST=0.0.0.0
    echo "Other devices on this network can open http://<this computer's address>:${PORT:-8000} and sign in as '${THEMATICML_USER:-researcher}'."
fi
# --no-access-log: request URLs can contain search terms and record filters typed by the researcher
exec "$PY" -m uvicorn thematicml_api.main:app --host "$HOST" --port "${PORT:-8000}" --no-access-log
