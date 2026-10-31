#!/usr/bin/env sh
# Start the ThematicML web application on http://127.0.0.1:${PORT:-8000}
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
[ -f frontend/dist/index.html ] || (cd frontend && npm install && npm run build)
# --no-access-log: request URLs can contain search terms and record filters typed by the researcher
exec "$PY" -m uvicorn thematicml_api.main:app --host 127.0.0.1 --port "${PORT:-8000}" --no-access-log
