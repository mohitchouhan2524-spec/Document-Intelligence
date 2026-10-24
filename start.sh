#!/bin/bash
set -e

export API_URL="http://127.0.0.1:8000"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

streamlit run frontend/streamlit_app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --server.enableCORS=false &
UI_PID=$!

trap 'kill "$API_PID" "$UI_PID" 2>/dev/null' TERM INT

wait -n
STATUS=$?
kill "$API_PID" "$UI_PID" 2>/dev/null || true
exit "$STATUS"