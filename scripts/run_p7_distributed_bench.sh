#!/usr/bin/env bash
set -euo pipefail

requests="${1:-100}"
mode="${2:-open}"
qps="${3:-50}"
router_binary="${4:-build/default/cpp/infersched_durable_router}"
engine_binary="${5:-build/default/cpp/infersched_engine}"
python_binary="${6:-.venv/bin/python}"
group="infersched-p7-bench-$$"
router_log="${TMPDIR:-/tmp}/infersched-p7-bench-router-$$.log"
gateway_log="${TMPDIR:-/tmp}/infersched-p7-bench-gateway-$$.log"
ready="${TMPDIR:-/tmp}/infersched-p7-bench-ready-$$"
rm -f "${ready}"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  rm -f "${ready}"
}
trap cleanup EXIT

"${router_binary}" --listen 127.0.0.1:55241 --router-id p7-bench-router \
  --consumer-group "${group}" --requests 100000000 --run-seconds 120 \
  --offset-reset latest \
  --ready-file "${ready}" \
  >"${router_log}" 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
"${engine_binary}" --listen 127.0.0.1:55242 --router 127.0.0.1:55241 \
  --engine-id p7-bench-engine --incarnation "p7-bench-$$" --batch-delay-ms 2 \
  >/dev/null 2>&1 &
engine=$!; pids+=("${engine}")
PYTHONPATH=python INFERSCHED_GATEWAY_CAPACITY=1024 \
  "${python_binary}" -m uvicorn infersched_api.app:app \
  --host 127.0.0.1 --port 58001 >"${gateway_log}" 2>&1 &
gateway=$!; pids+=("${gateway}")
for _ in $(seq 1 100); do
  curl -fsS http://127.0.0.1:58001/healthz >/dev/null 2>&1 && break
  sleep 0.05
done
# A `latest` group must finish assignment before its first submission.
for _ in $(seq 1 200); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
if ! PYTHONPATH=python "${python_binary}" python/loadgen.py \
  --url http://127.0.0.1:58001 --mode "${mode}" --requests "${requests}" \
  --qps "${qps}" --seed 7; then
  cat "${router_log}"
  cat "${gateway_log}"
  exit 1
fi
kill "${router}" 2>/dev/null || true
