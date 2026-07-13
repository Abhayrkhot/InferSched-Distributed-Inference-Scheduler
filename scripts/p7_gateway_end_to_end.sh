#!/usr/bin/env bash
set -euo pipefail

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
python_binary="${3:?python binary required}"
root="${4:?repository root required}"
group="infersched-p7-gateway-$$"
ready="${TMPDIR:-/tmp}/infersched-p7-gateway-ready-$$"
rm -f "${ready}"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  rm -f "${ready}"
}
trap cleanup EXIT

"${router_binary}" --listen 127.0.0.1:55231 --router-id p7-router \
  --consumer-group "${group}" --offset-reset latest --requests 1 --run-seconds 30 \
  --ready-file "${ready}" \
  >/dev/null 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
"${engine_binary}" --listen 127.0.0.1:55232 --router 127.0.0.1:55231 \
  --engine-id p7-engine --incarnation p7-engine-1 --batch-delay-ms 20 \
  >/dev/null 2>&1 &
engine=$!; pids+=("${engine}")

PYTHONPATH="${root}/python" INFERSCHED_GATEWAY_CAPACITY=8 \
  "${python_binary}" -m uvicorn infersched_api.app:app \
  --host 127.0.0.1 --port 58000 >/dev/null 2>&1 &
gateway=$!; pids+=("${gateway}")
for _ in $(seq 1 100); do
  curl -fsS http://127.0.0.1:58000/healthz >/dev/null 2>&1 && break
  sleep 0.05
done
curl -fsS http://127.0.0.1:58000/healthz >/dev/null
for _ in $(seq 1 200); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]]
PYTHONPATH="${root}/python" "${python_binary}" "${root}/python/loadgen.py" \
  --url http://127.0.0.1:58000 --mode open --requests 1 --qps 10 --seed 7
wait "${router}"
