#!/usr/bin/env bash
# Distributed benchmark: gateway -> Kafka -> one durable Router -> N Engines.
#
#   scripts/run_p7_distributed_bench.sh [requests] [open|closed] [qps]
#
# Environment knobs:
#   ENGINES=2            Engine processes behind the Router
#   ROUTER_WORKERS=32    concurrent dispatch workers in the Router
#   TIME_SCALE=0         Engine wall-clock pacing of the cost model (1 = real time)
#   STEP_DELAY_MS=2      fixed wall-clock delay per Engine iteration
#   WARMUP=50            unmeasured requests before the measured run
#   LOADGEN_ARGS=""      extra loadgen.py flags (e.g. "--prefixes 8 --prefix-tokens 256")
set -euo pipefail
# Unique per run even inside a fresh container, where $$ repeats.
uid="$$-${RANDOM}${RANDOM}"

requests="${1:-1000}"
mode="${2:-open}"
qps="${3:-200}"
engines="${ENGINES:-2}"
router_binary="${ROUTER_BINARY:-build/default/cpp/infersched_durable_router}"
engine_binary="${ENGINE_BINARY:-build/default/cpp/infersched_engine}"
python_binary="${PYTHON_BINARY:-.venv/bin/python}"
group="infersched-p7-bench-${uid}"
tmp="${TMPDIR:-/tmp}"
router_log="${tmp}/infersched-p7-bench-router-${uid}.log"
gateway_log="${tmp}/infersched-p7-bench-gateway-${uid}.log"
ready="${tmp}/infersched-p7-bench-ready-${uid}"
rm -f "${ready}"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  rm -f "${ready}"
}
trap cleanup EXIT

"${router_binary}" --listen 127.0.0.1:55241 --router-id p7-bench-router \
  --consumer-group "${group}" --offset-reset latest --ready-file "${ready}" \
  --workers "${ROUTER_WORKERS:-32}" >"${router_log}" 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
for index in $(seq 1 "${engines}"); do
  "${engine_binary}" --listen "127.0.0.1:$((55300 + index))" --router 127.0.0.1:55241 \
    --engine-id "p7-bench-engine-${index}" --incarnation "p7-bench-${index}-${uid}" \
    --batch-delay-ms "${STEP_DELAY_MS:-2}" --time-scale "${TIME_SCALE:-0}" \
    >/dev/null 2>&1 &
  pids+=("$!")
done
PYTHONPATH=python INFERSCHED_GATEWAY_CAPACITY=4096 \
  "${python_binary}" -m uvicorn infersched_api.app:app \
  --host 127.0.0.1 --port 58001 --log-level warning >"${gateway_log}" 2>&1 &
pids+=("$!")
# Ready = results consumer positioned (the gateway rejects submissions before).
for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:58001/readyz >/dev/null 2>&1 && break
  sleep 0.05
done
# A `latest` group must finish assignment before its first submission.
for _ in $(seq 1 200); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
sleep 0.5
# shellcheck disable=SC2086
if ! PYTHONPATH=python "${python_binary}" python/loadgen.py \
  --url http://127.0.0.1:58001 --mode "${mode}" --requests "${requests}" \
  --qps "${qps}" --seed 7 --warmup "${WARMUP:-50}" ${LOADGEN_ARGS:-}; then
  cat "${router_log}"
  cat "${gateway_log}"
  exit 1
fi
curl -fsS http://127.0.0.1:58001/v1/stats
echo
kill -TERM "${router}"
wait "${router}" || true
tail -n 1 "${router_log}"
