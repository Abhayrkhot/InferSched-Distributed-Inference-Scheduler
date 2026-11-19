#!/usr/bin/env bash
# Full HTTP path: FastAPI gateway -> Kafka -> durable Router -> two Engines ->
# Postgres outbox -> Kafka -> gateway long-poll. 200 open-loop requests, half
# of them sharing prefixes; every one must complete with its own trace id, and
# the gateway's Prometheus counters must agree.
set -euo pipefail
# Unique per run even inside a fresh container, where $$ repeats.
uid="$$-${RANDOM}${RANDOM}"

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
python_binary="${3:?python binary required}"
root="${4:?repository root required}"
requests=200
group="infersched-p7-gateway-${uid}"
ready="${TMPDIR:-/tmp}/infersched-p7-gateway-ready-${uid}"
router_log="${TMPDIR:-/tmp}/infersched-p7-gateway-router-${uid}.log"
gateway_log="${TMPDIR:-/tmp}/infersched-p7-gateway-${uid}.log"
rm -f "${ready}"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  rm -f "${ready}"
}
trap cleanup EXIT

"${router_binary}" --listen 127.0.0.1:55231 --router-id p7-router \
  --consumer-group "${group}" --offset-reset latest --requests "${requests}" \
  --run-seconds 40 --ready-file "${ready}" >"${router_log}" 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
for index in 1 2; do
  "${engine_binary}" --listen "127.0.0.1:5523$((index + 1))" --router 127.0.0.1:55231 \
    --engine-id "p7-engine-${index}" --incarnation "p7-engine-${index}-${uid}" \
    --batch-delay-ms 5 >/dev/null 2>&1 &
  pids+=("$!")
done

PYTHONPATH="${root}/python" INFERSCHED_GATEWAY_CAPACITY=512 \
  "${python_binary}" -m uvicorn infersched_api.app:app \
  --host 127.0.0.1 --port 58000 >"${gateway_log}" 2>&1 &
gateway=$!; pids+=("${gateway}")
# Ready = results consumer positioned (the gateway rejects submissions before).
for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:58000/readyz >/dev/null 2>&1 && break
  sleep 0.05
done
curl -fsS http://127.0.0.1:58000/readyz >/dev/null
for _ in $(seq 1 200); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
sleep 0.4  # Both Engines registered.
if ! PYTHONPATH="${root}/python" "${python_binary}" "${root}/python/loadgen.py" \
  --url http://127.0.0.1:58000 --mode open --requests "${requests}" --qps 100 \
  --seed 7 --prefixes 4 --prefix-tokens 128 --prefix-fraction 0.5 --require-all; then
  cat "${router_log}" "${gateway_log}"
  exit 1
fi
metrics=$(curl -fsS http://127.0.0.1:58000/metrics)
grep -qx "infersched_gateway_requests_completed_total ${requests}" <<<"${metrics}" || {
  echo "gateway metrics disagree:"; echo "${metrics}"; exit 1;
}
grep -qx "infersched_gateway_queue_depth 0" <<<"${metrics}"
wait "${router}" || { cat "${router_log}"; exit 1; }
tail -n 1 "${router_log}"
