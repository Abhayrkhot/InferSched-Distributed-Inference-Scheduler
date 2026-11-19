#!/usr/bin/env bash
# Router-level throughput without HTTP: a C++ client publishes COUNT requests
# to Kafka in one burst and waits until every result is on inference.results.
#
#   scripts/run_router_burst.sh [BIN_DIR] [ENGINES] [COUNT] [extra router flags...]
#   scripts/run_router_burst.sh build/default/cpp 2 5000 --workers 256
#
# Engines run with zero simulated compute, so this isolates the Router /
# Kafka / PostgreSQL / gRPC coordination path. Requires `docker compose up -d`.
set -euo pipefail
bin="${1:-build/default/cpp}"
engines="${2:-1}"
count="${3:-1000}"
shift $(( $# < 3 ? $# : 3 ))
uid="$$-${RANDOM}${RANDOM}"
tmp="${TMPDIR:-/tmp}"
ready="${tmp}/infersched-burst-ready-${uid}"
router_log="${tmp}/infersched-burst-router-${uid}.log"
client_log="${tmp}/infersched-burst-client-${uid}.log"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  rm -f "${ready}"
}
trap cleanup EXIT

"${bin}/infersched_durable_router" --listen 127.0.0.1:55401 --router-id "burst-${uid}" \
  --consumer-group "infersched-burst-${uid}" --offset-reset latest --ready-file "${ready}" \
  --requests 100000000 --run-seconds 300 "$@" >"${router_log}" 2>&1 &
pids+=("$!")
sleep 0.3
for index in $(seq 1 "${engines}"); do
  "${bin}/infersched_engine" --listen "127.0.0.1:$((55410 + index))" --router 127.0.0.1:55401 \
    --engine-id "burst-engine-${index}" --incarnation "burst-${uid}-${index}" >/dev/null 2>&1 &
  pids+=("$!")
done
for _ in $(seq 1 300); do
  [[ -f "${ready}" ]] && break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
sleep 0.5  # Engines registered.

start=$(date +%s%N)
"${bin}/infersched_p5_client" --mode batch --count "${count}" >"${client_log}" 2>&1 || true
end=$(date +%s%N)
completed=$(grep -o '"completed":[0-9]*' "${client_log}" | cut -d: -f2)
completed="${completed:-0}"
elapsed_ms=$(( (end - start) / 1000000 ))
echo "{\"engines\":${engines},\"requests\":${count},\"completed\":${completed},\"wall_ms\":${elapsed_ms},\"req_per_s\":$(( completed * 1000 / (elapsed_ms > 0 ? elapsed_ms : 1) ))}"
[[ "${completed}" == "${count}" ]]
