#!/usr/bin/env bash
set -euo pipefail

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
client_binary="${3:?client binary required}"
group="infersched-p6-chaos-$$"
marker="${TMPDIR:-/tmp}/infersched-p6-published-$$"
router1_log="${TMPDIR:-/tmp}/infersched-p6-router1-$$.log"
router2_log="${TMPDIR:-/tmp}/infersched-p6-router2-$$.log"
ready1="${TMPDIR:-/tmp}/infersched-p6-router1-ready-$$"
ready2="${TMPDIR:-/tmp}/infersched-p6-router2-ready-$$"
rm -f "${ready1}" "${ready2}"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  rm -f "${marker}" "${ready1}" "${ready2}"
}
trap cleanup EXIT

"${router_binary}" --listen 127.0.0.1:55211 --router-id p6-router-1 \
  --consumer-group "${group}" --requests 100000 --run-seconds 40 \
  --ready-file "${ready1}" \
  >"${router1_log}" 2>&1 &
router1=$!; pids+=("${router1}")
"${router_binary}" --listen 127.0.0.1:55212 --router-id p6-router-2 \
  --consumer-group "${group}" --requests 100000 --run-seconds 40 \
  --ready-file "${ready2}" \
  >"${router2_log}" 2>&1 &
router2=$!; pids+=("${router2}")
sleep 0.3
"${engine_binary}" --listen 127.0.0.1:55221 --router 127.0.0.1:55211 \
  --engine-id p6-engine-1 --incarnation p6-engine-1a --batch-delay-ms 500 \
  >/dev/null 2>&1 &
engine1=$!; pids+=("${engine1}")
"${engine_binary}" --listen 127.0.0.1:55222 --router 127.0.0.1:55212 \
  --engine-id p6-engine-2 --incarnation p6-engine-2a --batch-delay-ms 20 \
  >/dev/null 2>&1 &
engine2=$!; pids+=("${engine2}")
for _ in $(seq 1 300); do
  [[ -f "${ready1}" && -f "${ready2}" ]] && break
  kill -0 "${router1}" 2>/dev/null || break
  kill -0 "${router2}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready1}" && -f "${ready2}" ]] || {
  cat "${router1_log}" "${router2_log}"
  exit 1
}
"${client_binary}" --mode batch --count 24 --published-marker "${marker}" &
client=$!; pids+=("${client}")
for _ in $(seq 1 100); do
  [[ -f "${marker}" ]] && break
  sleep 0.05
done
[[ -f "${marker}" ]] || { echo "client did not publish"; exit 1; }
sleep 0.2
kill -9 "${router1}" "${engine1}" 2>/dev/null || true
wait "${router1}" 2>/dev/null || true
wait "${engine1}" 2>/dev/null || true
docker exec infersched-redis redis-cli FLUSHALL >/dev/null

if ! wait "${client}"; then
  cat "${router1_log}"
  cat "${router2_log}"
  exit 1
fi
prefix=$(<"${marker}")
read -r completed result_rows <<<"$(docker exec infersched-postgres psql \
  -U infersched -d infersched -At -F ' ' -c \
  "SELECT count(*) FILTER (WHERE state = 'COMPLETED'),
          count(DISTINCT outbox.message_key)
     FROM inference_requests
     LEFT JOIN outbox ON outbox.message_key = inference_requests.request_id
                     AND outbox.topic = 'inference.results'
    WHERE inference_requests.request_id LIKE '${prefix}-%';")"
[[ "${completed}" == 24 && "${result_rows}" == 24 ]] || {
  echo "durable invariant failed: completed=${completed} results=${result_rows}"
  exit 1
}
echo '{"multi_router_recovery":true,"lost_requests":0}'
