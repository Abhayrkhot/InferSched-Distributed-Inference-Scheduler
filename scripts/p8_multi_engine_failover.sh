#!/usr/bin/env bash
# One long-running durable Router (service mode) in front of three Engines.
# One Engine is slow and is SIGKILLed while it holds in-flight attempts; the
# Router must retry those attempts on the surviving Engines.
#
# Asserts: every request COMPLETED exactly once in Postgres with one result
# outbox row, work was spread over more than one Engine, killed attempts were
# retried, trace spans reached Kafka, and SIGTERM shuts the Router down cleanly.
set -euo pipefail
# Unique per run even inside a fresh container, where $$ repeats.
uid="$$-${RANDOM}${RANDOM}"

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
client_binary="${3:?client binary required}"
count=60
run="p8-failover-${uid}"
group="infersched-${run}"
tmp="${TMPDIR:-/tmp}"
ready="${tmp}/${run}-ready"
marker="${tmp}/${run}-published"
router_log="${tmp}/${run}-router.log"
rm -f "${ready}" "${marker}"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -9 "${pid}" 2>/dev/null || true; done
  rm -f "${ready}" "${marker}"
}
trap cleanup EXIT

trace_offsets() {
  docker exec infersched-kafka /opt/kafka/bin/kafka-get-offsets.sh \
    --bootstrap-server localhost:9092 --topic inference.traces 2>/dev/null |
    awk -F: '{sum += $3} END {print sum + 0}'
}
traces_before=$(trace_offsets)

"${router_binary}" --listen 127.0.0.1:55251 --router-id p8-router \
  --consumer-group "${group}" --offset-reset latest --ready-file "${ready}" \
  --workers 32 >"${router_log}" 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
"${engine_binary}" --listen 127.0.0.1:55252 --router 127.0.0.1:55251 \
  --engine-id p8-engine-a --incarnation "a-${uid}" --batch-delay-ms 5 >/dev/null 2>&1 &
pids+=("$!")
"${engine_binary}" --listen 127.0.0.1:55253 --router 127.0.0.1:55251 \
  --engine-id p8-engine-b --incarnation "b-${uid}" --batch-delay-ms 5 >/dev/null 2>&1 &
pids+=("$!")
"${engine_binary}" --listen 127.0.0.1:55254 --router 127.0.0.1:55251 \
  --engine-id p8-engine-slow --incarnation "slow-${uid}" --batch-delay-ms 400 >/dev/null 2>&1 &
slow=$!; pids+=("${slow}")

for _ in $(seq 1 300); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
sleep 0.5  # Let all three Engines register before the burst arrives.

"${client_binary}" --mode batch --count "${count}" --batch-prefix "${run}" \
  --max-output-tokens 8 --published-marker "${marker}" --wait-seconds 40 &
client=$!; pids+=("${client}")
for _ in $(seq 1 200); do
  [[ -f "${marker}" ]] && break
  sleep 0.05
done
[[ -f "${marker}" ]] || { echo "client did not publish"; exit 1; }
sleep 0.6  # Attempts are now in flight on every Engine, including the slow one.
kill -9 "${slow}"

if ! wait "${client}"; then
  cat "${router_log}"
  exit 1
fi

read -r completed outbox_rows <<<"$(docker exec infersched-postgres psql \
  -U infersched -d infersched -At -F ' ' -c \
  "SELECT (SELECT count(*) FROM inference_requests
            WHERE request_id LIKE '${run}-%' AND state = 'COMPLETED'),
          (SELECT count(*) FROM outbox
            WHERE topic = 'inference.results' AND message_key LIKE '${run}-%');")"
[[ "${completed}" == "${count}" && "${outbox_rows}" == "${count}" ]] || {
  echo "exactly-once result invariant failed: completed=${completed} outbox=${outbox_rows}"
  cat "${router_log}"
  exit 1
}

kill -TERM "${router}"
if ! wait "${router}"; then
  echo "router did not shut down cleanly on SIGTERM"
  cat "${router_log}"
  exit 1
fi
stats=$(tail -n 1 "${router_log}")
traces_after=$(trace_offsets)
STATS="${stats}" SPANS=$((traces_after - traces_before)) COUNT="${count}" python3 - <<'EOF'
import json, os, sys
stats = json.loads(os.environ["STATS"])
engines = stats["dispatch_by_engine"]
checks = {
    "dispatched to >= 2 engines": len([e for e, n in engines.items() if n > 0]) >= 2,
    "slow engine received work": engines.get("p8-engine-slow", 0) > 0,
    "killed attempts were retried": stats["retries"] >= 1,
    "no terminal failures": stats["failed_terminal"] == 0,
    "spans reached kafka": int(os.environ["SPANS"]) >= int(os.environ["COUNT"]),
}
print(json.dumps({"failover": checks, "router": stats}))
sys.exit(0 if all(checks.values()) else 1)
EOF
