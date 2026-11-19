#!/usr/bin/env bash
# Redis prefix affinity across a two-Engine pool.
#
# Wave 1 warms one Engine per prefix and records Redis hints. Wave 2 must be
# routed by those hints (affinity dispatches) and reuse KV prefix blocks.
# Redis is then flushed: wave 3 must still complete, because hints sit
# outside the correctness boundary.
set -euo pipefail
# Unique per run even inside a fresh container, where $$ repeats.
uid="$$-${RANDOM}${RANDOM}"

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
client_binary="${3:?client binary required}"
run="p8-affinity-${uid}"
group="infersched-${run}"
tmp="${TMPDIR:-/tmp}"
ready="${tmp}/${run}-ready"
router_log="${tmp}/${run}-router.log"
rm -f "${ready}"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -9 "${pid}" 2>/dev/null || true; done
  rm -f "${ready}"
}
trap cleanup EXIT

"${router_binary}" --listen 127.0.0.1:55261 --router-id p8-affinity-router \
  --consumer-group "${group}" --offset-reset latest --ready-file "${ready}" \
  >"${router_log}" 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
for engine in a b; do
  port=$([[ "${engine}" == a ]] && echo 55262 || echo 55263)
  "${engine_binary}" --listen "127.0.0.1:${port}" --router 127.0.0.1:55261 \
    --engine-id "p8-affinity-${engine}" --incarnation "${engine}-${uid}" \
    --batch-delay-ms 2 >/dev/null 2>&1 &
  pids+=("$!")
done
for _ in $(seq 1 300); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
sleep 0.5

wave() {
  "${client_binary}" --mode batch --count "$2" --prefixes 2 \
    --prefix-namespace "${run}" --batch-prefix "${run}-$1" --wait-seconds 30
}
wave w1 4
hints=$(docker exec infersched-redis redis-cli --scan \
  --pattern "infersched:prefix:v1:benchmark-model-v1:${run}-*" | wc -l)
[[ "${hints}" == 2 ]] || { echo "expected 2 Redis prefix hints, found ${hints}"; exit 1; }
wave w2 60
docker exec infersched-redis redis-cli FLUSHALL >/dev/null
wave w3 20

kill -TERM "${router}"
wait "${router}" || { cat "${router_log}"; exit 1; }
STATS="$(tail -n 1 "${router_log}")" python3 - <<'EOF'
import json, os, sys
stats = json.loads(os.environ["STATS"])
checks = {
    "all 84 requests resolved": stats["completed"] == 84,
    "hinted engine chosen": stats["affinity_dispatches"] >= 20,
    "engines reused prefix KV": stats["prefix_cache_hits"] >= 30,
    "redis hints were read": stats["redis"]["hits"] >= 30,
}
print(json.dumps({"affinity": checks, "router": stats}))
sys.exit(0 if all(checks.values()) else 1)
EOF
