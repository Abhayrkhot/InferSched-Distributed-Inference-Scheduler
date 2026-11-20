#!/usr/bin/env bash
# One fleet-efficiency run: gateway -> Kafka -> one durable Router -> N Engines
# whose time is paced by the cost model, then every process's counters.
#
# Environment:
#   ENGINE_SCALES="1 1 1 3"  one Engine per entry; each is that Engine's
#                            --time-scale (3 = three times slower hardware)
#   ENGINE_ARGS=""           extra flags for every Engine (cost model, policy, ...)
#   ROUTER_ARGS=""           extra flags for the Router (--routing, --redis none, ...)
#   LOADGEN_ARGS=""          extra loadgen.py flags (prefix workload, ...)
#   REQUESTS=3000 QPS=300    open-loop load
#   WARMUP=5                 unmeasured requests first (Kafka's first fetch after
#                            partition assignment is slow); they do reach Engines
#
# Prints one JSON object: {"loadgen": ..., "router": ..., "engines": [...]}.
set -euo pipefail
uid="$$-${RANDOM}${RANDOM}"
router_binary="${ROUTER_BINARY:-build/default/cpp/infersched_durable_router}"
engine_binary="${ENGINE_BINARY:-build/default/cpp/infersched_engine}"
python_binary="${PYTHON_BINARY:-.venv/bin/python}"
requests="${REQUESTS:-3000}"
qps="${QPS:-300}"
read -r -a scales <<<"${ENGINE_SCALES:-1 1 1 3}"
tmp="${TMPDIR:-/tmp}/infersched-fleet-${uid}"
mkdir -p "${tmp}"
pids=()
engine_pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -9 "${pid}" 2>/dev/null || true; done
  rm -rf "${tmp}"
}
trap cleanup EXIT

# shellcheck disable=SC2086
"${router_binary}" --listen 127.0.0.1:55501 --router-id "fleet-${uid}" \
  --consumer-group "infersched-fleet-${uid}" --offset-reset latest \
  --ready-file "${tmp}/ready" ${ROUTER_ARGS:-} >"${tmp}/router.log" 2>&1 &
router=$!; pids+=("${router}")
sleep 0.3
for index in "${!scales[@]}"; do
  # shellcheck disable=SC2086
  "${engine_binary}" --listen "127.0.0.1:$((55510 + index))" --router 127.0.0.1:55501 \
    --engine-id "engine-$((index + 1))" --incarnation "fleet-${uid}-${index}" \
    --time-scale "${scales[$index]}" ${ENGINE_ARGS:-} >"${tmp}/engine-${index}.log" 2>&1 &
  engine_pids+=("$!"); pids+=("$!")
done
PYTHONPATH=python INFERSCHED_GATEWAY_CAPACITY=8192 INFERSCHED_TRACES=false \
  "${python_binary}" -m uvicorn infersched_api.app:app \
  --host 127.0.0.1 --port 58011 --log-level warning >"${tmp}/gateway.log" 2>&1 &
pids+=("$!")
for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:58011/readyz >/dev/null 2>&1 && break
  sleep 0.05
done
for _ in $(seq 1 300); do
  [[ -f "${tmp}/ready" ]] && break
  kill -0 "${router}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${tmp}/ready" ]] || { cat "${tmp}/router.log" >&2; exit 1; }
sleep 0.6  # Every Engine registered and heartbeating.

# shellcheck disable=SC2086
PYTHONPATH=python "${python_binary}" python/loadgen.py --url http://127.0.0.1:58011 \
  --mode open --requests "${requests}" --qps "${qps}" --seed 7 --warmup "${WARMUP:-5}" \
  ${LOADGEN_ARGS:-} >"${tmp}/loadgen.json" || { cat "${tmp}/router.log" >&2; exit 1; }

kill -TERM "${router}"; wait "${router}" || true
for pid in "${engine_pids[@]}"; do kill -TERM "${pid}"; done
for pid in "${engine_pids[@]}"; do wait "${pid}" || true; done
TMPDIR_RUN="${tmp}" ENGINES="${#scales[@]}" SCALES="${scales[*]}" python3 -c '
import json, os
tmp = os.environ["TMPDIR_RUN"]
def last_json(path):
    lines = [l for l in open(path) if l.startswith("{")]
    return json.loads(lines[-1]) if lines else None
engines = []
for index, scale in enumerate(os.environ["SCALES"].split()):
    stats = last_json(f"{tmp}/engine-{index}.log") or {}
    stats["speed_factor"] = float(scale)
    engines.append(stats)
print(json.dumps({"loadgen": last_json(f"{tmp}/loadgen.json"),
                  "router": last_json(f"{tmp}/router.log"), "engines": engines}))
'
