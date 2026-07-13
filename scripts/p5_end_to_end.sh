#!/usr/bin/env bash
set -euo pipefail

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
client_binary="${3:?client binary required}"
router_endpoint="127.0.0.1:55191"
engine_endpoint="127.0.0.1:55192"
router_log="${TMPDIR:-/tmp}/infersched-p5-router.log"
engine_log="${TMPDIR:-/tmp}/infersched-p5-engine.log"
group="infersched-p5-e2e-$$"
ready="${TMPDIR:-/tmp}/infersched-p5-ready-$$"
rm -f "${ready}"

"${router_binary}" --listen "${router_endpoint}" --requests 1 \
  --consumer-group "${group}" --offset-reset latest --ready-file "${ready}" \
  >"${router_log}" 2>&1 &
router_pid=$!
engine_pid=""
cleanup() {
  if [[ -n "${engine_pid}" ]]; then kill "${engine_pid}" 2>/dev/null || true; fi
  kill "${router_pid}" 2>/dev/null || true
  rm -f "${ready}"
}
trap cleanup EXIT
sleep 0.3
"${engine_binary}" --listen "${engine_endpoint}" --router "${router_endpoint}" \
  --engine-id p5-engine --incarnation p5-incarnation >"${engine_log}" 2>&1 &
engine_pid=$!
for _ in $(seq 1 200); do
  [[ -f "${ready}" ]] && break
  kill -0 "${router_pid}" 2>/dev/null || break
  sleep 0.05
done
[[ -f "${ready}" ]] || { cat "${router_log}"; exit 1; }
"${client_binary}"
if ! wait "${router_pid}"; then
  cat "${router_log}"
  cat "${engine_log}"
  exit 1
fi
cat "${router_log}"
