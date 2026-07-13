#!/usr/bin/env bash
set -euo pipefail

router_binary="${1:?router binary required}"
engine_binary="${2:?engine binary required}"
router_endpoint="127.0.0.1:55181"
engine_endpoint="127.0.0.1:55182"
router_log="${TMPDIR:-/tmp}/infersched-p4-overlap-router.log"
engine_log="${TMPDIR:-/tmp}/infersched-p4-overlap-engine.log"

"${router_binary}" --listen "${router_endpoint}" --overlapping-retry-demo true \
  >"${router_log}" 2>&1 &
router_pid=$!
engine_pid=""
cleanup() {
  if [[ -n "${engine_pid}" ]]; then kill "${engine_pid}" 2>/dev/null || true; fi
  kill "${router_pid}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 0.2
"${engine_binary}" --listen "${engine_endpoint}" --router "${router_endpoint}" \
  --engine-id overlap-engine --incarnation overlap-1 --batch-delay-ms 500 \
  >"${engine_log}" 2>&1 &
engine_pid=$!

if ! wait "${router_pid}"; then
  cat "${router_log}"
  cat "${engine_log}"
  exit 1
fi
cat "${router_log}"
