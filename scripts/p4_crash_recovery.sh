#!/usr/bin/env bash
set -euo pipefail

router_binary="${1:?router binary required}"
engine_binary="${2:?engine binary required}"
router_endpoint="127.0.0.1:55171"
engine_endpoint="127.0.0.1:55172"
replacement_endpoint="127.0.0.1:55173"
marker="${TMPDIR:-/tmp}/infersched-p4-dispatch.marker"
router_log="${TMPDIR:-/tmp}/infersched-p4-retry-router.log"
engine_log="${TMPDIR:-/tmp}/infersched-p4-retry-engine.log"
rm -f "${marker}"

"${router_binary}" --listen "${router_endpoint}" --retry-demo true \
  --dispatch-marker "${marker}" >"${router_log}" 2>&1 &
router_pid=$!
engine_pid=""
cleanup() {
  if [[ -n "${engine_pid}" ]]; then kill "${engine_pid}" 2>/dev/null || true; fi
  kill "${router_pid}" 2>/dev/null || true
  rm -f "${marker}"
}
trap cleanup EXIT

sleep 0.2
"${engine_binary}" --listen "${replacement_endpoint}" --router "${router_endpoint}" \
  --engine-id crash-engine --incarnation before-crash --batch-delay-ms 2000 \
  >"${engine_log}" 2>&1 &
engine_pid=$!

for _ in $(seq 1 100); do
  [[ -f "${marker}" ]] && break
  sleep 0.02
done
[[ -f "${marker}" ]] || { echo "dispatch marker not created"; exit 1; }
kill -9 "${engine_pid}"
wait "${engine_pid}" 2>/dev/null || true

"${engine_binary}" --listen "${engine_endpoint}" --router "${router_endpoint}" \
  --engine-id crash-engine --incarnation after-crash >"${engine_log}" 2>&1 &
engine_pid=$!

if ! wait "${router_pid}"; then
  cat "${router_log}"
  cat "${engine_log}"
  exit 1
fi
cat "${router_log}"
