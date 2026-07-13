#!/usr/bin/env bash
set -euo pipefail

router_binary="${1:?durable router binary required}"
engine_binary="${2:?engine binary required}"
client_binary="${3:?client binary required}"
router_endpoint="127.0.0.1:55201"
engine_endpoint="127.0.0.1:55202"
request_file="${TMPDIR:-/tmp}/infersched-p5-recovery-id"
router_log="${TMPDIR:-/tmp}/infersched-p5-recovery-router.log"
engine_log="${TMPDIR:-/tmp}/infersched-p5-recovery-engine.log"
rm -f "${request_file}"

"${client_binary}" --mode seed --request-file "${request_file}"
"${client_binary}" --mode await --request-file "${request_file}" &
client_pid=$!
"${router_binary}" --listen "${router_endpoint}" --requests 1 >"${router_log}" 2>&1 &
router_pid=$!
engine_pid=""
cleanup() {
  if [[ -n "${engine_pid}" ]]; then kill "${engine_pid}" 2>/dev/null || true; fi
  kill "${router_pid}" "${client_pid}" 2>/dev/null || true
  rm -f "${request_file}"
}
trap cleanup EXIT
sleep 0.3
"${engine_binary}" --listen "${engine_endpoint}" --router "${router_endpoint}" \
  --engine-id p5-recovery-engine --incarnation p5-recovery-incarnation \
  >"${engine_log}" 2>&1 &
engine_pid=$!

wait "${client_pid}"
if ! wait "${router_pid}"; then
  cat "${router_log}"
  cat "${engine_log}"
  exit 1
fi
cat "${router_log}"
