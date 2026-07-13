#!/usr/bin/env bash
set -euo pipefail

requests="${1:-100000}"
binary="${2:-build/default/cpp/bench/infersched_local_bench}"
baseline_requests="${requests}"
if (( baseline_requests > 10000 )); then
  baseline_requests=10000
fi

"${binary}" --mode scheduler --engine continuous --requests "${requests}"
"${binary}" --mode scheduler --engine static --requests "${baseline_requests}"
"${binary}" --mode scheduler --engine continuous --requests "${baseline_requests}"
"${binary}" --mode queue --queue mutex --requests "${requests}" --producers 4
"${binary}" --mode queue --queue mpsc --requests "${requests}" --producers 4
"${binary}" --mode registry --registry global --requests "${requests}" --producers 4
"${binary}" --mode registry --registry sharded --requests "${requests}" --producers 4
