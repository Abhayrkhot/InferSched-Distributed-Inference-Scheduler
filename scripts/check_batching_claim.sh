#!/usr/bin/env bash
# Regression guard for the documented batching result (docs/p9-compute-efficiency.md):
# on the mixed output-length workload with the memory-bound cost model at 300
# offered QPS, continuous batching must sustain the load while request-level
# (static) batching saturates, and must spend less decode compute per useful
# token. Deterministic fake clock, so thresholds are exact, not statistical.
set -euo pipefail
bench="${1:?benchmark binary required}"
run() {
  "${bench}" --mode open_loop --requests 10000 --qps 300 --seed 7 \
    --cost-model memory-bound --workload mixed --policy "$1"
}
STATIC="$(run static)" CONTINUOUS="$(run continuous)" python3 - <<'PY'
import json, os, sys
static = json.loads(os.environ["STATIC"])
continuous = json.loads(os.environ["CONTINUOUS"])
checks = {
    "continuous sustains 300 QPS": continuous["achieved_qps"] >= 0.99 * 300,
    "static saturates below 70% of offered": static["achieved_qps"] < 0.7 * 300,
    "continuous batches are fuller": continuous["mean_batch_occupancy"]
        > 1.5 * static["mean_batch_occupancy"],
    "continuous spends less decode compute per useful token":
        continuous["decode_us_per_useful_token"] < 0.7 * static["decode_us_per_useful_token"],
    "both complete with no KV leak": all(r["complete"] and r["zero_kv_leak"]
                                         for r in (static, continuous)),
}
print(json.dumps({"checks": checks,
                  "static": {k: static[k] for k in ("achieved_qps", "mean_batch_occupancy",
                                                    "decode_us_per_useful_token")},
                  "continuous": {k: continuous[k] for k in ("achieved_qps",
                                                            "mean_batch_occupancy",
                                                            "decode_us_per_useful_token")}}))
sys.exit(0 if all(checks.values()) else 1)
PY
