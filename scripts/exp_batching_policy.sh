#!/usr/bin/env bash
# Experiment 1: static (request-level) vs continuous (iteration-level)
# batching on the same ContinuousEngine, same Poisson arrivals, same cost
# model. Only the admission policy changes. Deterministic fake clock, so the
# results are reproducible bit for bit and need no external services.
#
#   scripts/exp_batching_policy.sh [BENCH_BINARY] [REQUESTS]
#
# Workloads: "short" (outputs 1-16 tokens, the historical P7 workload) and
# "mixed" (80% 1-16, 20% 64-512 tokens). Cost models: "p7" (decode 50 us +
# 10 us/sequence) and "memory-bound" (decode 500 us + 10 us/sequence); both
# charge prefill 100 us + 10 us/prompt token. Prints one table per pair.
set -euo pipefail
bench="${1:-build/default/cpp/bench/infersched_local_bench}"
requests="${2:-10000}"
# Offered loads per (workload, cost model), chosen to span light load through
# saturation for each combination.
declare -A sweeps=(
  [short/p7]="250 500 750 1000 1250"
  [short/memory-bound]="250 500 1000 2000"
  [mixed/p7]="100 200 300 400 500 600 700"
  [mixed/memory-bound]="50 100 200 300 400 500"
)
for workload in short mixed; do
  for model in p7 memory-bound; do
    for qps in ${sweeps[$workload/$model]}; do
      for policy in static continuous; do
        "${bench}" --mode open_loop --requests "${requests}" --qps "${qps}" --seed 7 \
          --cost-model "${model}" --policy "${policy}" --workload "${workload}"
      done
    done
  done
done | python3 -c "$(cat <<'PY'
import json, sys
rows = [json.loads(line) for line in sys.stdin if line.startswith("{")]
for workload, model in [(w, m) for w in ("short", "mixed") for m in ("p7", "memory-bound")]:
    print(f"\n### Workload: {workload}, cost model: {model}\n")
    print("| Offered QPS | Policy | Achieved QPS | Busy % | Mean batch | Decode steps "
          "| Decode us / useful token | Useful tokens / busy s | E2E p50 | E2E p99 |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["cost_model"] != model or r["workload"] != workload:
            continue
        print(f"| {r['offered_qps']:,} | {r['policy']} | {r['achieved_qps']:,.1f} "
              f"| {100 * r['engine_utilization']:.1f}% | {r['mean_batch_occupancy']:.2f} "
              f"| {r['decode_iterations']:,} | {r['decode_us_per_useful_token']:.1f} "
              f"| {r['useful_tokens_per_busy_s']:,.0f} | {r['e2e_p50_exact_ms']:.1f} ms "
              f"| {r['e2e_p99_exact_ms']:.1f} ms |")
    # Capacity: highest offered load still completed at >= 99% of offered rate.
    for policy in ("static", "continuous"):
        sustained = [r["offered_qps"] for r in rows
                     if r["cost_model"] == model and r["workload"] == workload
                     and r["policy"] == policy and r["achieved_qps"] >= 0.99 * r["offered_qps"]]
        print(f"\n{policy} sustains up to {max(sustained) if sustained else 0:,} offered QPS "
              "(achieved >= 99% of offered) in this sweep.", end="")
    print()
PY
)"
