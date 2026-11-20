#!/usr/bin/env python3
"""Fleet compute-efficiency experiments on real processes.

Each experiment changes one scheduling decision and holds everything else
fixed. Engines execute the phenomenological cost model in wall-clock time
(--time-scale = hardware speed factor), so "compute" below means modeled
accelerator time, not GPU kernels.

  routing       first | round-robin | least-loaded on a heterogeneous fleet
  affinity      Redis prefix hints off vs on, shared-prefix workload
  cancellation  fenced cancellation off vs on when slow attempts time out

Usage: scripts/exp_fleet_efficiency.py [routing affinity cancellation] [--json FILE]
Requires `docker compose up -d` and a default build. Prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Decode is dominated by a per-step fixed cost; prefill is per prompt token.
COST = ("--prefill-fixed-us 100 --prefill-per-token-us 10 "
        "--decode-fixed-us 300 --decode-per-seq-us 20")

EXPERIMENTS = {
    "routing": {
        "title": "Routing policy, heterogeneous fleet (engine-4 is 3x slower), 400 QPS",
        "common": {"ENGINE_SCALES": "1 1 1 3", "ENGINE_ARGS": COST,
                   "REQUESTS": "4000", "QPS": "400"},
        "variants": [
            ("first", {"ROUTER_ARGS": "--routing first --redis none"}),
            ("round-robin", {"ROUTER_ARGS": "--routing round-robin --redis none"}),
            ("least-loaded", {"ROUTER_ARGS": "--routing least-loaded --redis none"}),
        ],
    },
    "affinity": {
        "title": "Prefix affinity, 4 Engines, 64 shared 512-token prefixes, 200 QPS",
        "common": {"ENGINE_SCALES": "1 1 1 1",
                   "ENGINE_ARGS": COST + " --kv-blocks 2048 --prefix-cache-entries 1024",
                   "LOADGEN_ARGS": "--prefixes 64 --prefix-tokens 512 --prefix-fraction 1.0",
                   "REQUESTS": "3000", "QPS": "200"},
        "variants": [
            ("hints off", {"ROUTER_ARGS": "--routing least-loaded --redis none"}),
            ("hints on", {"ROUTER_ARGS": "--routing least-loaded --redis 127.0.0.1:6379"}),
        ],
    },
    "cancellation": {
        "title": ("Cancellation, 3 Engines, 10% of requests need 800 output tokens and miss a "
                  "60 ms dispatch deadline (2 attempts each), 150 QPS"),
        "common": {"ENGINE_SCALES": "1 1 1",
                   "ROUTER_ARGS": ("--routing least-loaded --redis none "
                                   "--dispatch-timeout-ms 60 --max-attempts 2"),
                   "LOADGEN_ARGS": "--long-fraction 0.1 --long-output-tokens 800",
                   "REQUESTS": "3000", "QPS": "150"},
        "variants": [
            ("cancellation off", {"ENGINE_ARGS": COST + " --cancellation off"}),
            ("cancellation on", {"ENGINE_ARGS": COST + " --cancellation on"}),
        ],
    },
}


def run(env_overrides: dict[str, str]) -> dict:
    env = dict(os.environ, **env_overrides)
    output = subprocess.run(["scripts/run_fleet_experiment.sh"], env=env, check=True,
                            capture_output=True, text=True).stdout
    return json.loads(output.strip().splitlines()[-1])


def summarize(result: dict) -> dict:
    engines = result["engines"]
    # Hardware time = modeled time x the Engine's slowdown factor.
    hardware_ms = [e["modeled_busy_ms"] * e["speed_factor"] for e in engines]
    useful = sum(e["useful_output_tokens"] for e in engines)
    wasted = sum(e["wasted_decode_tokens"] + e["abandoned_output_tokens"] for e in engines)
    slots = sum(e["decode_token_slots"] for e in engines)
    iterations = sum(e["decode_iterations"] for e in engines)
    busy = [e["busy_fraction"] for e in engines]
    load = result["loadgen"]
    return {
        "completed": load["completed"], "errors": load["errors"] + load["timed_out"],
        "failed": load["failed"],
        "achieved_qps": load["achieved_qps"],
        "p50": load["e2e_p50_ms"], "p99": load["e2e_p99_ms"],
        "hardware_busy_s": sum(hardware_ms) / 1000,
        "useful_tokens": useful,
        "tokens_per_hw_s": useful / (sum(hardware_ms) / 1000) if sum(hardware_ms) else 0.0,
        "occupancy": slots / iterations if iterations else 0.0,
        "busy": busy, "busy_spread": max(busy) - min(busy),
        "wasted_pct": 100 * wasted / (useful + wasted) if useful + wasted else 0.0,
        "prefill_computed": sum(e["prefill_tokens_computed"] for e in engines),
        "prefill_reused": sum(e["prefill_tokens_reused"] for e in engines),
        "retries": result["router"]["retries"],
        "affinity": result["router"]["affinity_dispatches"],
        "dispatch": result["router"]["dispatch_by_engine"],
    }


def table(name: str, rows: list[tuple[str, dict]]) -> str:
    lines = [f"### {EXPERIMENTS[name]['title']}", ""]
    header = ("| Variant | Completed | Achieved QPS | e2e p50 | e2e p99 | Hardware busy | "
              "Useful tokens / hardware s | Mean batch | Per-Engine busy | ")
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---|"
    if name == "affinity":
        header += "Prefill tokens computed | Prefill reused | Hint dispatches |"
        divider += "---:|---:|---:|"
    elif name == "cancellation":
        header += "Deadline failures | Wasted tokens | Retries |"
        divider += "---:|---:|---:|"
    else:
        header += "Dispatch split |"
        divider += "---|"
    lines += [header, divider]
    for label, s in rows:
        busy = " / ".join(f"{100 * b:.0f}%" for b in s["busy"])
        line = (f"| {label} | {s['completed']:,} | {s['achieved_qps']:.1f} | {s['p50']:.1f} ms "
                f"| {s['p99']:.1f} ms | {s['hardware_busy_s']:.2f} s | {s['tokens_per_hw_s']:,.0f} "
                f"| {s['occupancy']:.2f} | {busy} | ")
        if name == "affinity":
            line += (f"{s['prefill_computed']:,} | {s['prefill_reused']:,} | {s['affinity']:,} |")
        elif name == "cancellation":
            line += f"{s['failed']:,} | {s['wasted_pct']:.1f}% | {s['retries']:,} |"
        else:
            split = " / ".join(str(s["dispatch"].get(f"engine-{i}", 0))
                               for i in range(1, len(s["busy"]) + 1))
            line += f"{split} |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument("--json", help="also write raw results to this file")
    args = parser.parse_args()
    raw: dict[str, list] = {}
    for name in args.experiments:
        spec = EXPERIMENTS[name]
        rows = []
        for label, overrides in spec["variants"]:
            print(f"running {name}: {label}", file=sys.stderr, flush=True)
            result = run({**spec["common"], **overrides})
            raw.setdefault(name, []).append({"variant": label, **result})
            rows.append((label, summarize(result)))
        print(table(name, rows), end="\n\n", flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(raw, handle, indent=1)


if __name__ == "__main__":
    main()
