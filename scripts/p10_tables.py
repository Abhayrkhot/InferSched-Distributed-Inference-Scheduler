#!/usr/bin/env python3
"""Markdown tables for docs/p10-real-model-serving.md from results.jsonl files
written by scripts/exp_real_serving.py. Every number in the doc's tables comes
from here; nothing is typed in by hand.

    python3 scripts/p10_tables.py docs/p10-results/<run>/results.jsonl [...]
"""

from __future__ import annotations

import argparse
import json


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


def number(value: float | int | None, digits: int = 0) -> str:
    if value is None:
        return "–"
    return f"{value:,.{digits}f}"


def joules_per_token(row: dict) -> float | None:
    power, duration, tokens = row.get("gpu_power_w"), row.get("duration_s"), row.get("output_tokens")
    if not power or not duration or not tokens:
        return None
    return power * duration / tokens


def table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def batching(rows: list[dict]) -> str:
    order = {"burst-serial": 0, "burst-static": 1, "burst-continuous": 2,
             "burst-continuous-graphs-on": 3}
    picked = sorted((r for r in rows if r["phase"] == "batching"), key=lambda r: order[r["label"]])
    names = {"burst-serial": "One request at a time", "burst-static": "Request-level batching",
             "burst-continuous": "Continuous batching (InferSched)",
             "burst-continuous-graphs-on": "Continuous, CUDA graphs on"}
    return table(
        ["Policy", "Output tok/s", "Mean batch", "GPU busy (NVML)", "GPU power",
         "Energy per output token", "Wall time"],
        [[names[r["label"]], number(r["output_tok_s"]), number(r["mean_batch_occupancy"], 1),
          f"{number(r.get('gpu_util_pct'))}%", f"{number(r.get('gpu_power_w'))} W",
          f"{number(joules_per_token(r), 2)} J", f"{number(r['duration_s'], 1)} s"]
         for r in picked])


def sweep(rows: list[dict]) -> str:
    picked = sorted((r for r in rows if r["phase"] == "sweep"),
                    key=lambda r: (r["qps"], r["policy"] != "continuous"))
    return table(
        ["Offered QPS", "Policy", "Achieved req/s", "Output tok/s", "TTFT p50 / p99",
         "TPOT p50 / p99", "E2E p99", "Mean batch", "GPU busy"],
        [[number(r["qps"], 1), r["policy"], number(r["achieved_rps"], 2), number(r["output_tok_s"]),
          f"{number(r['ttft_p50_ms'])} / {number(r['ttft_p99_ms'])} ms",
          f"{number(r['tpot_p50_ms'], 1)} / {number(r['tpot_p99_ms'], 1)} ms",
          f"{number(r['e2e_p99_ms'] / 1000, 1)} s", number(r["mean_batch_occupancy"], 1),
          f"{number(r.get('gpu_util_pct'))}%"] for r in picked])


def reference(rows: list[dict]) -> str:
    ours = {r["qps"]: r for r in rows if r["phase"] == "sweep" and r["policy"] == "continuous"}
    theirs = [r for r in rows if r["phase"] == "reference"]
    out = []
    for qps in sorted({r["qps"] for r in theirs}):
        group = [("InferSched + llama.cpp", ours.get(qps))]
        group += [("llama-server" if r["server"] == "llama" else "vLLM", r)
                  for r in theirs if r["qps"] == qps]
        for name, r in group:
            if r is None:
                continue
            failed = r.get("failed", 0)
            out.append([number(qps, 1), name, number(r["achieved_rps"], 2), number(r["output_tok_s"]),
                        f"{number(r['ttft_p50_ms'])} / {number(r['ttft_p99_ms'])} ms",
                        f"{number(r['tpot_p50_ms'], 1)} / {number(r['tpot_p99_ms'], 1)} ms",
                        f"{number(r['e2e_p99_ms'] / 1000, 1)} s",
                        f"{number(r.get('gpu_util_pct'))}%",
                        str(failed)])
    return table(["Offered QPS", "System", "Achieved req/s", "Output tok/s", "TTFT p50 / p99",
                  "TPOT p50 / p99", "E2E p99", "GPU busy", "Failed"], out)


def prefix(rows: list[dict]) -> str:
    picked = sorted((r for r in rows if r["phase"] == "prefix"), key=lambda r: r["prefix_cache"])
    return table(
        ["Prefix cache", "Prompt tokens computed", "Prompt tokens reused", "TTFT p50 / p99",
         "E2E p99", "Output tok/s", "GPU busy fraction (engine)"],
        [["on" if r["prefix_cache"] else "off", number(r["prefill_tokens_computed"]),
          number(r["prefill_tokens_reused"]),
          f"{number(r['ttft_p50_ms'])} / {number(r['ttft_p99_ms'])} ms",
          f"{number(r['e2e_p99_ms'] / 1000, 2)} s", number(r["output_tok_s"]),
          number(r["busy_fraction"], 2)] for r in picked])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    rows = load(args.results)
    for title, render in (("Batching", batching), ("Sweep", sweep), ("Reference", reference),
                          ("Prefix", prefix)):
        try:
            text = render(rows)
        except (KeyError, ValueError) as error:
            text = f"(no data: {error})"
        print(f"### {title}\n\n{text}\n")


if __name__ == "__main__":
    main()
