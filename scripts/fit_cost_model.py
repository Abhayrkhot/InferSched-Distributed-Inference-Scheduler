#!/usr/bin/env python3
"""Fit InferSched's cost model to measured llama.cpp step timings.

Input: step traces from `infersched_real_serving_bench --step-trace`, one JSON
object per scheduler iteration: prompt tokens computed, sequences decoded, the
KV positions those decode tokens attend to, and the measured step time (us).

Fits, by ordinary least squares on every step:

  step_us = fixed + per_seq * decode + per_prompt_token * prompt [+ per_ctx * context]

The first three terms are exactly the phenomenological model the scheduler
simulates (cost_model.hpp: decode_fixed + decode_per_sequence * n, and
prefill_per_token * uncached tokens); the context term shows how much of the
error comes from attention, which the model leaves out. Pure Python, no numpy.
"""

from __future__ import annotations

import argparse
import json
import statistics


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(size):
            if row != column and augmented[column][column] != 0:
                factor = augmented[row][column] / augmented[column][column]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [augmented[index][size] / augmented[index][index] for index in range(size)]


def least_squares(features: list[list[float]], target: list[float]) -> list[float]:
    width = len(features[0])
    normal = [[sum(row[i] * row[j] for row in features) for j in range(width)]
              for i in range(width)]
    rhs = [sum(row[i] * value for row, value in zip(features, target)) for i in range(width)]
    return solve(normal, rhs)


def quality(features: list[list[float]], target: list[float], coefficients: list[float]) -> dict:
    predicted = [sum(c * x for c, x in zip(coefficients, row)) for row in features]
    mean = statistics.fmean(target)
    total = sum((value - mean) ** 2 for value in target)
    residual = sum((value - guess) ** 2 for value, guess in zip(target, predicted))
    errors = sorted(abs(guess - value) / value for value, guess in zip(target, predicted) if value)
    return {"r2": round(1 - residual / total, 4) if total else 1.0,
            "median_abs_pct_error": round(100 * errors[len(errors) // 2], 1),
            "p90_abs_pct_error": round(100 * errors[int(len(errors) * 0.9)], 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    steps = []
    for path in args.traces:
        with open(path, encoding="utf-8") as handle:
            steps += [json.loads(line) for line in handle if line.strip()]
    target = [float(step["us"]) for step in steps]
    base = [[1.0, float(step["decode"]), float(step["prompt"])] for step in steps]
    with_context = [row + [float(step["context"])] for row, step in zip(base, steps)]
    fit = least_squares(base, target)
    fit_context = least_squares(with_context, target)
    decode_only = [step for step in steps if step["prompt"] == 0]
    result = {
        "steps": len(steps),
        "decode_only_steps": len(decode_only),
        "batch_sizes_seen": [min(s["decode"] for s in steps), max(s["decode"] for s in steps)],
        "cost_model": {
            "fixed_us": round(fit[0], 1),
            "per_sequence_us": round(fit[1], 2),
            "per_prompt_token_us": round(fit[2], 3),
            **quality(base, target, fit),
        },
        "with_attention_term": {
            "fixed_us": round(fit_context[0], 1),
            "per_sequence_us": round(fit_context[1], 2),
            "per_prompt_token_us": round(fit_context[2], 3),
            "per_context_token_us": round(fit_context[3], 5),
            **quality(with_context, target, fit_context),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
