#!/usr/bin/env python3
"""Open- or closed-loop load generator for the InferSched gateway.

Open loop: arrivals follow a Poisson process at --qps and every request runs as
its own task, so a slow system cannot slow the arrival rate (no coordinated
omission). Closed loop: --concurrency clients each submit, wait, repeat.
Results are awaited with the gateway's long-poll (`wait_ms`) rather than
client-side busy polling.

The load is split across --processes worker processes (each an independent
Poisson stream at qps/N, which merge into a Poisson stream at qps). A single
asyncio + httpx process saturates its CPU near ~100 req/s of POST + long-poll
and then inflates the latencies it reports; `client_cpu_saturated` flags that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import random
import time

import httpx


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(int((len(values) - 1) * q), len(values) - 1)] if values else 0


def build_body(rng: random.Random, args: argparse.Namespace) -> dict:
    body = {
        "prompt_tokens": rng.randint(args.prompt_min, args.prompt_max),
        "max_output_tokens": rng.randint(args.output_min, args.output_max),
        "priority": 0 if rng.random() < 0.1 else 1,
        "seed": rng.randrange(1 << 31),
    }
    if args.long_fraction > 0 and rng.random() < args.long_fraction:
        body["max_output_tokens"] = (
            rng.randint(args.long_output_tokens, args.long_output_max)
            if args.long_output_max > args.long_output_tokens else args.long_output_tokens)
    if args.prefixes > 0 and rng.random() < args.prefix_fraction:
        body["prefix_id"] = f"prefix-{rng.randrange(args.prefixes)}"
        body["prefix_tokens"] = args.prefix_tokens
        body["prompt_tokens"] = args.prefix_tokens + rng.randint(16, 64)
    return body


async def one_request(
    client: httpx.AsyncClient, body: dict, deadline_s: float, submit_ms: list[float]
) -> tuple[str, float, str | None, bool]:
    """Returns (outcome, e2e_ms, engine_id, prefix_cache_hit)."""
    started = time.perf_counter()
    try:
        response = await client.post("/v1/inference", json=body)
    except httpx.HTTPError as error:
        return "error", 0.0, type(error).__name__, False
    if response.status_code != 202:
        return "rejected", 0.0, None, False
    submit_ms.append((time.perf_counter() - started) * 1000)
    accepted = response.json()
    while time.perf_counter() - started < deadline_s:
        try:
            result = await client.get(
                f"/v1/inference/{accepted['request_id']}", params={"wait_ms": 5000}
            )
        except httpx.HTTPError:
            await asyncio.sleep(0.05)
            continue
        if result.status_code != 200:
            await asyncio.sleep(0.05)
            continue
        payload = result.json()
        if payload["state"] in ("COMPLETED", "FAILED"):
            assert payload.get("trace_id") == accepted["trace_id"]
            return (
                payload["state"].lower(),
                (time.perf_counter() - started) * 1000,
                payload.get("engine_id"),
                bool(payload.get("prefix_cache_hit")),
            )
    return "timed_out", 0.0, None, False


async def monitor_loop_lag(lags: list[float], interval_s: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    while True:
        started = loop.time()
        await asyncio.sleep(interval_s)
        lags.append(max(0.0, (loop.time() - started - interval_s) * 1000))


async def run_worker(args: argparse.Namespace, index: int, requests: int,
                     qps: float, concurrency: int) -> dict:
    rng = random.Random(args.seed * 1000 + index)
    submit_ms: list[float] = []
    lags: list[float] = []
    cpu_started = time.process_time()
    monitor = asyncio.create_task(monitor_loop_lag(lags))
    limits = httpx.Limits(
        max_connections=args.max_connections, max_keepalive_connections=args.max_connections
    )
    async with httpx.AsyncClient(base_url=args.url, timeout=30, limits=limits) as client:
        wall_started = time.time()
        if args.mode == "open":
            tasks = []
            next_arrival = time.perf_counter()
            for _ in range(requests):
                delay = next_arrival - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(
                    one_request(client, build_body(rng, args), args.deadline_s, submit_ms)))
                next_arrival += rng.expovariate(qps)
            outcomes = list(await asyncio.gather(*tasks))
        else:
            outcomes = []
            remaining = requests

            async def worker() -> None:
                nonlocal remaining
                while remaining > 0:
                    remaining -= 1
                    outcomes.append(await one_request(
                        client, build_body(rng, args), args.deadline_s, submit_ms))

            await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
        wall_finished = time.time()
    monitor.cancel()
    cpu_seconds = time.process_time() - cpu_started
    return {
        "outcomes": outcomes,
        "submit_ms": submit_ms,
        "loop_lag_ms": lags,
        "started": wall_started,
        "finished": wall_finished,
        "cpu_utilization": cpu_seconds / max(wall_finished - wall_started, 1e-9),
    }


def _worker_entry(args: argparse.Namespace, index: int, requests: int, qps: float,
                  concurrency: int, queue: multiprocessing.Queue) -> None:
    queue.put(asyncio.run(run_worker(args, index, requests, qps, concurrency)))


async def warm_up(args: argparse.Namespace) -> None:
    """Sequential unmeasured requests: first Kafka fetches and connection setup."""
    rng = random.Random(args.seed - 1)
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        for _ in range(args.warmup):
            await one_request(client, build_body(rng, args), args.deadline_s, [])


def run(args: argparse.Namespace) -> dict:
    if args.warmup > 0:
        asyncio.run(warm_up(args))
    processes = args.processes or max(1, min(os.cpu_count() or 1, math.ceil(args.qps / 40)))
    if args.mode == "closed":
        processes = min(processes, max(1, args.concurrency))
    processes = min(processes, max(1, args.requests))
    shares = [args.requests // processes + (i < args.requests % processes) for i in range(processes)]
    concurrency = [args.concurrency // processes + (i < args.concurrency % processes)
                   for i in range(processes)]
    if processes == 1:
        parts = [asyncio.run(run_worker(args, 0, shares[0], args.qps, args.concurrency))]
    else:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        workers = [
            context.Process(target=_worker_entry,
                            args=(args, i, shares[i], args.qps / processes, concurrency[i], queue))
            for i in range(processes)
        ]
        for worker in workers:
            worker.start()
        parts = [queue.get() for _ in workers]
        for worker in workers:
            worker.join()

    outcomes = [outcome for part in parts for outcome in part["outcomes"]]
    elapsed = max(p["finished"] for p in parts) - min(p["started"] for p in parts)
    latencies = [latency for kind, latency, _, _ in outcomes if kind == "completed"]
    engines: dict[str, int] = {}
    for kind, _, engine, _ in outcomes:
        if kind == "completed":
            engines[engine or "?"] = engines.get(engine or "?", 0) + 1
    submit = [value for part in parts for value in part["submit_ms"]]
    lags = [value for part in parts for value in part["loop_lag_ms"]]
    peak_cpu = max(part["cpu_utilization"] for part in parts)

    error_types: dict[str, int] = {}
    for kind, _, detail, _ in outcomes:
        if kind == "error":
            error_types[detail or "?"] = error_types.get(detail or "?", 0) + 1

    def count(kind: str) -> int:
        return sum(1 for outcome in outcomes if outcome[0] == kind)

    return {
        "mode": args.mode,
        "requests": args.requests,
        "offered_qps": args.qps if args.mode == "open" else None,
        "concurrency": args.concurrency if args.mode == "closed" else None,
        "client_processes": processes,
        "completed": len(latencies),
        "failed": count("failed"),
        "rejected": count("rejected"),
        "timed_out": count("timed_out"),
        "errors": count("error"),
        "error_types": error_types,
        "achieved_qps": len(latencies) / elapsed if elapsed else 0.0,
        "e2e_p50_ms": percentile(latencies, 0.50),
        "e2e_p90_ms": percentile(latencies, 0.90),
        "e2e_p99_ms": percentile(latencies, 0.99),
        "submit_p50_ms": percentile(submit, 0.50),
        "submit_p99_ms": percentile(submit, 0.99),
        "client_loop_lag_p99_ms": percentile(lags, 0.99),
        "client_peak_cpu_utilization": round(peak_cpu, 3),
        "client_cpu_saturated": peak_cpu > 0.8,
        "completed_by_engine": engines,
        "prefix_cache_hits": sum(1 for kind, _, _, hit in outcomes if kind == "completed" and hit),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--qps", type=float, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--processes", type=int, default=0,
                        help="client worker processes (0 = one per 40 offered QPS)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=0,
                        help="unmeasured sequential requests sent before the run")
    parser.add_argument("--deadline-s", type=float, default=120)
    parser.add_argument("--max-connections", type=int, default=512)
    parser.add_argument("--prefixes", type=int, default=0, help="distinct shared prefixes")
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--prefix-fraction", type=float, default=1.0)
    parser.add_argument("--long-fraction", type=float, default=0.0,
                        help="fraction of requests with --long-output-tokens outputs")
    parser.add_argument("--long-output-tokens", type=int, default=512)
    parser.add_argument("--long-output-max", type=int, default=0,
                        help="if above --long-output-tokens, long outputs are uniform in between")
    # Defaults reproduce the original workload's random draws exactly.
    parser.add_argument("--prompt-min", type=int, default=16)
    parser.add_argument("--prompt-max", type=int, default=128)
    parser.add_argument("--output-min", type=int, default=4)
    parser.add_argument("--output-max", type=int, default=32)
    parser.add_argument("--require-all", action="store_true",
                        help="exit non-zero unless every request completed")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary))
    if args.require_all and summary["completed"] != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
