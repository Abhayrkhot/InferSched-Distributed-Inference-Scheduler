#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(int((len(values) - 1) * q), len(values) - 1)] if values else 0


async def wait_result(
    client: httpx.AsyncClient, request_id: str, trace_id: str, started: float
) -> float:
    while True:
        response = await client.get(f"/v1/inference/{request_id}")
        if response.status_code == 200 and response.json()["state"] == "COMPLETED":
            assert response.json()["trace_id"] == trace_id
            return (time.perf_counter() - started) * 1000
        await asyncio.sleep(0.01)


async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    latencies: list[float] = []
    rejected = 0
    tasks: list[asyncio.Task[float]] = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        started_run = time.perf_counter()
        for _ in range(args.requests):
            started = time.perf_counter()
            response = await client.post(
                "/v1/inference",
                json={
                    "prompt_tokens": rng.randint(16, 128),
                    "max_output_tokens": rng.randint(4, 32),
                    "priority": 0 if rng.random() < 0.1 else 1,
                    "seed": rng.randrange(1 << 31),
                },
            )
            if response.status_code == 202:
                tasks.append(
                    asyncio.create_task(
                        wait_result(
                            client,
                            response.json()["request_id"],
                            response.json()["trace_id"],
                            started,
                        )
                    )
                )
            else:
                rejected += 1
            if args.mode == "open":
                await asyncio.sleep(rng.expovariate(args.qps))
            elif tasks:
                latencies.append(await tasks.pop(0))
        if tasks:
            latencies.extend(await asyncio.gather(*tasks))
        elapsed = time.perf_counter() - started_run
    print(
        json.dumps(
            {
                "mode": args.mode,
                "requests": args.requests,
                "offered_qps": args.qps if args.mode == "open" else None,
                "completed": len(latencies),
                "rejected": rejected,
                "achieved_qps": len(latencies) / elapsed,
                "e2e_p50_ms": percentile(latencies, 0.50),
                "e2e_p90_ms": percentile(latencies, 0.90),
                "e2e_p99_ms": percentile(latencies, 0.99),
            }
        )
    )


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--mode", choices=["open", "closed"], default="open")
parser.add_argument("--requests", type=int, default=1000)
parser.add_argument("--qps", type=float, default=100)
parser.add_argument("--seed", type=int, default=7)
asyncio.run(run(parser.parse_args()))
