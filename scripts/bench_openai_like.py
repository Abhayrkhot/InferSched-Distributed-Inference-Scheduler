#!/usr/bin/env python3
"""Replay an InferSched workload against llama-server or vLLM.

The workload file comes from `infersched_real_serving_bench --dump-workload`:
one JSON object per line with the prompt as token ids, the exact number of
output tokens, and the request's arrival offset. Every system therefore serves
byte-identical prompts on the same open-loop schedule, and every request is
forced to its exact length (ignore_eos), so tokens/s and latency compare
like for like. Output: one JSON line with the same keys as the C++ benchmark.

Needs aiohttp (present in the vLLM image). GPU telemetry comes from
`nvidia-smi` when the container has GPU access.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import threading
import time

import aiohttp


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


class GpuSampler:
    """Polls nvidia-smi every 100 ms; summarizes a time window."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float, float, float]] = []
        self.name = ""
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.name = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()[0]
            self._process = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                 "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=subprocess.PIPE, text=True)
        except (OSError, subprocess.SubprocessError, IndexError):
            self._process = None
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            parts = [part.strip() for part in line.split(",")]
            try:
                util, mem_util, power, used = (float(part) for part in parts[:4])
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), util, mem_util, power, used))

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)

    def summarize(self, start: float, end: float) -> dict[str, object]:
        window = [sample for sample in self.samples if start <= sample[0] <= end]
        if not window:
            return {"gpu": self.name, "gpu_samples": 0}
        count = len(window)
        return {
            "gpu": self.name,
            "gpu_samples": count,
            "gpu_util_pct": round(sum(s[1] for s in window) / count, 1),
            "gpu_mem_util_pct": round(sum(s[2] for s in window) / count, 1),
            "gpu_power_w": round(sum(s[3] for s in window) / count, 1),
            "gpu_peak_used_mib": int(max(s[4] for s in window)),
        }


async def run_request(session: aiohttp.ClientSession, args: argparse.Namespace,
                      item: dict, start: float) -> dict:
    arrival = start + item["arrival_s"]
    delay = arrival - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    if args.server == "llama":
        url = f"{args.url}/completion"
        body = {"prompt": item["prompt"], "n_predict": item["max_tokens"],
                "ignore_eos": True, "temperature": 0.0, "top_k": 1, "stream": True,
                "cache_prompt": args.cache_prompt, "return_tokens": True}
    else:
        url = f"{args.url}/v1/completions"
        body = {"model": args.model, "prompt": item["prompt"],
                "max_tokens": item["max_tokens"], "min_tokens": item["max_tokens"],
                "ignore_eos": True, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True}}
    first = None
    produced = 0
    token_ids: list[int] = []
    error = ""
    try:
        async with session.post(url, json=body) as response:
            if response.status != 200:
                error = f"http {response.status}: {(await response.text())[:200]}"
            else:
                async for raw in response.content:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    data = json.loads(payload)
                    now = time.perf_counter()
                    if args.server == "llama":
                        chunk = data.get("tokens") or []
                        token_ids.extend(chunk)
                        if first is None and (chunk or data.get("content")):
                            first = now
                        if data.get("stop"):
                            produced = int(data.get("tokens_predicted", len(token_ids)))
                    else:
                        if first is None and data.get("choices"):
                            first = now
                        if data.get("usage"):
                            produced = int(data["usage"]["completion_tokens"])
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        error = repr(exc)
    end = time.perf_counter()
    return {"id": item["id"], "arrival": arrival, "first": first or end, "end": end,
            "expected": item["max_tokens"], "produced": produced,
            "prompt_tokens": len(item["prompt"]), "tokens": token_ids, "error": error}


async def replay(args: argparse.Namespace, items: list[dict]) -> tuple[float, list[dict]]:
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start = time.perf_counter() + 0.5
        results = await asyncio.gather(*(run_request(session, args, item, start)
                                         for item in items))
    return start, list(results)


async def warm_up(args: argparse.Namespace, items: list[dict]) -> None:
    """A few requests before the clock starts (kernel load, CUDA graphs)."""
    warm = [dict(item, arrival_s=0.0, id=f"warm-{index}")
            for index, item in enumerate(items[:args.warmup])]
    await replay(args, warm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=["llama", "vllm"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="", help="vLLM served model name")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--cache-prompt", action="store_true",
                        help="llama-server: reuse a slot's previous prompt KV")
    parser.add_argument("--dump-outputs", default="")
    args = parser.parse_args()

    with open(args.workload, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    if args.warmup:
        asyncio.run(warm_up(args, items))
    sampler = GpuSampler()
    sampler.start()
    start, results = asyncio.run(replay(args, items))
    sampler.stop()

    ok = [r for r in results if not r["error"] and r["produced"] == r["expected"]]
    failed = [r for r in results if r not in ok]
    end = max(r["end"] for r in results)
    duration = end - start
    output_tokens = sum(r["produced"] for r in ok)
    prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    ttft = [(r["first"] - r["arrival"]) * 1e3 for r in ok]
    e2e = [(r["end"] - r["arrival"]) * 1e3 for r in ok]
    tpot = [(r["end"] - r["first"]) * 1e3 / (r["produced"] - 1) for r in ok if r["produced"] > 1]
    summary = {
        "label": args.label or args.server, "server": args.server,
        "requests": len(results), "failed": len(failed),
        "failure_sample": failed[0]["error"] or
        f"produced {failed[0]['produced']} of {failed[0]['expected']}" if failed else "",
        "duration_s": round(duration, 3),
        "achieved_rps": round(len(ok) / duration, 3),
        "output_tokens": output_tokens, "prompt_tokens": prompt_tokens,
        "output_tok_s": round(output_tokens / duration, 1),
        "total_tok_s": round((output_tokens + prompt_tokens) / duration, 1),
        "ttft_p50_ms": round(percentile(ttft, 0.5), 1),
        "ttft_p90_ms": round(percentile(ttft, 0.9), 1),
        "ttft_p99_ms": round(percentile(ttft, 0.99), 1),
        "tpot_p50_ms": round(percentile(tpot, 0.5), 2),
        "tpot_p99_ms": round(percentile(tpot, 0.99), 2),
        "e2e_p50_ms": round(percentile(e2e, 0.5), 1),
        "e2e_p99_ms": round(percentile(e2e, 0.99), 1),
    }
    summary.update(sampler.summarize(start, end))
    if args.dump_outputs:
        with open(args.dump_outputs, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps({"id": result["id"], "tokens": result["tokens"]}) + "\n")
    print(json.dumps(summary))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
