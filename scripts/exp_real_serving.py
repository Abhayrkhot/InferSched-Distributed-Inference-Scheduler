#!/usr/bin/env python3
"""Real-model serving experiments (docs/p10-real-model-serving.md).

Every phase uses the same model, KV budget, batch limits and token-id
workload, so InferSched + llama.cpp, llama-server and vLLM serve identical
requests:

  batching   offline capacity: serial vs request-level vs continuous
  calibrate  per-step timings -> least-squares fit of the cost model
  sweep      open-loop QPS sweep: InferSched continuous and static
  reference  the same sweep against llama-server and vLLM
  prefix     shared-prefix workload, prefix cache on vs off
  all        everything above

Runners:
  docker     a Linux host with Docker and an NVIDIA GPU (images
             infersched-gpu:24.04 and vllm/vllm-openai:v0.34.0). The GPU may be
             shared, so each phase takes the cooperative lock at $GPU_LOCK and
             waits until no other process holds GPU memory.
  apptainer  inside a Slurm GPU allocation (Explorer), with the same images as
             .sif files. The allocation owns the GPU, so there is no lock.

Raw JSON lines go to <out>/results.jsonl.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLOCK = 16


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class Runner:
    """Runs commands in the build image ("gpu") or the vLLM image ("vllm").

    Inside either image the repository is /src, models are /models (read-only)
    and the output directory is /out.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.kind = args.runner
        self.models = Path(args.models).resolve()
        self.out = Path(args.out).resolve()
        self.images = {"gpu": args.gpu_image, "vllm": args.vllm_image}
        self.cuda_graphs = args.cuda_graphs == "on"

    def command(self, image: str, argv: list[str], extra_env: dict | None = None) -> list[str]:
        env = {"HOME": "/tmp/home", "HF_HUB_OFFLINE": "1", "USER": "infersched",
               "PYTHONNOUSERSITE": "1", "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
               "VLLM_CACHE_ROOT": "/tmp/vllm-cache", "HF_HOME": "/tmp/hf",
               "no_proxy": "localhost,127.0.0.1", "NO_PROXY": "localhost,127.0.0.1"}
        if image == "gpu" and not self.cuda_graphs:
            # llama.cpp b12188 replays a CUDA graph with an illegal instruction
            # on H200 once a step carries many sequences' prompts (reproduced
            # with llama.cpp's own kernels; see the P10 doc), so llama.cpp runs
            # (InferSched and llama-server alike) default to graphs off.
            env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        env.update(extra_env or {})
        # A None value removes the variable (ggml only checks that it is set).
        env = {key: value for key, value in env.items() if value is not None}
        if self.kind == "docker":
            args = ["docker", "run", "--rm", "--gpus", "all", "--network", "host",
                    "--ipc", "host", "-u", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{REPO}:/src", "-v", f"{self.models}:/models:ro",
                    "-v", f"{self.out}:/out", "-w", "/src"]
            for key, value in env.items():
                args += ["-e", f"{key}={value}"]
            if image == "vllm":  # The image's entrypoint is `vllm serve`.
                if argv[:2] == ["vllm", "serve"]:
                    argv = argv[2:]
                else:
                    args += ["--entrypoint", argv[0]]
                    argv = argv[1:]
            return args + [self.images[image]] + argv
        # Apptainer. The vLLM image is CUDA 13 and Explorer's driver is 570, so
        # it needs CUDA's forward-compatibility libraries (verified in
        # inference-control-plane/hpc/README.md); the build image is CUDA 12.8.
        if image == "vllm":
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda/compat"
        args = ["apptainer", "exec", "--nv", "--cleanenv", "--pwd", "/src",
                "--bind", f"{REPO}:/src,{self.models}:/models:ro,{self.out}:/out"]
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        return args + [self.images[image]] + argv

    def run(self, image: str, argv: list[str],
            extra_env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(self.command(image, argv, extra_env), capture_output=True,
                              text=True)

    def start(self, image: str, argv: list[str], name: str, log_path: Path) -> subprocess.Popen:
        command = self.command(image, argv)
        if self.kind == "docker":
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            command[2:2] = ["--name", name]
        handle = open(log_path, "w")
        return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def stop(self, process: subprocess.Popen, name: str) -> None:
        if self.kind == "docker":
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def gpu_memory_used_mib() -> int:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout
    return int(output.split()[0])


@contextlib.contextmanager
def gpu_lock(args: argparse.Namespace, phase: str):
    if args.runner != "docker" or args.no_lock:
        yield
        return
    lock = Path(os.environ.get("GPU_LOCK", Path.home() / "qildata/logs/gpu.lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as handle:
        log(f"{phase}: waiting for GPU lock {lock}")
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            while True:  # Another project's job may run without the lock.
                foreign = subprocess.run(
                    ["docker", "ps", "-q", "--filter", "ancestor=qil-dev:latest",
                     "--filter", "ancestor=qil-ref:latest"],
                    capture_output=True, text=True).stdout.strip()
                used = gpu_memory_used_mib()
                if not foreign and used < 800:
                    break
                log(f"{phase}: GPU busy ({used} MiB used, foreign={bool(foreign)}); waiting")
                time.sleep(30)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def last_json(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON in output:\n{text[-3000:]}")
    return json.loads(lines[-1])


class Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runner = Runner(args)
        self.out = self.runner.out
        self.bench_binary = f"/src/{args.build_dir}/cpp/bench/infersched_real_serving_bench"
        self.llama_server = f"/src/{args.build_dir}/bin/llama-server"
        self.gguf = f"/models/{args.gguf}"
        self.hf_model = f"/models/{args.hf_model}"
        self.mixed = ["--prompt-min", "128", "--prompt-max", "512", "--short-min", "16",
                      "--short-max", "64", "--long-min", "256", "--long-max", "512",
                      "--long-fraction", "0.2"]
        self.capacity_rps: float | None = None
        self.prefixed = ["--prefixes", "8", "--prefix-tokens", "512", "--prompt-min", "32",
                         "--prompt-max", "128", "--short-min", "16", "--short-max", "64",
                         "--long-fraction", "0"]

    def engine_args(self, policy: str, extra: list[str] | None = None) -> list[str]:
        # Unique-prompt workloads run with the prefix cache off: nothing can be
        # reused, and llama.cpp's unified KV attends over every occupied cell,
        # so blocks kept for finished requests would only slow decode
        # (llama-server gets cache_prompt=false for the same reason). The
        # prefix phase overrides this; the bench takes the last occurrence.
        a = self.args
        return ["--model", self.gguf, "--policy", policy, "--kv-blocks", str(a.kv_blocks),
                "--max-batch", str(a.max_seqs), "--max-batch-tokens", str(a.step_tokens),
                "--max-prefill-seqs", str(a.prefill_seqs), "--prefix-cache", "off",
                "--holders", "0", "--threads", "8",
                "--backend-sampling", self.args.backend_sampling] + (extra or [])

    def record(self, phase: str, result: dict) -> None:
        result.update({"phase": phase, "model_file": self.args.gguf,
                       "load_avg": round(os.getloadavg()[0], 1)})
        with open(self.out / "results.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        keys = ["label", "achieved_rps", "output_tok_s", "ttft_p50_ms", "ttft_p99_ms",
                "tpot_p50_ms", "e2e_p99_ms", "mean_batch_occupancy", "gpu_util_pct", "failed"]
        log(" ".join(f"{key}={result.get(key)}" for key in keys if key in result))

    def bench(self, label: str, policy: str, qps: float, requests: int, workload: list[str],
              extra: list[str] | None = None, extra_env: dict | None = None) -> dict:
        argv = [self.bench_binary, *self.engine_args(policy, extra), "--qps", str(qps),
                "--requests", str(requests), "--label", label, *workload]
        log(f"bench {label}")
        run = self.runner.run("gpu", argv, extra_env)
        if "{" not in run.stdout:
            raise RuntimeError(f"bench {label} failed:\n{run.stdout[-1500:]}\n{run.stderr[-3000:]}")
        result = last_json(run.stdout)
        graphs = self.runner.cuda_graphs
        if extra_env and "GGML_CUDA_DISABLE_GRAPHS" in extra_env:
            graphs = extra_env["GGML_CUDA_DISABLE_GRAPHS"] is None
        result["cuda_graphs"] = graphs
        return result

    def dump_workload(self, name: str, qps: float, requests: int, workload: list[str]) -> str:
        path = f"/out/{name}.jsonl"
        argv = [self.bench_binary, *self.engine_args("continuous", ["--gpu-layers", "0"]),
                "--qps", str(qps), "--requests", str(requests), *workload,
                "--dump-workload", path, "--dump-only"]
        run = self.runner.run("gpu", argv)
        if run.returncode != 0:
            raise RuntimeError(f"workload dump failed:\n{run.stderr[-3000:]}")
        return path

    @contextlib.contextmanager
    def server(self, kind: str):
        a = self.args
        name = f"infersched-p10-{kind}"
        log_path = self.out / f"{kind}-server.log"
        if kind == "llama":
            port = a.port_base
            argv = [self.llama_server, "-m", self.gguf, "-ngl", "999",
                    "-c", str(a.kv_blocks * BLOCK), "-np", str(a.max_seqs),
                    "-b", str(a.step_tokens), "-ub", "512", "-fa", "on", "-kvu",
                    "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
            if a.backend_sampling == "on":
                argv.append("--backend-sampling")
            process = self.runner.start("gpu", argv, name, log_path)
            timeout = 600
        else:
            port = a.port_base + 1
            argv = ["vllm", "serve", self.hf_model, "--served-model-name", "model",
                    "--dtype", "float16", "--max-num-seqs", str(a.max_seqs),
                    "--max-num-batched-tokens", str(a.step_tokens),
                    "--num-gpu-blocks-override", str(a.kv_blocks), "--block-size", str(BLOCK),
                    "--max-model-len", str(a.max_model_len),
                    "--gpu-memory-utilization", str(a.vllm_memory_fraction),
                    "--host", "127.0.0.1", "--port", str(port)]
            process = self.runner.start("vllm", argv, name, log_path)
            timeout = 1200
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + timeout
        try:
            while True:
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if process.poll() is not None:
                    raise RuntimeError(f"{kind} server exited; see {log_path}")
                if time.time() > deadline:
                    raise RuntimeError(f"{kind} server not ready after {timeout}s")
                time.sleep(3)
            log(f"{kind} server ready")
            yield url
        finally:
            self.runner.stop(process, name)

    def client(self, kind: str, url: str, workload: str, label: str) -> dict:
        argv = ["python3", "/src/scripts/bench_openai_like.py", "--server", kind, "--url", url,
                "--workload", workload, "--label", label, "--model", "model"]
        log(f"client {label}")
        run = self.runner.run("vllm", argv)
        return last_json(run.stdout + "\n" + run.stderr)

    # Phases ------------------------------------------------------------------

    def points(self) -> list[float]:
        # --qps auto: fractions of the capacity the batching phase measured.
        if self.args.qps != "auto":
            return [float(value) for value in self.args.qps.split(",")]
        if self.capacity_rps is None:
            raise RuntimeError("--qps auto needs the batching phase first")
        return [round(fraction * self.capacity_rps, 1)
                for fraction in (0.25, 0.5, 0.75, 0.9, 1.0, 1.15)]

    def calibrate(self) -> None:
        with gpu_lock(self.args, "calibrate"):
            for label, qps in (("calibrate-burst", 0.0), ("calibrate-mid", self.points()[1])):
                self.record("calibrate", self.bench(
                    label, "continuous", qps, 240, self.mixed,
                    ["--step-trace", f"/out/{label}.steps.jsonl"]))

    def batching(self) -> None:
        with gpu_lock(self.args, "batching"):
            for policy in ("continuous", "static", "serial"):
                result = self.bench(f"burst-{policy}", policy, 0, self.args.burst_requests,
                                    self.mixed)
                if policy == "continuous":
                    self.capacity_rps = result["achieved_rps"]
                self.record("batching", result)
            if not self.runner.cuda_graphs:
                # What turning graphs off costs, where graphs are safe (steps of
                # at most --step-tokens with --prefill-seqs prompts).
                self.record("batching", self.bench(
                    "burst-continuous-graphs-on", "continuous", 0, self.args.burst_requests,
                    self.mixed, extra_env={"GGML_CUDA_DISABLE_GRAPHS": None}))

    def sweep(self) -> None:
        with gpu_lock(self.args, "sweep"):
            for qps in self.points():
                for policy in ("continuous", "static"):
                    result = self.bench(f"{policy}-q{qps:g}", policy, qps, self.args.requests,
                                        self.mixed)
                    result["qps"] = qps
                    self.record("sweep", result)

    def reference(self) -> None:
        workloads = {qps: self.dump_workload(f"mixed-q{qps:g}", qps, self.args.requests,
                                             self.mixed) for qps in self.points()}
        for kind in self.args.servers:
            try:
                with gpu_lock(self.args, f"reference-{kind}"), self.server(kind) as url:
                    for qps in workloads:
                        result = self.client(kind, url, workloads[qps], f"{kind}-q{qps:g}")
                        result["qps"] = qps
                        self.record("reference", result)
            except RuntimeError as error:  # Keep the other server's results.
                log(f"reference {kind} failed: {error}")

    def prefix(self) -> None:
        with gpu_lock(self.args, "prefix"):
            qps = (round(0.5 * self.capacity_rps, 1) if self.args.prefix_qps == "auto"
                   else float(self.args.prefix_qps))
            for setting in ("off", "on"):
                result = self.bench(f"prefix-cache-{setting}-q{qps:g}", "continuous", qps,
                                    self.args.requests, self.prefixed,
                                    ["--prefix-cache", setting, "--holders", "64"])
                result["qps"] = qps
                self.record("prefix", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phases", nargs="+",
                        choices=["calibrate", "batching", "sweep", "reference", "prefix", "all"])
    parser.add_argument("--runner", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--gpu-image", default="infersched-gpu:24.04",
                        help="docker image or .sif with the CUDA build environment")
    parser.add_argument("--vllm-image", default="vllm/vllm-openai:v0.34.0")
    parser.add_argument("--build-dir", default="build/llama-cuda")
    parser.add_argument("--models", default=str(Path.home() / "infersched-models"))
    parser.add_argument("--gguf", default="gguf/Qwen3-0.6B-F16.gguf")
    parser.add_argument("--hf-model", default="hf/Qwen3-0.6B")
    parser.add_argument("--out", default=str(REPO / "build" / "p10"))
    # Limits shared by all three systems. Defaults fit a 4 GB GPU with
    # Qwen3-0.6B: 768 x 16 = 12,288 KV tokens (1.3 GB at 112 KiB/token).
    parser.add_argument("--kv-blocks", type=int, default=768)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--step-tokens", type=int, default=2048)
    parser.add_argument("--prefill-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-memory-fraction", type=float, default=0.8)
    parser.add_argument("--qps", default="auto",
                        help="comma-separated offered QPS, or auto (fractions of capacity)")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--burst-requests", type=int, default=192)
    parser.add_argument("--prefix-qps", default="auto")
    parser.add_argument("--servers", default="llama,vllm")
    parser.add_argument("--port-base", type=int, default=58201)
    parser.add_argument("--cuda-graphs", choices=["on", "off"], default="off")
    parser.add_argument("--backend-sampling", choices=["on", "off"], default="on")
    parser.add_argument("--no-lock", action="store_true",
                        help="skip the shared-GPU lock (CPU-only dry runs)")
    args = parser.parse_args()
    args.servers = args.servers.split(",")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o777)
    experiment = Experiment(args)
    phases = args.phases if "all" not in args.phases else [
        "batching", "calibrate", "sweep", "prefix", "reference"]
    log(f"phases {phases} on {args.gguf}; kv_blocks={args.kv_blocks} "
        f"max_seqs={args.max_seqs} step_tokens={args.step_tokens}")
    for phase in phases:
        getattr(experiment, phase)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
