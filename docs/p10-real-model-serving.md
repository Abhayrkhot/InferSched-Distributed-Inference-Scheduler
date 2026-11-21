# P10: real-model serving throughput

Up to P9, an Engine never ran a model: each iteration advanced a fake clock by a cost formula,
so every throughput and compute number described the scheduler under an assumed cost, not
inference. P10 puts a real model behind the same scheduler and measures serving throughput,
latency and GPU utilization on a GPU.

The scheduler is unchanged. `ContinuousEngine` still decides admission, batch membership,
preemption, KV block placement and prefix reuse. What changed is who performs that work:

```
ContinuousEngine ──decides──► StepExecutor ──performs──► CostModelExecutor   (default, P1–P9)
  admission, batching,          Prefill()                  LlamaExecutor       (P10, llama.cpp)
  preemption, paged KV,         Decode()
  radix prefix cache            Release() / OnBlockFreed()
```

## Setup

| | |
|---|---|
| GPU | NVIDIA H200, 141 GB (143,771 MiB), driver 570.86.15; one dedicated GPU per Slurm job on Northeastern's Explorer cluster |
| Host | Intel Xeon Gold 6548Y+ (16 cores allocated) |
| Model | Qwen3-8B (8.19B parameters), FP16: a 16.4 GB GGUF for llama.cpp, the Hugging Face weights for vLLM |
| Software | llama.cpp b12188 (one build serves InferSched and llama-server), vLLM 0.34.0, CUDA 12.8 (vLLM: 13 with forward compatibility) |
| Shared limits | 64 running sequences; 65,536 KV tokens (4,096 blocks of 16); 4,096 tokens per step. InferSched admits up to 16 new prompts per step |
| Workload | Prompts of 128–512 tokens; 80% of outputs are 16–64 tokens and 20% are 256–512, all exact length with greedy decoding. Open-loop Poisson arrivals with 600 requests per point; "burst" is 256 requests at once |
| llama.cpp settings | Sampling on the device, CUDA graphs off (both explained under "How the real backend works") |

All three systems serve **the same requests**. The benchmark writes its workload as token ids
(`--dump-workload`), and `scripts/bench_openai_like.py` replays that file against llama-server
and vLLM on the same arrival schedule.

There is one important asymmetry. InferSched is measured in-process: the scheduler plus
llama.cpp, with no network layer. llama-server and vLLM are measured through their HTTP APIs with
per-token streaming. The full distributed path (HTTP, Kafka, Router, Postgres) with a real model
is checked separately below, on CPU only.

## Results

### Batching policy changes what the GPU produces, not how busy it looks

The same 256 requests (28,504 output tokens and 84,982 prompt tokens) served all at once by one
Engine. Energy is NVML power × wall time.

| Policy | Output tok/s | Mean batch | GPU busy (NVML) | GPU power | Energy per output token | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One request at a time | 179 | 1.0 | 98% | 638 W | 3.56 J | 159.1 s |
| Request-level batching | 1,251 | 14.7 | 96% | 478 W | 0.38 J | 22.8 s |
| Continuous batching (InferSched) | 1,683 | 41.7 | 90% | 449 W | 0.27 J | 16.9 s |
| Continuous, CUDA graphs on | 1,738 | 41.7 | 89% | 481 W | 0.28 J | 16.4 s |

- Continuous batching delivers **9.4×** the throughput of one-at-a-time serving and **1.35×**
  that of request-level batching on the same GPU, model and kernels.
- It does the same work for **30% less GPU energy** than request-level batching, and **93% less**
  than serving one request at a time.
- NVML's "GPU busy" is 90–98% under all three policies. It records that a kernel was running,
  not how much useful work the kernels did. Tokens per GPU-second and joules per token tell the
  policies apart; utilization does not.
- Turning CUDA graphs back on, where that is safe, adds 3.3% throughput. That is what the
  graphs-off setting costs the llama.cpp numbers below.

### Under open-loop load

| Offered QPS | Policy | Achieved req/s | Output tok/s | TTFT p50 / p99 | TPOT p50 / p99 | E2E p99 | Mean batch | GPU busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.8 | continuous | 3.62 | 362 | 26 / 54 ms | 8.5 / 11.4 ms | 4.3 s | 3.0 | 92% |
| 3.8 | static | 3.59 | 358 | 1,444 / 3,844 ms | 9.1 / 10.9 ms | 6.1 s | 2.6 | 96% |
| 7.6 | continuous | 7.09 | 708 | 30 / 70 ms | 10.2 / 14.6 ms | 5.1 s | 7.0 | 96% |
| 7.6 | static | 6.92 | 691 | 2,450 / 5,114 ms | 11.1 / 13.6 ms | 8.5 s | 6.1 | 97% |
| 11.3 | continuous | 10.32 | 1,030 | 35 / 100 ms | 12.4 / 19.9 ms | 6.6 s | 12.1 | 95% |
| 11.3 | static | 9.79 | 976 | 3,272 / 6,930 ms | 13.1 / 16.1 ms | 9.7 s | 9.9 | 97% |
| 13.6 | continuous | 12.23 | 1,220 | 40 / 131 ms | 14.3 / 27.2 ms | 8.5 s | 16.6 | 94% |
| 13.6 | static | 10.78 | 1,075 | 4,726 / 9,900 ms | 14.5 / 16.2 ms | 11.8 s | 11.5 | 97% |
| 15.1 | continuous | 13.38 | 1,335 | 46 / 154 ms | 15.9 / 35.3 ms | 10.2 s | 20.2 | 93% |
| 15.1 | static | 10.97 | 1,094 | 6,115 / 12,956 ms | 14.6 / 16.2 ms | 14.2 s | 11.8 | 96% |
| 17.4 | continuous | 14.63 | 1,459 | 60 / 989 ms | 20.4 / 48.9 ms | 13.9 s | 27.9 | 91% |
| 17.4 | static | 11.02 | 1,100 | 8,015 / 17,399 ms | 14.4 / 15.9 ms | 18.3 s | 11.8 | 96% |

The offered rates are fractions of the burst capacity (25% to 115%), which the driver derives
from the batching run.

- **Time to first token:** at light load both policies keep up, but under request-level batching
  a new request waits for the running batch to drain. Its median time to first token is 1.4–8.0 s,
  against 26–60 ms under continuous batching.
- **Throughput:** request-level batching saturates near 11 req/s. Continuous batching reaches
  14.6 req/s at 17.4 offered QPS, **33% more**.
- **The cost of continuous batching:** each running sequence's per-token time grows (TPOT p99 is
  48.9 ms against 15.9 ms at 17.4 QPS), because new prompts ride in the same steps. This is the
  prefill interference P9 reported from the cost model, now measured. Chunked prefill with a
  smaller per-step prompt budget is the usual remedy, and InferSched has the knob
  (`--max-batch-tokens`, `--max-prefill-seqs`), but this sweep did not tune it.

### Against llama-server and vLLM

The same token-id workloads, replayed against llama-server (the same llama.cpp build and kernels,
`-np 64 -kvu -b 4096 -fa on --backend-sampling`, CUDA graphs off) and vLLM (FP16,
`--max-num-seqs 64 --max-num-batched-tokens 4096 --num-gpu-blocks-override 4096`).

| Offered QPS | System | Achieved req/s | Output tok/s | TTFT p50 / p99 | TPOT p50 / p99 | E2E p99 | GPU busy | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.8 | InferSched + llama.cpp | 3.62 | 362 | 26 / 54 ms | 8.5 / 11.4 ms | 4.3 s | 92% | 0 |
| 3.8 | llama-server | 3.60 | 359 | 111 / 326 ms | 17.4 / 38.5 ms | 9.0 s | 59% | 0 |
| 3.8 | vLLM | 3.65 | 364 | 20 / 38 ms | 5.4 / 5.9 ms | 2.7 s | 87% | 0 |
| 7.6 | InferSched + llama.cpp | 7.09 | 708 | 30 / 70 ms | 10.2 / 14.6 ms | 5.1 s | 96% | 0 |
| 7.6 | llama-server | 6.55 | 654 | 206 / 3,370 ms | 71.7 / 104.3 ms | 35.5 s | 42% | 0 |
| 7.6 | vLLM | 7.18 | 716 | 21 / 33 ms | 5.5 / 6.2 ms | 2.7 s | 98% | 0 |
| 11.3 | InferSched + llama.cpp | 10.32 | 1,030 | 35 / 100 ms | 12.4 / 19.9 ms | 6.6 s | 95% | 0 |
| 11.3 | llama-server | 7.11 | 710 | 4,143 / 20,742 ms | 88.2 / 130.7 ms | 43.1 s | 47% | 1 |
| 11.3 | vLLM | 10.52 | 1,050 | 21 / 58 ms | 5.7 / 6.5 ms | 2.8 s | 99% | 0 |
| 13.6 | InferSched + llama.cpp | 12.23 | 1,220 | 40 / 131 ms | 14.3 / 27.2 ms | 8.5 s | 94% | 0 |
| 13.6 | llama-server | 7.32 | 730 | 7,284 / 28,071 ms | 84.8 / 144.0 ms | 46.7 s | 46% | 0 |
| 13.6 | vLLM | 12.55 | 1,252 | 22 / 36 ms | 5.8 / 6.7 ms | 2.9 s | 99% | 0 |
| 15.1 | InferSched + llama.cpp | 13.38 | 1,335 | 46 / 154 ms | 15.9 / 35.3 ms | 10.2 s | 93% | 0 |
| 15.1 | llama-server | 7.19 | 715 | 9,687 / 33,932 ms | 85.7 / 147.4 ms | 51.1 s | 46% | 1 |
| 15.1 | vLLM | 13.85 | 1,382 | 22 / 208 ms | 5.9 / 7.9 ms | 2.9 s | 99% | 0 |
| 17.4 | InferSched + llama.cpp | 14.63 | 1,459 | 60 / 989 ms | 20.4 / 48.9 ms | 13.9 s | 91% | 0 |
| 17.4 | llama-server | 7.46 | 741 | 10,682 / 36,094 ms | 83.2 / 117.6 ms | 51.3 s | 48% | 2 |
| 17.4 | vLLM | 15.82 | 1,578 | 23 / 43 ms | 6.0 / 7.3 ms | 3.0 s | 99% | 0 |

- **llama-server, same kernels:** it saturates near 7.3 req/s with the GPU busy only 42–48% of the
  time. InferSched driving the same kernels reaches 14.6 req/s, **1.96×** at 17.4 QPS. llama-server
  also dropped 1–2 of 600 connections per run from 11.3 QPS up ("Server disconnected").
  - What this does not show: that llama-server's batch scheduler alone is twice as slow. Part of
    its gap is its HTTP server and per-token streaming, which the in-process InferSched number does
    not pay. Its idle GPU says the gap is on the host side, not in the kernels.
- **vLLM:** InferSched's throughput is within **1–4% of vLLM's** up to 15.1 QPS (its measured burst
  capacity) and 7.5% below at 17.4 QPS, where vLLM still has headroom. vLLM's per-token latency is far lower and stays flat:
  TPOT p50 is 5.4–6.0 ms, against 8.5–20.4 ms for InferSched, 1.6–3.4× higher. The fitted step
  cost below shows why. In llama.cpp a step costs 6.5 ms before the first sequence and 0.28 ms for
  each one after it, while vLLM's per-token time barely moves as its batch grows. So InferSched's
  scheduler matches vLLM on how much work it extracts from the GPU, but llama.cpp's kernels and
  per-step overhead (no CUDA graphs here) cost each request more time.

### Repeatability

A second trial of the InferSched runs (batching, sweep, prefix; same flags, a separate H200
allocation; [`h200-qwen3-8b-trial2/`](p10-results/h200-qwen3-8b-trial2/)) reproduced the first:

| Metric | Largest change |
| --- | --- |
| Output tokens/s, achieved req/s | 1.4% |
| TTFT p50 | 2.9% |
| TPOT p50 | 3.8% |
| E2E p99 | 3.3% |
| TTFT p99 | 24% (continuous at 17.4 QPS, 989 → 1,231 ms); 8.6% with the prefix cache on (42 → 46 ms); under 5% everywhere else |

The prompt-token counts are identical, because the workload is seeded. The reference servers ran
once.

### Prefix caching on a real model

The workload is 8 shared 512-token system prompts plus 32–128 unique tokens, with 16–64-token
outputs, at 7.6 QPS. The prompts are identical between the two runs; only the radix prefix cache
is switched.

| Prefix cache | Prompt tokens computed | Prompt tokens reused | TTFT p50 / p99 | E2E p99 | Output tok/s | Engine busy fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 355,999 | 0 | 42 / 110 ms | 0.87 s | 290 | 0.91 |
| on | 52,895 | 303,104 | 17 / 42 ms | 0.58 s | 290 | 0.86 |

- **Compute:** prompt computation drops **85%**, with KV that is really shared. 2,317 blocks were
  parked in holder sequences and reused across requests.
- **Latency:** time to first token drops 59% at the median and 62% at p99, and end-to-end p99
  drops 33%.
- **Busy time:** model time falls only 5% (75.6 s to 71.5 s of an 83 s run), and GPU energy over
  the run falls 7%. The fitted cost model predicts about 22% (303k fewer prompt tokens at
  55 µs each). I did not isolate the gap. One candidate is specific to llama.cpp: its unified KV
  cache attends over every occupied cell, so parked prefix cells make each decode step attend
  over more (masked) cells. The cost model has no term for that.
- **Compared with P9:** P9's modeled affinity experiment reported a fleet hardware-time saving
  of 29–33%. On this hardware and workload the measured saving is mainly latency; the compute
  saving is small.

### Calibrating the cost model

Every scheduler step of two runs was recorded: prompt tokens computed, sequences decoded and
measured time, 4,298 steps with batches of 0–64. A least-squares fit of the cost model the
simulator uses (`scripts/fit_cost_model.py`;
[`cost-model-fit.json`](p10-results/h200-qwen3-8b/cost-model-fit.json)):

```
step time = 6,481 µs + 285 µs × decoding sequences + 55 µs × prompt tokens
            R² 0.92, median error 5.9%, p90 error 25%
```

Adding an attention term (the KV positions the decode tokens attend to) barely improves the fit
(R² 0.93) and gets a negative coefficient, so I left it out. A step's fixed cost equals about 23
sequences' worth of per-sequence cost. That is the memory-bound regime P9 assumed for its
batching result: P9's model put it at 50 (500 µs + 10 µs per sequence).

With these constants, the simulator (`infersched_local_bench --cost-model h200-qwen3-8b
--workload p10`, which draws the same length distributions) predicts the measured results:

| | Measured on the H200 | Calibrated simulator | Error |
|---|---:|---:|---:|
| Burst, continuous | 15.1 req/s | 14.7 req/s | −3% |
| Burst, request-level | 11.2 req/s | 10.3 req/s | −8% |
| Continuous ÷ request-level | 1.35× | 1.43× | |
| Burst, continuous busy time | 16.9 s | 17.4 s | +3% |
| Burst, continuous mean batch | 41.7 | 38.7 | −7% |
| ~17 QPS, continuous achieved | 14.6 req/s | 15.0 req/s | +3% |
| ~17 QPS, continuous TTFT p99 | 989 ms | 851 ms | −14% |
| ~17 QPS, request-level achieved | 11.0 req/s | 9.8 req/s | −11% |

The simulator ran at 17 QPS against the measured 17.4 (it takes whole numbers). Its raw output is
[`calibrated-simulation.jsonl`](p10-results/h200-qwen3-8b/calibrated-simulation.jsonl).

- **Accuracy:** calibrated on one run, the phenomenological model predicts continuous batching
  within a few percent. It underestimates request-level batching by 8–11%, because a linear
  per-token prompt cost misses how efficient very large prefill batches are.
- **What this means for P9:** P9's batching result was 400 vs ~188 QPS, about 2.1×, under an
  assumed model. On a real GPU the gain is **1.35×**, and the calibrated model predicts 1.43×. The
  direction holds; the size was optimistic.

### The whole distributed path, with a real model

On the shared Linux box (i7-12700H, CPU only, because the box's 4 GB GPU was running another
project's queue), the full path served Qwen3-0.6B (FP16, 8 cores): HTTP gateway → Kafka → Router
→ llama.cpp Engine → Postgres → client.

- All 40 open-loop requests completed with 0 failures, and the Engine computed 2,357 real output
  tokens.
- End-to-end latency was 2.5 s p50 and 11.1 s p99 at 0.31 req/s, a load the CPU could sustain.
- Results in Postgres hold the model's text. With a random-token prompt that is a run of years,
  `" 2019年 2020年 2021年 …"`.

This is a functional check: CPU decode at this size is about 26 output tokens per busy second,
so it says nothing about GPU throughput. Raw files: [`p10-results/box-cpu-full-path/`](p10-results/box-cpu-full-path/).

## What this shows, and what it does not

**Shown:**

- The scheduler's decisions drive real inference. Continuous batching gets 1.35× the tokens per
  GPU-second of request-level batching and 9.4× that of serial serving on an H200, for 30% and
  93% less energy per token.
- The same scheduler on llama.cpp's kernels matches vLLM's request throughput within 1–4% up to
  its measured capacity.
- Prefix caching shares real KV and cuts prompt computation by 85%.
- The phenomenological cost model, once fitted to measured steps, predicts real capacity within
  3–11%.

**Not shown:**

- **A network serving layer on the GPU.** InferSched's GPU numbers are in-process; llama-server's
  and vLLM's include HTTP streaming.
- **Repeatability of the references.** InferSched's runs were repeated (above); llama-server and
  vLLM ran once each.
- **Other hardware, models, or quantization.** One GPU and one model in FP16 were measured.
- **Kernel-level efficiency.** NVML busy time and power were measured, but not memory-bandwidth
  or tensor-core utilization.
- **Real prompts and stopping.** Prompts are synthetic token ids and EOS is ignored, so
  throughput reflects scheduling, not how long a model chooses to answer.
- **Default llama.cpp settings.** CUDA graphs are off for every llama.cpp run because of the
  b12188 bug.

## Reproduce

On a machine with Docker and an NVIDIA GPU:

```bash
docker build -t infersched-gpu:24.04 -f docker/gpu.Dockerfile docker
docker run --rm -v $PWD:/src -w /src infersched-gpu:24.04 bash -c \
  "cmake --preset llama-cuda -DINFERSCHED_LLAMA_TOOLS=ON && cmake --build build/llama-cuda"
python3 scripts/exp_real_serving.py all --models <dir with gguf/ and hf/>   # see --help
python3 scripts/p10_tables.py build/p10/results.jsonl
```

On a Slurm cluster with Apptainer (how the numbers above were produced), use the job scripts in
`scripts/explorer/`:

1. `build.sbatch` converts the image and compiles for sm_80 and sm_90.
2. `models.sbatch` downloads the models and converts them to GGUF.
3. `experiments.sbatch` runs `exp_real_serving.py --runner apptainer` on one GPU.

The exact command for these results:

```bash
sbatch --gres=gpu:h200:1 --mem=40G scripts/explorer/experiments.sbatch all \
  --gguf gguf/Qwen3-8B-F16.gguf --hf-model hf/Qwen3-8B --kv-blocks 4096 --max-seqs 64 \
  --step-tokens 4096 --prefill-seqs 16 --requests 600 --burst-requests 256
```

Raw results are in [`p10-results/h200-qwen3-8b/`](p10-results/h200-qwen3-8b/).

## How the real backend works

`LlamaExecutor` (`cpp/src/llama_executor.cpp`) runs a GGUF model with llama.cpp (pinned release
b12188) on exactly the batches the scheduler forms.

- **One model call per scheduler iteration.** Prompt tokens of newly admitted sequences and one
  token for every running sequence go into a single `llama_decode` call, the same fused
  prefill+decode batching that llama-server and vLLM's chunked prefill use. Greedy sampling
  reads each sequence's logits on the host.
- **The scheduler owns KV memory.** llama.cpp gets one unified KV pool sized to the Engine's
  `kv_blocks × 16` tokens (plus a small slack, explained below). The `PagedKvCache` decides what
  is admitted and what is preempted, so llama.cpp never runs out of cells. Preemption frees the
  victim's cells and later recomputes its prompt, which is exactly what the cost model charged.
- **Prefix-cache hits reuse real KV.** When the radix cache matches blocks, the executor shares
  those KV cells with the new sequence through `llama_memory_seq_cp`, which only tags cells and
  copies no data. When the sequence that computed a cached block leaves the batch, the block's
  cells are *parked* in a holder sequence, so the KV lives exactly as long as the radix cache
  references the block. `PagedKvCache` reports each block whose reference count reaches zero, and
  the executor drops its cells. If every holder is full, the block is dropped and recomputed on
  its next hit, and the Engine's statistics count it as computed, not reused.
- **Only computed KV is shared.** A request whose prefix match includes blocks admitted in the
  same iteration waits one iteration, because those blocks are only computed by that iteration's
  model call. This matches vLLM, which caches only computed blocks. The cost-model path keeps its
  original behaviour, so P1–P9 numbers are unchanged.
- **Fixed-length outputs.** EOS is ignored and every request produces exactly
  `max_output_tokens` tokens, the vLLM benchmark `--ignore-eos` convention. Throughput then
  compares schedulers, not how chatty a model is.
- **Synthetic prompts, real compute.** Requests still carry token counts and block hashes, not
  text. The executor derives prompt token ids deterministically from the block hashes, so
  requests sharing a `prefix_id` share identical prefix tokens and reusable KV, and every other
  prompt differs. This is the "random tokens" dataset vLLM's benchmark uses: the attention and
  matmul work is the same as for real text.

Slack cells: a request whose whole prompt matched still recomputes its last token, because the
model needs that token's logits. Sequences that share a partially filled tail block each append
to their own copy of it. Both cost at most a block per running sequence, so the llama.cpp pool
is the scheduler's capacity plus two blocks per running sequence.

Two llama.cpp settings matter for throughput, and both were found by measuring:

- **Sampling on the device.** With host-side greedy sampling, every step copies each running
  sequence's 151,936-entry logits row to the host and takes the argmax there. On the H200 with
  Qwen3-8B that was about 20% of wall time, and the GPU was busy only 51–61% of the time. The
  executor therefore attaches a greedy sampler chain to each sequence id (llama.cpp's backend
  sampling), so only the chosen token comes back. A test checks that it picks exactly the tokens
  the host argmax picks. llama-server is run with its matching `--backend-sampling` flag.
- **CUDA graphs off.** At b12188, llama.cpp's CUDA-graph replay fails on the H200 with "an illegal
  instruction was encountered" (`ggml_cuda_graph_launch_chunk`) once one step carries many
  sequences' prompts. Request-level batching does that by design, and so does continuous
  batching with a 65,536-token step and 64 prompts. Each failing case completes with
  `GGML_CUDA_DISABLE_GRAPHS=1`. Small steps (4,096 tokens, 16 prompts) work with graphs on, which
  is how the first continuous run passed. Every llama.cpp run below, InferSched and llama-server
  alike, has graphs off. One extra run measures what that costs where graphs are safe. vLLM keeps
  its own CUDA graphs.

### Correctness checks

`cpp/tests/llama_executor_test.cpp` runs the Engine on a real 260K-parameter model (CPU, in CI)
and checks:

- every scheduled token comes out of the model;
- a request that reuses a cached prefix produces the **same greedy continuation** as the same
  request computed from scratch;
- preemption under KV pressure still completes every token;
- cancellation frees the running sequence's cells;
- prefix-cache eviction drops the parked KV.

After each test the scheduler and llama.cpp both hold zero KV. A model-free contract test,
`cpp/tests/executor_contract_test.cpp`, runs in every CI job. Under preemption, cancellation and
radix eviction, it checks that the Engine never decodes a released sequence, never matches a freed
block, and reports every block's death before it can be reused.
