# Local benchmark baseline

## Snapshot: 2026-07-13

Suite: real-time local CPU scheduler and concurrency microbenchmarks. These numbers do not include
Kafka, PostgreSQL, gRPC, network time, or GPU inference. Simulated compute costs are set to zero so
the scheduler's CPU overhead is measured directly.

- Machine: Apple M4 MacBook Air, 16 GiB RAM
- OS: Darwin arm64 25.5.0
- Compiler: Apple clang 21.0.0
- Build: CMake 4.4.0, `RelWithDebInfo`
- Workload seed: deterministic index-derived workload

| Experiment | Workload | Result |
|---|---:|---:|
| Continuous scheduler CPU path | 100,000 requests | 732,310 req/s |
| Legacy P1 runner (static algorithm + vector queue) | 10,000 requests | 34,373 req/s |
| P3 runner (continuous algorithm + heap queue), same 10k input | 10,000 requests | 780,381 req/s |
| Global-lock registry | 100,000 writes, 4 writers | 7.43M ops/s |
| 32-shard registry | 100,000 writes, 4 writers | 8.33M ops/s (+12.1%) |
| Mutex deque | 100,000 events, 4 producers | 41.76M events/s |
| Lock-free MPSC candidate | 100,000 events, 4 producers | 10.44M events/s (-75.0%) |

The MPSC result is intentionally retained as a rejected optimization. On this machine and workload,
atomic contention costs more than the short mutex critical section. The runtime must not claim that
this MPSC implementation improves throughput. It remains available for further profiling and for
workloads where producer critical sections or blocking behavior differ.

The scheduler also traversed request metadata representing 58.95M tokens/s. This is a **zero-compute
scheduler bookkeeping rate**, not model inference throughput, and must not be quoted as tokens/s on a
résumé or compared with GPU serving systems.

The legacy/P3 runner comparison changes both the batching algorithm and queue complexity
(vector-copy/sort/erase versus heap-backed admission). Its roughly 20× gap is therefore **not evidence
that continuous batching provides a 20× throughput gain**. Current batching evidence is the
deterministic P2 experiment where a late short request completes before a long active decode ends.
Registry and queue comparisons each change one implementation variable.

## Frozen P3 acceptance thresholds

Thresholds were fixed after the first baseline and before further P3 tuning:

- 100,000-request continuous run completes with no failed/lost requests.
- KV allocation returns to zero after cache cleanup.
- Continuous scheduler throughput is at least 500,000 req/s on the machine above.
- Sharded registry sustains at least 5,000,000 inserts/s with four writers.
- Concurrent MPSC stress delivers every event exactly once under TSan; no speedup is claimed.

P3 does **not** yet provide latency percentiles, utilization, stable queue-depth, starvation, or cache
hit-rate thresholds. P7 must run a nonzero calibrated compute model under open-loop arrivals, wire the
histogram into TTFT/end-to-end measurements, sweep cache workloads, and freeze those thresholds before
using any resulting p99, utilization, cache-hit, or batching-throughput number in résumé bullets.

Run the reproducible snapshot with:

```bash
cmake --preset default
cmake --build build/default --parallel
./scripts/run_bench.sh 100000
```

The one-variable static-versus-continuous comparison this section says is missing now exists: see
[`p9-compute-efficiency.md`](p9-compute-efficiency.md), which runs both policies on the same
`ContinuousEngine` data structures.

## P7 nonzero-compute open-loop baseline

Suite: deterministic fake-clock simulation. Arrival times are Poisson-distributed (seed 7); prompts
are 16–128 tokens, outputs 1–16 tokens, and 10% of requests are low priority. The illustrative
phenomenological model charges 100 µs + 10 µs/prompt token for prefill and 50 µs + 10 µs/active
sequence per decode iteration, with 2% seeded jitter. It is not calibrated to a GPU and is not a
claim about model inference throughput.

Latency uses a lock-free log2 histogram. Values in the table are **bucket upper bounds**, not
point estimates: for example, a reported 65.5 ms means the observed percentile lies in
`[32.8 ms, 65.5 ms)`. Résumé-facing text should call this approximately 65 ms bucketed p99.

| Offered QPS | Achieved QPS | Utilization | TTFT p99 | E2E p99 |
|---:|---:|---:|---:|---:|
| 250 | 250.3 | 30.8% | 8.2 ms | 8.2 ms |
| 500 | 500.6 | 56.7% | 16.4 ms | 32.8 ms |
| 750 | 750.9 | 78.0% | 65.5 ms | 65.5 ms |
| 1,000 | 1,001.0 | 94.7% | 65.5 ms | 131.1 ms |
| 1,500 | 1,097.2 | 99.9% | 4.19 s | 4.19 s |

Each row contains 10,000 requests. The 1,500-QPS row demonstrates saturation rather than a passing
operating point. `scheduled_event_depth` is not reported as runtime queue depth because this
deterministic harness preloads future arrivals.

At 250 offered QPS, controlled cache workloads produced request-level result-hit rate 19.7%, prefix
hit rate 80.0%, and both-cache rates 20.0%/75.0%. E2E p99 was 8.2 ms with caches disabled and 2.0 ms
with both enabled. These workloads intentionally repeat every fifth request or a three-block prefix;
the hit rates are workload properties, not general production expectations.

The 100,000-request, 750-QPS run completed with zero leaked KV blocks and an approximately 65 ms
bucketed e2e p99 (65.536 ms bucket upper bound). Histogram
enabled/disabled wall times were 151.0/155.7 ms in this snapshot, so no measurable telemetry overhead
was observed; noise makes the apparent negative overhead non-actionable.

### Frozen P7 thresholds

- At 750 offered QPS: 100% completion, zero KV leaks, achieved QPS at least 740, utilization at least
  70%, TTFT and e2e p99 bucket upper bounds at most 131.1 ms, low-priority TTFT p99 bucket upper
  bound at most 131.1 ms, and drain lag at
  most 100 ms.
- Designed result-cache workload: request-level hit rate at least 18%.
- Designed prefix-cache workload: request-level hit rate at least 70%.
- Histogram overhead at 100,000 requests: no more than 10% wall-time regression.

There is still no valid one-variable static-versus-continuous throughput result: the static runner
does not share the P3 heap data structures. No batching speedup number is claimed.

Run a row with:

```bash
build/default/cpp/bench/infersched_local_bench --mode open_loop \
  --requests 10000 --qps 750 --seed 7 --cache none --telemetry enabled
```

## P7 distributed end-to-end baseline (historical, serial Router)

The separate live suite covers HTTP → FastAPI → Kafka → serial durable Router → gRPC Engine →
PostgreSQL/result outbox → Kafka → FastAPI. On the same machine, 100 open-loop requests at 50 offered
QPS completed without rejection at 46.5 achieved QPS and 125.6 ms e2e p99. A 100-request closed-loop
run achieved 30.9 QPS and 60.7 ms p99. These are small integration baselines, not the 100,000-request
scheduler result and not a distributed scaling curve.

These numbers are superseded by the P8 section below. They were measured with a single-process
load generator that serialized submissions and busy-polled results; P8 found that this tooling
saturates its own CPU near 100 req/s, so the historical figures bound the tooling as much as the
system.

## P8 distributed path: snapshot 2026-11-19

What changed between the two snapshots is described in
[`p8-multi-engine-routing.md`](p8-multi-engine-routing.md): concurrent dispatch across an Engine
fleet, continuous admission across RPCs, batched Router ingest, and corrected load-generation
methodology.

Environment (different from the Apple M4 snapshot above):

- Host: Intel Core i7-12700H (20 logical CPUs), 61 GiB RAM, Ubuntu 26.04, kernel 7.0.
- Shared host: other unrelated workloads (Kubernetes-in-Docker clusters, Terraform runs) were active
  during measurement. Load average is recorded per run; numbers were taken at load average 2–6
  on 20 CPUs. Treat tails as noisy.
- Runtime container: Ubuntu 24.04, GCC 13.3, gRPC 1.51, librdkafka 2.3, hiredis 1.2, libpqxx 8.0.1,
  Python 3.12. Kafka 3.7.2 (KRaft, one broker), PostgreSQL 16.16 with default `synchronous_commit`,
  Redis 7, all via `docker compose` on the same host.
- Engines run the cost model with zero simulated compute plus `--batch-delay-ms 2` per iteration,
  so each request spends roughly 10–70 ms in decode (4–32 output tokens). This is a scheduling and
  coordination benchmark, not model inference.

### Router throughput, before and after

A C++ client publishes a burst of requests straight to Kafka and waits for every result
(`scripts/run_router_burst.sh BIN_DIR ENGINES COUNT`: one durable Router, Engines with zero simulated
compute, so this measures the Router/Kafka/Postgres/gRPC coordination path alone). The before row
used the original client, which publishes one acknowledged message at a time; that client is not
the bottleneck at 68 req/s.

| Build | Engines | Requests | Wall time | Throughput |
|---|---:|---:|---:|---:|
| Before P8 (commit `9d22bd6`, serial dispatch) | 1 | 1,000 | 14.6 s | 68 req/s |
| P8 | 1 | 5,000 | 3.42 s | 1,460 req/s |
| P8 | 2 | 5,000 | 3.47 s | 1,439 req/s |
| P8 | 4 | 5,000 | 3.47 s | 1,440 req/s |

Two trials of the before row gave 68 req/s both times. Wall time includes client start-up and
consumer-group join. With zero compute, adding Engines does not help: the Router's main thread
(Kafka ingest, outbox relay, offset commits) and Postgres commits are the bound.

### Full HTTP path (gateway → Kafka → Router → 2 Engines → Postgres → Kafka → gateway)

`scripts/run_p7_distributed_bench.sh REQUESTS open QPS` with the defaults: one Router (128 dispatch
workers, 16 Postgres connections), two Engines, Poisson arrivals, 50 unmeasured warm-up requests,
and client load split across processes (no client saturated its CPU in any row). Latency is
client-observed (POST → result), exact percentiles over all completed requests.

| Offered QPS | Requests | Achieved QPS | e2e p50 | e2e p90 | e2e p99 | Submit p99 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 3,000 | 98.8 | 51.4 ms | 75.0 ms | 86.9 ms | 10.7 ms | 0 |
| 300 | 3,000 | 278.6 | 54.3 ms | 80.1 ms | 94.0 ms | 11.7 ms | 0 |
| 600 | 6,000 | 558.4 | 58.2 ms | 84.3 ms | 97.7 ms | 12.9 ms | 0 |
| 800 | 6,000 | 712.1 | 313.1 ms | 456.6 ms | 498.4 ms | 40.1 ms | 0 |

The 800-QPS row is past saturation (queueing visible in p50), so this configuration sustains
roughly 600–700 QPS end to end. Router-side spans at that point show dispatch itself unchanged
(p50 ≈ 40 ms); the queueing is in front of the Router (single gateway process plus Router ingest).
Achieved QPS is completions divided by the whole window, including the final requests' latency, so
it reads slightly below the offered rate even when nothing queues.

Per-stage breakdown from `inference.traces` at 100 offered QPS (1,000 requests, before the warm-up
was added, which is why p99 ingest is high: those are the first requests after assignment):

| Span | p50 | p90 | p99 |
|---|---:|---:|---:|
| `gateway.submit_ack` (admission → broker ack) | 1.6 ms | 2.4 ms | 3.4 ms |
| `router.ingest` (CreateTime → enqueued) | 3.8 ms | 8.7 ms | 356 ms |
| `router.dispatch` (one attempt on an Engine) | 36.7 ms | 60.6 ms | 66.9 ms |
| `router.request` (enqueued → finalized) | 41.6 ms | 66.5 ms | 132 ms |
| `gateway.result_transit` (outbox publish → gateway) | 2.3 ms | 2.9 ms | 4.2 ms |

### What these numbers do not show

- No GPU or model execution; Engine time is a fixed per-iteration delay.
- One Router and one gateway process on one host. This is not a horizontal scaling curve; adding
  Engines raises dispatch capacity but not the gateway or Router-ingest ceiling.
- Shared-host noise affects tails. Re-run on a dedicated machine before quoting a p99.

## P9 compute efficiency

Batching policy, routing, prefix affinity, and cancellation measured as modeled compute per
delivered token: [`p9-compute-efficiency.md`](p9-compute-efficiency.md).
