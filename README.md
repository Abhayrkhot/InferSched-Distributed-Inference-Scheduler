# InferSched — Distributed Inference Scheduler

A two-tier, ML-infra-style inference **scheduling runtime** built to coordinate 100,000+
*simulated* inference requests with defensible correctness guarantees:

- a **Router** fleet (routing, ownership, fault tolerance) over
- **Engine** replicas that each run **iteration-level continuous batching** on a worker-owned
  **paged KV-cache**,

integrated with **gRPC** (Router↔Engine), **Kafka** (ingestion / results / traces),
**Redis** (ephemeral coordination), and **PostgreSQL** (durable source of truth), and driven /
observed by a **Python** control plane.

> **Simulated compute.** Engines model inference cost with a documented, *phenomenological*
> token-based latency model — they do **not** run real models. All numbers characterize the
> **scheduler**, not a GPU. The P7 parameters are illustrative and are not calibrated to GPU
> measurements. See [`docs/benchmarks.md`](docs/benchmarks.md).

## Guarantees we actually claim

- **At most one accepted terminal result per `request_id`** (Postgres uniqueness + attempt/epoch
  fencing). We do *not* claim zero duplicate execution or zero duplicate Kafka delivery — under
  at-least-once both are possible; consumers are idempotent → **effectively-once results**.
- **Zero leaked KV blocks** after every run (property-tested invariant).
- Real multi-process distribution, but **no consensus/leader-election**: coordination is Kafka
  partition ownership + Postgres fencing + Redis TTL hints.

The design and measured guarantees are documented across [`docs/`](docs/), including distributed
execution, the durable pipeline, multi-Router recovery, and benchmark methodology.

## Status

**Phases 0–7 complete (v0.7.0).** The build/protocol scaffold is verified, and the Engine runtime
now includes both the P1 static baseline and P2 continuous runner:

- a validated Engine-local execution state machine;
- a fake monotonic clock, seeded integer cost model, and pluggable static-batch policy using FCFS
  ordering with greedy token-budget packing;
- transactional paged-KV allocation with shared-prefix refcounts and prefix pins;
- separate bounded LRU result and prefix caches with hit/miss counters, plus a typed result-key
  builder that includes model, tokenizer, prompt, sampling parameters, and seed;
- seeded randomized KV tests that check accounting invariants after every operation and zero leaked
  blocks after cleanup.

The integrated runner composes static batching, KV admission, prefix-prefill reuse, result-cache
bypass, and seeded prefill/decode timing into a repeatable execution timeline. The continuous runner
admits requests between decode steps, grows KV incrementally, preempts under live memory pressure,
and performs block-aligned longest-prefix matching across different prompts.

The P1 static baseline intentionally retains exact-prompt KV reuse so its behavior stays stable for
comparison. P2 uses optional per-block prompt hashes for true-prefix matching. Both correctness
runners favor clarity and determinism with vector copies/sorts/erasures; P3 replaces those paths with
indexed/concurrent queues before the 100,000-request benchmark.

Continuous-runner metrics distinguish unique requests from execution events: prefix hit/miss counts
record one outcome per unique successfully admitted request with block hashes, while `admissions`
counts every admission event, including re-admission after preemption. P3 replaces P2's intentionally
simple O(P²) materialized-prefix representation with a one-node-per-block radix/tree index.

P3 adds a heap-backed ready queue, one-node-per-block radix prefix index, bounded lock-free MPSC
candidate, sharded concurrent registry, atomic log histogram, and a wall-clock 100,000-request
benchmark. The measured MPSC candidate is slower than the mutex baseline on the development machine,
so it is documented as a rejected optimization rather than used to support a throughput claim. See
[`docs/benchmarks.md`](docs/benchmarks.md).

The P3 benchmark measures **CPU scheduler overhead with simulated compute set to zero**. Its token
bookkeeping rate is not inference tokens/s, and its legacy-static/P3-continuous gap also includes a
vector-to-heap queue change. Batching evidence currently comes from the controlled fake-clock latency
test. P7 adds separate open-loop, nonzero-compute latency/utilization/cache measurements.

P4 adds real Router and Engine OS processes over gRPC. Engines register and heartbeat to a
Router-hosted control service; Routers dispatch fenced attempts to Engine-hosted streaming services.
Integration tests cover concurrent streams, stale attempt/partition-epoch rejection, graceful drain,
SIGKILL of an in-flight Engine, new-incarnation registration, and retry with a higher fence. The P4
lease view is in Router memory; Redis-backed hints and PostgreSQL-authoritative completion begin in P5.
Overlapping retries for a request already live on the same Engine are rejected without changing its
fence. P4 does not yet claim load/prefix-aware routing or authoritative effectively-once completion.

P5 adds PostgreSQL-authoritative partition epochs, attempts, versioned lifecycle state, conditional
completion fencing, atomic result/DLQ outbox writes, restart recovery, and a broker-acknowledged Kafka
relay. Kafka polling runs on a dedicated thread with auto-commit disabled; offsets advance only from
the contiguous finalized/published watermark. See
[`docs/p5-durable-pipeline.md`](docs/p5-durable-pipeline.md).

P6 adds Kafka consumer-group partition sharding across multiple Router processes, explicit
assignment/revocation handling, PostgreSQL epoch handoff, and assigned-partition recovery. Its chaos
test kills one Router and Engine, flushes Redis, and verifies all requests and unique authoritative
results through the surviving Router. See
[`docs/p6-multi-router-rebalancing.md`](docs/p6-multi-router-rebalancing.md).

P7 adds a bounded FastAPI gateway with explicit HTTP 429 overload behavior, open- and closed-loop
Python load generation, request/trace/span/fence propagation, a live metrics dashboard, and
request-level cache accounting. The deterministic 100,000-request suite reports TTFT/e2e
percentiles and utilization under a documented nonzero cost model; a separate live benchmark covers
the complete HTTP→Kafka→Router→gRPC→Postgres/outbox→Kafka path. See
[`docs/p7-control-plane-observability.md`](docs/p7-control-plane-observability.md) and
[`docs/benchmarks.md`](docs/benchmarks.md).

Simulation latency percentiles are log2-histogram bucket upper bounds. Accordingly, résumé-facing
latency is reported approximately (for example, ~65 ms bucketed p99), not with false decimal
precision.

## Layout

```
proto/           Router <-> Engine gRPC contract (infersched.proto)
cpp/             C++ runtime (Engine core, Router, integrations, tests, benchmarks)
python/          load generator, FastAPI gateway, dashboard
docs/            architecture, design decisions, benchmarks, resume mapping
docker-compose.yml   kafka (kraft), redis, postgres
```

## Toolchain

- CMake ≥ 3.20, a C++20 compiler (Apple clang / clang / gcc)
- gRPC + protobuf, abseil (Router↔Engine contract)
- Docker + Docker Compose (Kafka / Redis / Postgres; Postgres host port `55432`)
- Python 3.11+ (control plane)

Per-phase client libraries (librdkafka, redis-plus-plus, libpqxx) are installed when their
integration phase begins.

## Build & test

```bash
# Configure + build (RelWithDebInfo) and run the test suite
cmake --preset default
cmake --build build/default
ctest --preset default

# Sanitizer builds are SEPARATE configs (never combined in one binary):
cmake --preset asan && cmake --build build/asan && ctest --preset asan   # AddressSanitizer
cmake --preset tsan && cmake --build build/tsan && ctest --preset tsan   # ThreadSanitizer
```

## Infrastructure (for later phases)

```bash
docker compose up -d      # kafka (kraft), redis, postgres
docker compose down -v
```
