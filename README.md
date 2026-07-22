# InferSched - Distributed Inference Scheduler

InferSched is a two-tier, ML-infrastructure-style scheduling runtime that models how a distributed
inference service accepts, batches, executes, retries, and completes large request workloads.

The project combines:

- a Python/FastAPI gateway for admission, backpressure, tracing, and results;
- Kafka for durable request transport and Router partition sharding;
- a C++ Router fleet for ownership, dispatch, retry, and recovery;
- separate C++ Engine processes running iteration-level continuous batching;
- paged KV memory, radix prefix caching, result caching, and preemption;
- PostgreSQL for authoritative lifecycle state, fencing, and the transactional outbox; and
- Redis for disposable coordination hints that are not part of the correctness boundary.

> InferSched does not execute a neural network or GPU kernel. Engines use a documented,
> deterministic phenomenological cost model. Scheduler throughput and simulated latency results
> characterize the scheduling system, not model inference throughput.

## What the project demonstrates

InferSched addresses four connected systems problems:

1. **Efficient scheduling:** continuously admit work between decode iterations instead of waiting
   for an entire static batch to drain.
2. **Memory-aware execution:** allocate reference-counted KV pages, reuse block-aligned prompt
   prefixes, and preempt work safely under memory pressure.
3. **Distributed recovery:** shard requests across Routers with Kafka and recover unresolved work
   after Router or Engine failure.
4. **Authoritative completion:** allow at-least-once execution while accepting at most one current,
   fenced completion in PostgreSQL.

## System architecture

View the [live InferSched architecture](https://abhayrkhot.github.io/InferSched-Distributed-Inference-Scheduler/)
for the native HTML/CSS system diagram, request lifecycle, component ownership, implementation
guide, and measured results. The source is available in [`architecture.html`](architecture.html).

## End-to-end request path

1. The FastAPI gateway validates a request, assigns request and trace identifiers, and publishes a
   protobuf message to `inference.requests`.
2. Kafka assigns the request partition to one Router in the consumer group.
3. The Router durably records the request, acquires the current partition epoch, assigns an attempt,
   and dispatches it to an Engine over streaming gRPC.
4. The Engine admits the request into its continuous batch, manages its paged KV allocation, and
   reports progress and completion with the original fence.
5. PostgreSQL accepts the completion only if its partition, ownership epoch, Engine, attempt, and
   lifecycle state are all current.
6. The accepted result and outbox publication intent commit in one transaction. The outbox relay
   publishes the result to Kafka and marks it published only after broker acknowledgement.
7. The gateway correlates the result and returns it to the client.

## Measured results

All figures below are reproducible local baselines on the documented Apple M4 test machine. See
[`docs/benchmarks.md`](docs/benchmarks.md) for hardware, workloads, commands, limitations, and frozen
acceptance thresholds.

| Scope | Workload | Result |
|---|---:|---:|
| Scheduler CPU path, simulated compute disabled | 100,000 requests | 732,310 scheduler req/s |
| Nonzero-cost open-loop simulation | 10,000 requests at 750 offered QPS | 750.9 achieved QPS, 78.0% utilization |
| Nonzero-cost open-loop simulation | Same 750-QPS workload | approximately 65 ms bucketed TTFT and E2E p99 |
| Large open-loop correctness run | 100,000 requests at 750 QPS | 100% completion, zero leaked KV blocks |
| Controlled prefix-cache workload | 250 offered QPS | 80.0% request-level prefix hit rate |
| Controlled both-cache workload | 250 offered QPS | E2E p99 reduced from 8.2 ms to 2.0 ms |
| Full distributed integration path | 100 open-loop requests at 50 offered QPS | 46.5 achieved QPS, 125.6 ms E2E p99 |

The scheduler CPU result uses zero simulated compute and must not be interpreted as neural-network
or GPU inference throughput. Latency percentiles come from a power-of-two histogram and are bucket
upper bounds. The distributed result is a small correctness and integration baseline, not a scaling
curve.

## Reliability model

- **Delivery:** Kafka ingestion is at least once.
- **Deduplication:** PostgreSQL constraints make request ingestion idempotent.
- **Fencing:** every attempt carries `(partition_id, ownership_epoch, attempt_id)`.
- **Finalization:** one conditional PostgreSQL update accepts only the current dispatched attempt.
- **Publication:** a transactional outbox prevents an unsafe PostgreSQL and Kafka dual write.
- **Recovery:** a newly assigned Router reconstructs unresolved requests and redispatches them with
  a new epoch and attempt.
- **Cache loss:** Redis loss cannot invalidate authoritative PostgreSQL state.

The result is at-most-one authoritative accepted completion per request ID, not exactly-once physical
execution or exactly-once Kafka delivery.

## Repository layout

```text
proto/              Router <-> Engine gRPC contract
cpp/                C++ runtime, integrations, tests, and benchmarks
python/             FastAPI gateway, load generator, and dashboard
docs/               Architecture decisions and benchmark reports
scripts/            Integration, chaos, and benchmark runners
docker-compose.yml  Kafka, Redis, and PostgreSQL
```

## Toolchain

- CMake 3.20 or newer and a C++20 compiler
- gRPC, protobuf, abseil, librdkafka, and libpqxx 8
- Docker and Docker Compose
- Python 3.11 or newer

## Build and test

```bash
cmake --preset default
cmake --build build/default
ctest --preset default

# Sanitizers use separate binaries.
cmake --preset asan && cmake --build build/asan && ctest --preset asan
cmake --preset tsan && cmake --build build/tsan && ctest --preset tsan
```

## Infrastructure

```bash
docker compose up -d
docker compose down -v
```

## Further reading

- [Live architecture](https://abhayrkhot.github.io/InferSched-Distributed-Inference-Scheduler/): rendered interactive architecture viewer
- [`docs/p4-distributed-execution.md`](docs/p4-distributed-execution.md): process separation and gRPC fencing
- [`docs/p5-durable-pipeline.md`](docs/p5-durable-pipeline.md): authoritative state and transactional outbox
- [`docs/p6-multi-router-rebalancing.md`](docs/p6-multi-router-rebalancing.md): rebalance and crash recovery
- [`docs/p7-control-plane-observability.md`](docs/p7-control-plane-observability.md): gateway, load generation, and metrics
- [`docs/benchmarks.md`](docs/benchmarks.md): reproducible measurements and claim boundaries
