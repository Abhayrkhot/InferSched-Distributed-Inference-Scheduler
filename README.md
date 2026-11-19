# InferSched - Distributed Inference Scheduler

InferSched is a two-tier, ML-infrastructure-style scheduling runtime that models how a distributed
inference service accepts, batches, executes, retries, and completes large request workloads.

The project combines:

- a Python/FastAPI gateway for admission, backpressure, tracing, and results;
- Kafka for durable request transport and Router partition sharding;
- a C++ Router fleet for ownership, load- and prefix-aware dispatch across an Engine fleet,
  retry, cancellation, and recovery;
- separate C++ Engine processes running iteration-level continuous batching across RPCs;
- paged KV memory, radix prefix caching, result caching, and preemption;
- PostgreSQL for authoritative lifecycle state, fencing, and the transactional outbox;
- Redis for disposable prefix-affinity hints that are not part of the correctness boundary; and
- span export to Kafka (`inference.traces`) plus Prometheus metrics at the gateway.

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
5. **Fleet dispatch:** keep many attempts in flight per Router, route each one by Engine load and
   cached-prefix affinity, retry failures on another Engine, and cancel work nobody will accept.

## System architecture

View the [live InferSched architecture](https://abhayrkhot.github.io/InferSched-Distributed-Inference-Scheduler/)
for the native HTML/CSS system diagram, request lifecycle, component ownership, implementation
guide, and measured results. The source is available in [`architecture.html`](architecture.html).

## End-to-end request path

1. The FastAPI gateway validates a request, assigns request and trace identifiers, publishes a
   protobuf message to `inference.requests`, and returns 202 only after the broker acknowledges it.
2. Kafka assigns the request partition to one Router in the consumer group.
3. The Router durably records the request, chooses an Engine (least loaded, unless a Redis hint
   names an Engine incarnation that already holds the request's prompt prefix), assigns a fenced
   attempt, and dispatches it over streaming gRPC. Many attempts are in flight at once.
4. The Engine admits the request into its running batch at the next iteration boundary, manages its
   paged KV allocation, and streams first-token and completion events with the original fence. A
   failed or lost attempt is retried on another Engine; an unrecoverable one is recorded as a fenced
   terminal failure.
5. PostgreSQL accepts the completion only if its partition, ownership epoch, Engine, attempt, and
   lifecycle state are all current.
6. The accepted result and outbox publication intent commit in one transaction. The outbox relay
   publishes the result to Kafka and marks it published only after broker acknowledgement.
7. The gateway correlates the result and returns it to the client, which long-polls
   `GET /v1/inference/{id}?wait_ms=…`.

## Measured results

The in-process figures below are reproducible local baselines on the documented Apple M4 test
machine. See [`docs/benchmarks.md`](docs/benchmarks.md) for hardware, workloads, commands,
limitations, and frozen acceptance thresholds.

| Scope | Workload | Result |
|---|---:|---:|
| Scheduler CPU path, simulated compute disabled | 100,000 requests | 732,310 scheduler req/s |
| Nonzero-cost open-loop simulation | 10,000 requests at 750 offered QPS | 750.9 achieved QPS, 78.0% utilization |
| Nonzero-cost open-loop simulation | Same 750-QPS workload | approximately 65 ms bucketed TTFT and E2E p99 |
| Large open-loop correctness run | 100,000 requests at 750 QPS | 100% completion, zero leaked KV blocks |
| Controlled prefix-cache workload | 250 offered QPS | 80.0% request-level prefix hit rate |
| Controlled both-cache workload | 250 offered QPS | E2E p99 reduced from 8.2 ms to 2.0 ms |

Distributed path, measured on a separate shared Linux host (i7-12700H, 20 CPUs; see the P8 section
of [`docs/benchmarks.md`](docs/benchmarks.md) for environment and caveats):

| Scope | Workload | Result |
|---|---:|---:|
| Router coordination path, zero simulated compute | 1,000-request burst, original serial Router | 68 req/s |
| Router coordination path, zero simulated compute | 5,000-request burst, concurrent Router | ~1,450 req/s |
| Full HTTP → Kafka → Router → 2 Engines → Postgres → HTTP | 6,000 open-loop requests at 600 QPS | 558 achieved QPS, 58 ms p50, 98 ms p99, 0 errors |
| Same path past saturation | 6,000 requests at 800 QPS | 712 achieved QPS, queueing (313 ms p50) |

The scheduler CPU result uses zero simulated compute and must not be interpreted as neural-network
or GPU inference throughput. In-process latency percentiles come from a power-of-two histogram and
are bucket upper bounds; distributed percentiles are exact, client-observed, and taken after a
warm-up on a host shared with other workloads. The distributed results are single-Router,
single-gateway measurements, not a horizontal scaling curve.

## Reliability model

- **Delivery:** Kafka ingestion is at least once.
- **Deduplication:** PostgreSQL constraints make request ingestion idempotent.
- **Fencing:** every attempt carries `(partition_id, ownership_epoch, attempt_id)`.
- **Finalization:** one conditional PostgreSQL update accepts only the current dispatched attempt.
- **Publication:** a transactional outbox prevents an unsafe PostgreSQL and Kafka dual write.
- **Recovery:** a newly assigned Router reconstructs unresolved requests and redispatches them with
  a new epoch and attempt.
- **Retry and failover:** a rejected, failed, or lost attempt is retried with a new attempt id on a
  different Engine, with backoff and a bounded budget.
- **Terminal failure:** an unrecoverable request becomes `FAILED_TERMINAL` through an owner-fenced
  update that commits its failure result to the outbox; any later completion from an older attempt
  is rejected.
- **Cancellation:** a revoked partition, an expired deadline, or a closed stream cancels the exact
  fenced attempt on its Engine, which frees the sequence's KV blocks.
- **Cache loss:** Redis loss or `FLUSHALL` only costs prefix locality; it cannot invalidate
  authoritative PostgreSQL state.

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
- gRPC, protobuf, abseil, librdkafka, hiredis, and libpqxx 8
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

Tests labelled `live` need Kafka, PostgreSQL, and Redis (`docker compose up -d`); run only the
hermetic ones with `ctest --preset default -LE live`.

## Infrastructure

```bash
docker compose up -d
docker compose down -v
```

## Running the services

```bash
# Router: long-running until SIGTERM; prints a JSON stats line on exit.
build/default/cpp/infersched_durable_router --listen 127.0.0.1:50051 --redis 127.0.0.1:6379
# Engines: register with one or more Routers; --time-scale 1 paces the cost model in real time.
build/default/cpp/infersched_engine --listen 127.0.0.1:50061 --router 127.0.0.1:50051 --engine-id engine-1
build/default/cpp/infersched_engine --listen 127.0.0.1:50062 --router 127.0.0.1:50051 --engine-id engine-2
# Gateway and load.
PYTHONPATH=python .venv/bin/python -m uvicorn infersched_api.app:app --port 8000
PYTHONPATH=python .venv/bin/python python/loadgen.py --url http://127.0.0.1:8000 --qps 200 --requests 2000
```

Benchmarks: `scripts/run_p7_distributed_bench.sh` (full HTTP path) and `scripts/run_router_burst.sh`
(Router path only).

## Further reading

- [Live architecture](https://abhayrkhot.github.io/InferSched-Distributed-Inference-Scheduler/): rendered interactive architecture viewer
- [`docs/p4-distributed-execution.md`](docs/p4-distributed-execution.md): process separation and gRPC fencing
- [`docs/p5-durable-pipeline.md`](docs/p5-durable-pipeline.md): authoritative state and transactional outbox
- [`docs/p6-multi-router-rebalancing.md`](docs/p6-multi-router-rebalancing.md): rebalance and crash recovery
- [`docs/p7-control-plane-observability.md`](docs/p7-control-plane-observability.md): gateway, load generation, and metrics
- [`docs/p8-multi-engine-routing.md`](docs/p8-multi-engine-routing.md): multi-Engine dispatch, retries, cancellation, Redis hints, tracing
- [`docs/benchmarks.md`](docs/benchmarks.md): reproducible measurements and claim boundaries
