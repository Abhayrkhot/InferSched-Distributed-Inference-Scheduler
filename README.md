# InferSched - Distributed Inference Scheduler

A two-tier, ML-infra-style inference **scheduling runtime** built to coordinate 100,000+
*simulated* inference requests:

- a **Router** fleet for routing, ownership, and fault tolerance; and
- **Engine** replicas running iteration-level continuous batching on worker-owned paged KV caches.

The system integrates **gRPC** for Router↔Engine communication, **Kafka** for ingestion and results,
**PostgreSQL** as the durable source of truth, **Redis** for ephemeral coordination experiments, and
a **Python/FastAPI** control plane.

> Engines use a documented phenomenological latency model; they do not execute real neural-network
> inference. Benchmark results characterize scheduling behavior rather than GPU throughput. See
> [`docs/benchmarks.md`](docs/benchmarks.md).

Design and measurement details are available in [`docs/`](docs/), including distributed execution,
the durable pipeline, multi-Router recovery, control-plane observability, and benchmark methodology.

## Layout

```
proto/           Router <-> Engine gRPC contract (infersched.proto)
cpp/             C++ runtime, integrations, tests, and benchmarks
python/          FastAPI gateway, load generator, and dashboard
docs/            architecture decisions and benchmark reports
scripts/         integration, chaos, and benchmark runners
docker-compose.yml   Kafka, Redis, and PostgreSQL
```

## Toolchain

- CMake ≥ 3.20 and a C++20 compiler
- gRPC, protobuf, abseil, librdkafka, and libpqxx 8
- Docker and Docker Compose
- Python 3.11+

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
