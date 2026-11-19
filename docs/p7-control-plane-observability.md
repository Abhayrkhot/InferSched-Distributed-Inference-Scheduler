# P7 control plane and observability

P7 closes the loop around the C++ runtime without changing its reliability boundary. The FastAPI
gateway accepts validated inference metadata, assigns request/trace/span identifiers, and publishes
protobuf requests to Kafka. A result consumer correlates the durable result envelope, while the
PostgreSQL lookup remains a fallback when an in-memory gateway result is absent.

## Overload and lifecycle

`INFERSCHED_GATEWAY_CAPACITY` bounds the gateway's in-flight request map. A submission beyond that
limit receives HTTP 429 and is not published. Since P8 a submission returns 202 only after the
Kafka broker acknowledges it (503 otherwise), pending entries expire after
`INFERSCHED_RESULT_TIMEOUT_S`, and the result cache is bounded (`INFERSCHED_RESULT_CACHE`).

- `/metrics`: Prometheus exposition (request counters, queue depth and capacity, e2e latency
  histogram, event-loop lag).
- `/v1/stats`: the same figures as JSON, plus HTTP-observed e2e p50/p90/p99.
- `/healthz`: liveness. `/readyz`: the results consumer is positioned; submissions are refused
  until then.
- `GET /v1/inference/{id}?wait_ms=N`: long-poll for a result (up to 30 s).
- `/`: a small dashboard over `/v1/stats`.

This is deliberately a control-plane demo, not a production authentication, quota, or multi-tenant
implementation.

The load generator supports two distinct modes:

- `open`: Poisson arrivals at the requested offered QPS, independent of completions.
- `closed`: `--concurrency` outstanding requests, useful for the no-queue latency floor.

Load is split across `--processes` client processes (default: one per 40 offered QPS) because one
asyncio + httpx process saturates its CPU near 100 req/s and would report its own queueing as
latency; the summary includes `client_cpu_saturated`. `--warmup N` sends unmeasured requests first,
and `--prefixes K --prefix-tokens T --prefix-fraction F` generate shared-prefix traffic.

Every successful response must echo the originally assigned `trace_id`. The same trace/span fields
travel in `InferenceRequest`, gRPC progress events, and the Kafka result envelope alongside the
PostgreSQL ownership epoch and attempt id.

## Running it

```bash
docker compose up -d
PYTHONPATH=python .venv/bin/python -m uvicorn infersched_api.app:app
PYTHONPATH=python .venv/bin/python python/loadgen.py --mode open --requests 100 --qps 50
```

For a self-contained process run:

```bash
./scripts/run_p7_distributed_bench.sh 100 open 50
./scripts/run_p7_distributed_bench.sh 100 closed 50
```

The benchmark gives its Router a fresh Kafka group with `latest` reset and waits for assignment, so
old test-topic traffic is excluded. Normal durable operation keeps the safer `earliest` default.

## Scope and limitations

As of P8 the durable Router dispatches concurrently across an Engine fleet; see
[`p8-multi-engine-routing.md`](p8-multi-engine-routing.md). Distributed wall-clock QPS must still not
be compared with the in-process scheduler CPU benchmark. Gateway correlation is process-local
(one gateway process); authentication, quotas, and PostgreSQL/outbox retention remain out of scope.
The dashboard is intentionally dependency-free HTML rather than an operational monitoring stack.

During P7 integration, live protobuf requests exposed an invalid schema assumption: arbitrary
serialized protobuf bytes had been stored in PostgreSQL `TEXT`. The durable schema and pqxx bindings
now use `BYTEA`; restart recovery reconstructs the original bytes, and DLQ publication hex-encodes
binary poison payloads.
