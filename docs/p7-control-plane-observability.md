# P7 control plane and observability

P7 closes the loop around the C++ runtime without changing its reliability boundary. The FastAPI
gateway accepts validated inference metadata, assigns request/trace/span identifiers, and publishes
protobuf requests to Kafka. A result consumer correlates the durable result envelope, while the
PostgreSQL lookup remains a fallback when an in-memory gateway result is absent.

## Overload and lifecycle

`INFERSCHED_GATEWAY_CAPACITY` bounds the gateway's in-flight request map. A submission beyond that
limit receives HTTP 429 and is not published. `/metrics` reports submitted, completed, rejected,
current queue depth, and HTTP-observed e2e p50/p90/p99. `/healthz` is a process-health check; `/` is a
small polling dashboard. This is deliberately a control-plane demo, not a production authentication,
quota, or multi-tenant implementation.

The load generator supports two distinct modes:

- `open`: Poisson arrivals at the requested offered QPS, independent of completions.
- `closed`: one outstanding request at a time, useful for the no-queue latency floor.

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

The durable Router demo dispatches synchronously and serially. Its wall-clock QPS must not be compared
with the in-process scheduler CPU benchmark. Gateway result retention is process-local and bounded
only by process lifetime; a production service needs TTL/size eviction, authentication, quotas, and
PostgreSQL/outbox retention. The dashboard is intentionally dependency-free HTML rather than an
operational monitoring stack.

During P7 integration, live protobuf requests exposed an invalid schema assumption: arbitrary
serialized protobuf bytes had been stored in PostgreSQL `TEXT`. The durable schema and pqxx bindings
now use `BYTEA`; restart recovery reconstructs the original bytes, and DLQ publication hex-encodes
binary poison payloads.
