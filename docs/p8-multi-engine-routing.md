# P8 multi-Engine routing, concurrency, and coordination hints

P4–P7 proved the correctness boundary (fenced attempts, transactional outbox, Router rebalance)
but the durable path was a demonstration harness: each Router dispatched one request at a time to
the first Engine that registered, the Engine batched only per RPC, Redis held nothing, and the
Router exited after a fixed request count. P8 turns that path into a running service without moving
the correctness boundary.

## What changed

### Engine: continuous batching across RPCs

`EngineService` owns one `ContinuousEngine` on a single worker thread that runs iterations back to
back. A Dispatch RPC hands its request to the worker, which submits it at the next iteration
boundary, so a request that arrives while a long decode is running joins the live batch instead of
waiting for it (the P4 service drained its queue, ran `RunUntilIdle`, and only then looked at new
arrivals). Progress is streamed as it happens: `PREFILL_DONE` when the first token is decoded (TTFT),
then `COMPLETED` or `FAILED`.

- **Cancellation is real.** `Cancel` (fenced by epoch and attempt) and a broken or cancelled client
  stream both remove the sequence from the engine and free its KV blocks
  (`ContinuousEngine::Cancel`). Previously a cancelled attempt ran to completion.
- **Bounded memory.** Results are drained (`TakeResults`) instead of accumulating forever, per-request
  bookkeeping is erased on completion, and the remembered `(request_id → attempt)` fences are capped
  (`fence_memory`, default 65,536). Postgres stays the authoritative fence.
- **Pacing knobs.** `--batch-delay-ms` adds a fixed wall-clock delay per iteration (used by tests to
  hold work in flight); `--time-scale 1.0` sleeps the cost model's simulated time so simulated cost
  becomes real latency. Both default to 0.
- **Shared prefixes reach the engine.** `InferenceRequest` gained `prefix_id`/`prefix_tokens`; the
  Engine derives per-block hashes from them, so the radix prefix cache is exercised in the
  distributed path (it never was before — no block hashes crossed gRPC).
- **Router membership.** An Engine can register with several Routers (`--router a,b`) and
  re-registers when a heartbeat reports a lost lease or a Router restarts.

### Router: many Engines, many attempts in flight

`infersched_durable_router` now runs a pool of dispatch workers (`--workers`, default 128) sharing a
small Postgres pool (`--db-connections`, default 16). A worker leases a connection only for its
short fenced writes (`AssignAttempt`, `Finalize`, `FailTerminal`), never while it streams a Dispatch
RPC, so in-flight concurrency is not capped by Postgres `max_connections`.

Engine choice per attempt (`engine_selector.cpp`):

1. Load estimate = max(this Router's in-flight attempts on the Engine, the Engine's last heartbeat
   `running + waiting`), plus a penalty under 5% free KV. Engines without batch headroom are used only
   when nothing else is available.
2. A Redis prefix hint wins if it names a live Engine **with the same incarnation** and that Engine
   is within `--affinity-slack` sequences (default 8) of the least-loaded one.
3. Otherwise the least-loaded Engine; ties are broken by `hash(engine_id, request_id)`.

Selection and the in-flight reservation happen atomically under one lock, so concurrent workers see
each other's choices.

Failure handling:

| Engine response | Router action |
|---|---|
| `REJECTED` ack (queue full, draining, in flight elsewhere) | exclude that Engine for this request, back off, retry (bounded by `--max-rejections`) |
| `FAILED` retryable, or stream/transport error | exclude, exponential backoff, new attempt on another Engine (bounded by `--max-attempts`) |
| `FAILED` non-retryable (e.g. exceeds KV capacity) | owner-fenced `FAILED_TERMINAL` |
| retry budget exhausted | owner-fenced `FAILED_TERMINAL` with the last failure reason |
| no live Engine | wait; an Engine outage does not consume the retry budget |
| partition revoked mid-attempt | `TryCancel` the stream, fenced `Cancel` RPC, leave it to the new owner |
| deadline exceeded | fenced `Cancel` RPC so the Engine frees the sequence, then retry |

`DurableStore::FailTerminal` succeeds only while `(partition, ownership_epoch)` is the current owner,
and commits the failure result plus its outbox row in one transaction. A late completion from an
older attempt is then rejected by `Finalize`'s state check, so the result is still
at-most-one-authoritative. The gateway receives failures on `inference.results` with
`"status": "FAILED"` instead of waiting forever.

Service behaviour: `--requests 0` (now the default) and `--run-seconds 0` run until SIGTERM/SIGINT,
which drains in-flight attempts for up to `--drain-timeout-ms`, revokes the rest (they remain
unresolved for recovery), flushes the outbox, and prints a JSON stats line. Result envelopes are
built with proper JSON escaping.

### Redis coordination hints

Redis now holds prefix-affinity hints:

```text
key   infersched:prefix:v1:<model_revision>:<prefix_id>
value <engine_id>\n<incarnation_id>
TTL   120 s, refreshed by every completion that used the prefix (last writer wins)
```

Every operation is best effort (`RedisHints`): 25 ms timeouts, and after any error Redis is skipped
for a 1 s backoff instead of paying the timeout on every request. A hint for a different
incarnation is ignored because a restarted Engine has an empty prefix cache. Losing, flushing, or
poisoning Redis can only cost cache locality; every completion is still fenced in Postgres.

### Tracing

Routers emit spans to `inference.traces` (keyed by `trace_id`): `router.ingest` (producer CreateTime
→ enqueued), `router.dispatch` per attempt (Engine, attempt id, outcome, TTFT), and `router.request`
(enqueued → finalized). The gateway emits `gateway.request` (admission → result) with the broker-ack
and result-transit times. Export is fire-and-forget and never blocks request processing.

### Gateway

- `POST /v1/inference` returns 202 **only after the broker acknowledges** the request; an
  unacknowledged produce returns 503 and releases its admission slot.
- `GET /v1/inference/{id}?wait_ms=N` long-polls (up to 30 s) instead of client-side busy polling.
- Pending requests expire after `INFERSCHED_RESULT_TIMEOUT_S` (120 s) so a lost result cannot hold
  admission capacity forever; a late result is still kept. The result cache and latency window are
  bounded, and only this gateway's own requests are retained.
- The results consumer starts at each partition's high watermark when assigned. `/readyz` (and
  submission) wait for that, so a restart neither replays the topic's history nor misses a result.
  `/healthz` is liveness only.
- `/metrics` is Prometheus text exposition (counters, queue depth, e2e latency histogram, event-loop
  lag). The JSON view moved to `/v1/stats`.

### Bugs found by measuring

Building the distributed benchmark exposed several latent problems, each fixed and kept as a test or
a documented methodology rule:

- **Per-offset synchronous Kafka commits** on the poller thread stalled consumption and loaded the
  group coordinator. Commits are now coalesced per partition and flushed asynchronously at most every
  100 ms (synchronously on shutdown); a lost commit only causes a deduplicated redelivery.
- **`IsSourcePublished` was a sequential scan** of the outbox (no index on the source position), so
  its cost grew with every result ever written. Added `outbox_source_idx` and a partial index for
  unresolved requests, and batched both the ingest commit and the published check per poll batch.
- **Gateway result-topic replay.** A fresh consumer group with `earliest` re-read every historical
  result on each gateway start, competing with request handling for the GIL.
- **The load generator measured itself.** One asyncio + httpx process saturates its CPU near
  100 req/s of POST + long-poll and then reports its own queueing as system latency. The load
  generator now splits arrivals across processes and reports `client_cpu_saturated`. It also has an
  unmeasured warm-up, because the first requests after Router assignment wait for the consumer's
  first fetch.

## Tests

| Test | Proves |
|---|---|
| `EngineServiceTest.LateArrivalJoinsRunningBatchInsteadOfWaiting` | a 3-token request sent 200 ms into a 2 s decode finishes first |
| `EngineServiceTest.StreamsFirstTokenBeforeCompletion` | `PREFILL_DONE` arrives ≥150 ms before `COMPLETED` |
| `EngineServiceTest.ClientCancellationFreesTheSequence` | `TryCancel` frees the sequence and its admission slot |
| `EngineServiceTest.FencedCancelRpcSupersedesOnlyTheMatchingAttempt` | a stale-attempt Cancel is refused; the current one supersedes |
| `EngineSelector.*` | load choice, exclusion, saturation and low-KV avoidance, incarnation-checked affinity within slack |
| `RedisHints.UnreachableRedisDegradesToNoHintsQuickly` | 100 calls against a dead Redis take <500 ms with one connect attempt |
| `RedisHintsLive.*` | live round trip, model-revision namespacing, TTL expiry |
| `PostgresDurability.OwnerFencedTerminalFailureBlocksLateCompletion` | deposed owner cannot fail a request; late success is rejected; one result row |
| `Distributed.MultiEngineFailoverExactlyOnce` | 60 requests over 3 Engines, the slow one SIGKILLed mid-flight: all 60 complete once, killed attempts retried, spans reach Kafka, SIGTERM exits cleanly |
| `Distributed.RedisPrefixAffinitySurvivesFlush` | hints written and followed, prefix KV reused, then `FLUSHALL` and every request still completes |
| `Distributed.FastApiGatewayEndToEnd` | 200 open-loop HTTP requests over two Engines, half with shared prefixes, all complete with their own trace ids; Prometheus counters agree |
| `python/tests/test_gateway.py` | broker-ack gating, 503 on unacknowledged produce, readiness, long-poll wake-up, expiry, bounded caches, foreign results ignored |

## Limits

- Engines still execute a phenomenological cost model, not a neural network.
- The gateway is one Python process with in-memory correlation; scaling it out needs sticky routing
  or a shared result store.
- The Router's main thread serializes Kafka ingest and completion bookkeeping. It is batched, but at
  the rates in `benchmarks.md` it is the next bottleneck after dispatch concurrency.
- Prefix hints are last-writer-wins per prefix; there is no cluster-wide cache directory, eviction
  feedback from Engines, or cost model for the cache-miss penalty.
- Authentication, quotas, and outbox/request retention remain out of scope.
