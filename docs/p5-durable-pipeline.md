# P5 durable pipeline evidence

P5 moves lifecycle authority from process memory into PostgreSQL and uses Kafka only as an
at-least-once transport. PostgreSQL runs on host port `55432` in this repository to avoid colliding
with common local development databases.

## Authoritative state and fencing

`DurableStore` creates and owns four durable structures:

- `partition_ownership`: monotonically increasing, partition-specific ownership epochs.
- `inference_requests`: unique request/source positions, versioned lifecycle state, current Engine,
  attempt, fence, and accepted terminal result.
- `poison_messages`: raw payload and parse failure keyed by Kafka partition/offset.
- `outbox`: unique result/DLQ publication intent with a nullable broker-ack timestamp.

Attempt assignment is conditional on the current PostgreSQL partition epoch. Completion is a
conditional update requiring the current `(partition, epoch, engine, attempt)` and `DISPATCHED`
state. The accepted completion and its result outbox row are written in one transaction. A stale or
duplicate completion changes neither table. General lifecycle transitions use `state_version`
compare-and-swap semantics, and terminal states are immutable.

## At-least-once ingestion and publication

`KafkaPoller` owns librdkafka polling on a dedicated thread, with auto-commit disabled. Processing is
decoupled through a queue, and safe contiguous offsets are returned to the poll thread for commit.
`RequestIngestor` parses protobuf requests, deduplicates delivery through PostgreSQL constraints, and
atomically records poison payload plus DLQ outbox intent after bounded parse retries.

`OutboxRelay` reads unpublished rows in order, publishes using an idempotent Kafka producer, and sets
`published_at` only after the delivery callback acknowledges the message. If the process dies after
the database transaction but before publication, a restarted relay sees and republishes the row. If
the broker received a message but its acknowledgement was lost, publication may repeat; downstream
consumers must deduplicate by message key. This is effectively-once result authority, not exactly-once
Kafka delivery.

## Automated evidence

The P5 tests use live PostgreSQL and Kafka from `docker-compose.yml` and cover:

- monotonic ownership and attempt fencing, including rejection of stale completion;
- optimistic `state_version` rejection and terminal immutability;
- duplicate Kafka delivery and poison-to-DLQ idempotency;
- recovery of unresolved requests after reconnect;
- process-level startup redispatch of unresolved requests with a fresh epoch/attempt;
- terminal duplicate redelivery detection and safe published-offset release;
- crash-after-transaction/before-publish recovery;
- retention of an outbox row when publication is not acknowledged;
- real Kafka broker acknowledgement before marking an outbox row published;
- a dedicated Kafka poll thread with explicit offset commit handoff;
- contiguous offset advancement across out-of-order completion;
- heartbeat lease expiration.
- a process-level Kafka request → durable Router → gRPC Engine → fenced PostgreSQL completion →
  outbox relay → Kafka result path.

```bash
docker compose up -d
cmake --preset default
cmake --build --preset default
ctest --preset default --output-on-failure
```

P5 startup recovery is scoped to the demo topic's configured partitions. P5 does not claim
multi-Router rebalance safety. Partition revoke/drain/reconstruction across a
consumer group and the full distributed Kafka-to-gRPC latency suite belong to P6 and P7.
