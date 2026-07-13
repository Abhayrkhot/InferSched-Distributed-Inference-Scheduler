# P6 multi-Router rebalancing evidence

P6 runs multiple durable Routers in one Kafka consumer group. Kafka partitions are the sharding
unit; PostgreSQL partition epochs are the fencing authority. Kafka generation numbers are never used
to authorize dispatch or completion.

## Assignment protocol

`KafkaPoller` owns librdkafka and rebalance callbacks on its dedicated polling thread. It reports
assignment and revocation events to the Router processing thread.

On assignment, the Router:

1. atomically increments the assigned partition's PostgreSQL ownership epoch;
2. records the epoch locally only while Kafka reports the partition as owned;
3. queries unresolved requests for its assigned partitions;
4. assigns each recovered request a new attempt and redispatches it with the new fence.

On revocation, the poller:

1. removes the partition from its owned set immediately;
2. drops queued but undispatched messages for that partition;
3. discards its local contiguous-offset tracker;
4. reports revocation so the Router drops its cached epoch;
5. unassigns the partition in librdkafka.

An RPC already in progress may finish after revocation, but its PostgreSQL completion succeeds only
if no new owner has fenced it. Once the new Router increments the epoch and assigns a new attempt,
the old completion is rejected. This allows bounded drain without treating a TTL or Kafka generation
as mutual exclusion.

## Live chaos evidence

`KafkaDurability.ConsumerGroupRebalancesPartitionsAfterRouterStops` starts two real Kafka consumers,
requires disjoint initial partition sets, stops one, and requires the survivor to own all 12 topic
partitions.

`Distributed.MultiRouterKillAndRebalance` starts two Router processes, two Engine processes, and a
24-request Kafka workload. It kills one Router and its Engine during execution, flushes Redis, and
requires the surviving Router to finish every request after reassignment. The test then queries live
PostgreSQL and requires exactly 24 completed requests and 24 distinct authoritative result outbox
keys.

```bash
docker compose up -d
ctest --test-dir build/default \
  -R 'ConsumerGroupRebalances|MultiRouterKillAndRebalance' --output-on-failure
```

## Guarantee boundary

P6 proves consumer-group partition sharding, epoch fencing, reconstruction after Router/Engine loss,
and Redis-loss tolerance. It does not implement consensus or claim uninterrupted availability during
Kafka/PostgreSQL outages. The demo Router still dispatches synchronously per process; distributed
throughput and partition/network fault injection belong to P7's end-to-end benchmark and chaos
suite.
