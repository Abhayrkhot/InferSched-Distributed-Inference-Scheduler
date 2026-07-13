# P4 distributed execution evidence

P4 runs the Router and Engine as separate operating-system processes connected over gRPC.

## RPC ownership

- `RouterControl` is hosted by the Router: `Register`, `Heartbeat`.
- `EngineControl` is hosted by each Engine: `Dispatch`, `Cancel`, `Drain`.
- Every dispatch and progress event carries `(partition_id, ownership_epoch, attempt_id)`.
- An Engine rejects stale partition epochs and duplicate/stale attempts before execution.

The first item in a dispatch stream is `ACCEPTED` or `REJECTED`. Accepted work is coalesced by the
Engine worker thread into the existing continuous-batching runtime, followed by progress and terminal
events on the stream.

## Automated process tests

`Distributed.MultiProcessDispatchAndFencing` starts both binaries, sends 32 concurrent requests, and
requires all completions, stale-attempt rejection, stale-epoch rejection, and a zero-in-flight drain.

`Distributed.EngineCrashRetryWithNewFence` starts a delayed Engine, kills it with `SIGKILL` after
dispatch begins, starts a new incarnation, and requires the Router to complete the request using a
higher ownership epoch and attempt before draining the replacement.

`Distributed.SameEngineOverlappingRetryRejected` sends a higher attempt for the same request while
the original is still executing on one Engine. The overlap is rejected without advancing the fence,
the original completes, and the Engine drains. This prevents duplicate live sequence IDs; P5 decides
whether the Router reroutes or delays such retries.

```bash
ctest --test-dir build/default -R Distributed --output-on-failure
```

## Guarantee boundary

P4 demonstrates real process/network separation, streamed RPC behavior, transport failure detection,
and fence propagation. It does not yet claim an authoritative effectively-once result: accepted
completion is still in memory. P5 adds PostgreSQL conditional finalization, the transactional outbox,
Kafka offsets, and Redis-backed ephemeral lease hints.

P4 also does not yet implement load- or prefix-aware routing, heartbeat lease expiry, or bounded
retention for in-memory fence/dedup records. The 1 ms worker coalescing window is a correctness-demo
setting, not a latency-tuned production value.

Current P4-safe resume wording: “Built process-separated C++ Engine execution over gRPC with
epoch/attempt fencing and SIGKILL crash recovery.” Routing/load-balancing and authoritative
effectively-once completion are P5 evidence claims, not P4 claims.
