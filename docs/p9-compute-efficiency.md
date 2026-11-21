# P9 compute efficiency: does the scheduling use the hardware better?

Throughput and latency (P7, P8) do not answer whether the scheduler gets more useful work out of
the same compute. A system can raise QPS by burning more accelerator time. This phase adds
compute accounting to the Engine and four experiments. Each experiment changes one scheduling
decision and measures how much modeled accelerator time is spent per token delivered to a client.

## What "compute" means here

Engines do not run a neural network. They execute a phenomenological cost model:

| Cost model | Prefill | Decode step |
|---|---|---|
| `p7` (historical) | 100 µs + 10 µs/prompt token | 50 µs + 10 µs/active sequence |
| `memory-bound` | 100 µs + 10 µs/prompt token | 500 µs + 10 µs/active sequence |
| fleet experiments | 100 µs + 10 µs/prompt token | 300 µs + 20 µs/active sequence |

The decode step's fixed term stands in for work paid once per step regardless of batch size, such
as streaming weights on a bandwidth-bound accelerator. Batching helps only by amortizing that term.
The size of every batching effect below therefore depends on this split, which is why two shapes
are reported. In the fleet experiments, each Engine sleeps its modeled time in wall-clock time
(`--time-scale`). A time scale of 3 models hardware three times slower, and *hardware time* is
modeled time × time scale.

## Metrics (Engine counters, `ContinuousEngineStats` / `EngineService::UtilizationJson`)

| Metric | Definition |
|---|---|
| Mean batch | decode tokens computed ÷ decode steps: how full each step was |
| Decode µs / useful token | modeled decode time ÷ output tokens delivered |
| Useful tokens / busy s | output tokens a live client accepted ÷ modeled busy time (× time scale for hardware seconds) |
| Hardware busy | Σ over Engines of modeled busy time × time scale |
| Wasted tokens | decode tokens discarded by preemption or cancellation, plus tokens completed for attempts nobody accepted |
| Prefill computed / reused | uncached prompt tokens prefilled vs served from the prefix cache |
| Busy % | hardware busy ÷ wall-clock window from first accept to last completion |

`TokenAccountingBalancesUnderPreemption` checks the invariant *decode tokens computed = useful +
wasted*.

## Experiment 1: request-level vs iteration-level batching

`scripts/exp_batching_policy.sh` runs the same `ContinuousEngine`, the same 10,000 Poisson
arrivals, and the same cost model, changing only the admission policy:

- `static`: admit only once the running batch has fully drained (request-level batching, no
  batching window);
- `continuous`: admit into free slots at every iteration boundary.

It uses the deterministic fake clock, so the results reproduce exactly.
`Experiments.ContinuousBatchingOutperformsStaticOnMixedWorkload` (CTest, hermetic) freezes the
headline row. The *mixed* workload is 80% short replies (1–16 tokens) and 20% long ones (64–512
tokens). The *short* workload is the historical P7 one (1–16 tokens).

**Mixed workload, memory-bound model**

| Offered QPS | Policy | Achieved QPS | Mean batch | Decode µs / useful token | Useful tokens / busy s | E2E p50 | E2E p99 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | static | 50.1 | 1.78 | 290.7 | 3,296 | 132.3 ms | 416.1 ms |
| 50 | continuous | 50.1 | 1.96 | 264.9 | 3,602 | 6.8 ms | 268.2 ms |
| 200 | static | 188.1 | 8.36 | 69.8 | 12,117 | 1,778.8 ms | 3,156.5 ms |
| 200 | continuous | 199.7 | 9.09 | 65.0 | 12,866 | 8.8 ms | 352.7 ms |
| 300 | static | 188.4 | 8.38 | 69.7 | 12,136 | 10,078.6 ms | 19,530.8 ms |
| 300 | continuous | 298.5 | 17.07 | 39.3 | 19,231 | 10.8 ms | 444.0 ms |
| 400 | static | 188.6 | 8.39 | 69.6 | 12,151 | 14,101.2 ms | 27,660.8 ms |
| 400 | continuous | 396.4 | 30.39 | 26.5 | 25,537 | 14.2 ms | 603.0 ms |

Static batching saturates at about 188 QPS, and continuous batching sustains 400 QPS on the same
Engine (2.1×). The cause is stragglers. A static batch starts full, but four in five of its
sequences finish within 16 steps, and the remaining hundreds of steps decode a handful of long
replies while the freed slots sit empty. Mean occupancy stays near 8.4 however large the queue
grows. At 300 QPS, continuous batching spends 44% less decode compute per delivered token and
delivers 58% more useful tokens per busy second.

**Mixed workload, p7 model**

| Offered QPS | Policy | Busy % | Mean batch | Decode µs / useful token | Useful tokens / busy s | E2E p50 | E2E p99 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | static | 80.5% | 1.26 | 49.7 | 16,035 | 13.9 ms | 51.3 ms |
| 200 | continuous | 72.5% | 1.49 | 43.5 | 17,792 | 1.9 ms | 43.3 ms |
| 500 | static | 100.0% | 5.98 | 18.4 | 32,188 | 58.0 ms | 154.9 ms |
| 500 | continuous | 99.6% | 6.10 | 18.2 | 32,359 | 4.0 ms | 130.0 ms |
| 600 | static | 100.0% (540.6 achieved) | 8.33 | 16.0 | 34,838 | 1,006.6 ms | 1,861.7 ms |
| 600 | continuous | 100.0% (599.5 achieved) | 15.73 | 13.2 | 38,634 | 7.5 ms | 305.0 ms |

Here the same work costs 10% less compute at 200 QPS (72.5% vs 80.5% busy). Capacity rises from
about 540 to about 600–650 QPS.

**Short workload (outputs 1–16 tokens).** The gain is small, because a static batch waits at most
16 steps for its longest member. With the p7 model, continuous batching takes 6–45% fewer decode
steps but only 5–10% less decode compute per token, and both policies saturate near 1,090 QPS. With
the memory-bound model, continuous batching sustains 1,000 QPS against static's ~970, at 15% less
decode compute per token. Full tables: `scripts/exp_batching_policy.sh`.

**A cost of continuous admission.** With the p7 model and short workload, continuous batching's p99
is *worse* at every load from 250 to 1,000 QPS (for example 116 ms vs 72 ms at 1,000 QPS). Every new request's
prefill is charged into the running batch's step, so sequences already decoding stall behind each
admission. Chunked prefill is the standard remedy and is not implemented.

## Experiment 2: routing on a heterogeneous fleet

`scripts/exp_fleet_efficiency.py routing`: 4 Engines on real processes, `engine-4` three times
slower, 4,000 open-loop requests at 400 QPS through the HTTP gateway, Redis hints off.

| Routing | e2e p50 | e2e p99 | Hardware busy | Useful tokens / hardware s | Mean batch | Per-Engine busy | Dispatch split |
|---|---:|---:|---:|---:|---:|---|---|
| first | 53.3 ms | 103.9 ms | 8.49 s | 8,480 | 5.73 | 74% / 0% / 0% / 0% | 4005 / 0 / 0 / 0 |
| round-robin | 31.9 ms | 87.8 ms | 26.05 s | 2,764 | 1.43 | 48% / 51% / 51% / 86% | 1002 / 1001 / 1001 / 1001 |
| least-loaded | 31.7 ms | 68.5 ms | 23.40 s | 3,077 | 1.55 | 49% / 50% / 47% / 65% | 1142 / 1145 / 1136 / 582 |

Load-aware routing beats round-robin on every measure. It sends the slow Engine 42% fewer requests,
which keeps its batches from piling up, uses 10% less hardware time, delivers 11% more tokens per
hardware second, and cuts p99 by 22%.

The honest caveat is the `first` row. Sending everything to one Engine uses about 64% less hardware
time than spreading it, because one well-filled batch amortizes the per-step cost far better than
four small ones (mean batch 5.7 vs 1.6). Spreading load is a latency and headroom decision, not a
compute-saving one. At this load one Engine is 74% busy and its p50 is 53 ms against 32 ms. A
policy that consolidates onto as few Engines as an SLO allows (and lets the rest idle or scale
down) would beat both, and it is not implemented.

## Experiment 3: Redis prefix affinity

`scripts/exp_fleet_efficiency.py affinity`: 4 identical Engines, 3,000 requests at 200 QPS. Every
request shares one of 64 512-token prefixes. Each Engine's prefix cache holds about 16 prefixes
(1,024 nodes), so the working set fits only when each Engine sees a quarter of them.

| Hints | e2e p50 | e2e p99 | Hardware busy | Useful tokens / hardware s | Prefill tokens computed | Prefill tokens reused | Hint dispatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| off (least-loaded) | 27.3 ms | 51.4 ms | 24.55 s | 2,234 | 1,000,009 | 658,816 | 0 |
| on | 21.5 ms | 39.2 ms | 16.50 s | 3,324 | 158,185 | 1,500,640 | 2,935 |

Hints cut prefill computation by 84% and fleet hardware time by 33%, raise useful tokens per
hardware second by 49%, and lower p99 by 24%. Without hints, every Engine sees all 64 prefixes and
its cache thrashes. With hints, each prefix tends to stay on the Engine that already holds it.

## Experiment 4: cancellation of abandoned attempts

`scripts/exp_fleet_efficiency.py cancellation`: 3 identical Engines, 3,000 requests at 150 QPS.
10% of requests need 800 output tokens, which cannot finish within the Router's 60 ms dispatch
deadline. Each such request gets two attempts and then a fenced `FAILED_TERMINAL`. The variants
differ only in whether Engines honour cancellation (`--cancellation off` keeps computing abandoned
attempts to completion).

| Cancellation | Completed | e2e p50 | e2e p99 | Hardware busy | Useful tokens / hardware s | Mean batch | Per-Engine busy | Wasted tokens |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| off | 2,703 | 35.9 ms | 83.7 ms | 48.06 s | 1,009 | 4.50 | 78% / 76% / 76% | 90.7% |
| on | 2,703 | 32.6 ms | 68.8 ms | 31.74 s | 1,527 | 1.45 | 51% / 49% / 50% | 62.3% |

Both variants deliver the same 2,703 requests and fail the same 297 at the deadline. With
cancellation on, the fleet spends 34% less hardware time to do it, the Engines run half empty
instead of filling their batches with orphaned work, and useful tokens per hardware second rise
51%. Waste is still 62%, because each doomed attempt runs for the full 60 ms deadline before it is
cancelled. Rejecting work that cannot meet its deadline at admission would remove most of the rest.

## Repeatability

The fleet experiments ran three times on a host shared with other workloads. The tables above are
from each experiment's first run, at a 1-minute load average of roughly 4–13 on 20 CPUs. The second
and third runs saw 13 and up to 29. The compute accounting is
stable across runs. Latency is not: under heavy host contention, the trial-2 routing p99 reached
13 s.

| Comparison | Hardware time | Useful tokens / hardware s | Other |
|---|---:|---:|---|
| least-loaded vs round-robin | −10% to −19% | +11% to +24% | slow-Engine share 582–722 vs 1,001 |
| hints on vs off | −29% to −33% | +40% to +49% | prefill computed −84% in all three runs |
| cancellation on vs off | −34% to −42% | +51% to +73% | wasted share 91% → 59–62% |
| first vs least-loaded | −60% to −64% | +150% to +176% | at higher latency, one Engine hot |

Experiment 1 is deterministic and needs no repeat.

## What this shows and what it does not

It shows that, against a modeled accelerator, four scheduling decisions change how much compute is
spent per delivered token:
- iteration-level batching keeps batches full;
- prefix affinity avoids recomputing prefill;
- cancellation stops abandoned work;
- load-aware routing keeps slow hardware from accumulating queues.

It does not show GPU-level utilization, such as SM occupancy, memory bandwidth, or kernel
efficiency. The magnitudes depend on the assumed cost-model shape, which the p7 and memory-bound
results bracket. Validating them needs a real model behind the Engine interface.

## Reproduce

```bash
cmake --preset default && cmake --build build/default --parallel
scripts/exp_batching_policy.sh                  # deterministic, ~1 s, no services
docker compose up -d
scripts/exp_fleet_efficiency.py --json build/fleet.json   # ~5 min, all fleet experiments
scripts/exp_fleet_efficiency.py affinity         # one experiment
```
