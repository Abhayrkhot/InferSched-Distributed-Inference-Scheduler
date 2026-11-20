#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "infersched/coordination/prefix_hint.hpp"
#include "infersched/distributed/router_service.hpp"

namespace infersched::distributed {

enum class RoutingPolicy {
  kLeastLoaded,  // Load estimate + optional prefix affinity (default).
  kRoundRobin,   // Rotate through live Engines, ignoring load.
  kFirst,        // Always the lowest engine_id: the pre-P8 single-Engine behaviour.
};

[[nodiscard]] std::optional<RoutingPolicy> ParseRoutingPolicy(std::string_view name);

struct SelectionInput {
  const std::vector<EngineRecord>& engines;
  // Attempts this Router has in flight per engine_id (exact for this Router).
  const std::unordered_map<std::string, std::size_t>& local_inflight;
  // Engines this request already failed on; skipped while alternatives exist.
  const std::unordered_set<std::string>& excluded;
  std::optional<coordination::PrefixHint> hint;
  // Prefer the hinted Engine while its load is within this many sequences of
  // the least-loaded Engine; beyond that, load balance wins over cache reuse.
  std::size_t affinity_slack{8};
  // Equal-load ties are broken by hash(engine_id, tie_breaker) — typically
  // the request id — so ties spread across Engines instead of always landing
  // on the lexicographically first one.
  std::string_view tie_breaker{};
  RoutingPolicy policy{RoutingPolicy::kLeastLoaded};
  // Monotonic per-Router counter used by kRoundRobin.
  std::size_t round_robin_ticket{};
};

struct Selection {
  EngineRecord engine;
  bool affinity{};  // Chosen because of a valid prefix-affinity hint.
};

// Load estimate in sequences: the larger of this Router's exact in-flight
// count and the Engine's last heartbeat (running + waiting), which also
// covers other Routers sharing the Engine. Low free KV adds a penalty.
[[nodiscard]] std::size_t EstimateLoad(const EngineRecord& engine,
                                       std::size_t local_inflight);

// kLeastLoaded: valid affinity hint within slack, otherwise the least loaded
// non-saturated Engine (ties by hash). kRoundRobin / kFirst ignore load and
// hints but still skip excluded Engines. nullopt when every live Engine is
// excluded or there are none.
[[nodiscard]] std::optional<Selection> SelectEngine(const SelectionInput& input);

}  // namespace infersched::distributed
