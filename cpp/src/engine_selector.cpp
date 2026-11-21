#include "infersched/distributed/engine_selector.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace infersched::distributed {
namespace {

constexpr std::size_t kLowKvPenalty = 64;

bool Saturated(const EngineRecord& engine) {
  return engine.capacity.total_kv_blocks() != 0 &&
         engine.capacity.batch_headroom() == 0;
}

std::size_t LocalInflight(const SelectionInput& input, const std::string& engine_id) {
  const auto found = input.local_inflight.find(engine_id);
  return found == input.local_inflight.end() ? 0 : found->second;
}

}  // namespace

std::size_t EstimateLoad(const EngineRecord& engine, const std::size_t local_inflight) {
  const std::size_t reported = static_cast<std::size_t>(engine.capacity.running_seqs()) +
                               static_cast<std::size_t>(engine.capacity.waiting_seqs());
  std::size_t load = std::max(local_inflight, reported);
  const auto total = engine.capacity.total_kv_blocks();
  if (total != 0 && engine.capacity.free_kv_blocks() * 20U < total) {
    load += kLowKvPenalty;  // Under 5% free KV: admission would preempt.
  }
  return load;
}

std::optional<RoutingPolicy> ParseRoutingPolicy(const std::string_view name) {
  if (name == "least-loaded") {
    return RoutingPolicy::kLeastLoaded;
  }
  if (name == "round-robin") {
    return RoutingPolicy::kRoundRobin;
  }
  if (name == "first") {
    return RoutingPolicy::kFirst;
  }
  return std::nullopt;
}

namespace {

std::optional<Selection> SelectByOrder(const SelectionInput& input) {
  std::vector<const EngineRecord*> candidates;
  for (const auto& engine : input.engines) {
    if (!input.excluded.contains(engine.engine_id)) {
      candidates.push_back(&engine);
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }
  std::ranges::sort(candidates, [](const EngineRecord* lhs, const EngineRecord* rhs) {
    return lhs->engine_id < rhs->engine_id;
  });
  const std::size_t index = input.policy == RoutingPolicy::kRoundRobin
                                ? input.round_robin_ticket % candidates.size()
                                : 0;
  return Selection{.engine = *candidates[index], .affinity = false};
}

}  // namespace

std::optional<Selection> SelectEngine(const SelectionInput& input) {
  if (input.policy != RoutingPolicy::kLeastLoaded) {
    return SelectByOrder(input);
  }
  const EngineRecord* best = nullptr;
  using Key = std::tuple<bool, std::size_t, std::size_t, std::string_view>;
  Key best_key;
  for (const auto& engine : input.engines) {
    if (input.excluded.contains(engine.engine_id)) {
      continue;
    }
    std::string tie(engine.engine_id);
    tie.append(input.tie_breaker);
    Key key{Saturated(engine), EstimateLoad(engine, LocalInflight(input, engine.engine_id)),
            std::hash<std::string>{}(tie), engine.engine_id};
    if (best == nullptr || key < best_key) {
      best = &engine;
      best_key = key;
    }
  }
  if (best == nullptr) {
    return std::nullopt;
  }
  if (input.hint.has_value()) {
    for (const auto& engine : input.engines) {
      if (engine.engine_id != input.hint->engine_id ||
          engine.incarnation_id != input.hint->incarnation_id ||
          input.excluded.contains(engine.engine_id) || Saturated(engine)) {
        continue;
      }
      const std::size_t load =
          EstimateLoad(engine, LocalInflight(input, engine.engine_id));
      if (load <= std::get<1>(best_key) + input.affinity_slack) {
        return Selection{.engine = engine, .affinity = true};
      }
    }
  }
  return Selection{.engine = *best, .affinity = false};
}

}  // namespace infersched::distributed
