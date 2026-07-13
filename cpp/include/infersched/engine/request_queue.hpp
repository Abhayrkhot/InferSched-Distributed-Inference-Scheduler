#pragma once

#include <cstddef>
#include <optional>
#include <queue>
#include <vector>

#include "infersched/engine/deterministic_engine.hpp"

namespace infersched::engine {

// Deterministic min-heap for the optimized continuous path. Equal arrival
// times are ordered by request_id, avoiding unordered-container iteration.
class RequestQueue {
 public:
  void Push(EngineRequest request);
  [[nodiscard]] std::optional<EngineRequest> PopReady(
      FakeClock::time_point now);
  [[nodiscard]] std::optional<FakeClock::time_point> NextArrival() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return queue_.size(); }
  [[nodiscard]] bool empty() const noexcept { return queue_.empty(); }

 private:
  struct Later {
    bool operator()(const EngineRequest& lhs,
                    const EngineRequest& rhs) const noexcept;
  };
  std::priority_queue<EngineRequest, std::vector<EngineRequest>, Later> queue_;
};

}  // namespace infersched::engine
