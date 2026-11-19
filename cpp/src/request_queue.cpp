#include "infersched/engine/request_queue.hpp"

#include <utility>

namespace infersched::engine {

void RequestQueue::Push(EngineRequest request) {
  queue_.push(std::move(request));
}

std::optional<EngineRequest> RequestQueue::PopReady(
    const FakeClock::time_point now) {
  if (queue_.empty() || queue_.top().arrival_time > now) {
    return std::nullopt;
  }
  EngineRequest request = queue_.top();
  queue_.pop();
  return request;
}

std::optional<FakeClock::time_point> RequestQueue::NextArrival() const noexcept {
  return queue_.empty() ? std::nullopt
                        : std::optional<FakeClock::time_point>{
                              queue_.top().arrival_time};
}

std::optional<EngineRequest> RequestQueue::Remove(
    const std::string_view request_id) {
  std::vector<EngineRequest> kept;
  kept.reserve(queue_.size());
  std::optional<EngineRequest> removed;
  while (!queue_.empty()) {
    EngineRequest request = queue_.top();
    queue_.pop();
    if (!removed.has_value() && request.request_id == request_id) {
      removed = std::move(request);
    } else {
      kept.push_back(std::move(request));
    }
  }
  for (auto& request : kept) {
    queue_.push(std::move(request));
  }
  return removed;
}

bool RequestQueue::Later::operator()(const EngineRequest& lhs,
                                     const EngineRequest& rhs) const noexcept {
  if (lhs.arrival_time != rhs.arrival_time) {
    return lhs.arrival_time > rhs.arrival_time;
  }
  return lhs.request_id > rhs.request_id;
}

}  // namespace infersched::engine
