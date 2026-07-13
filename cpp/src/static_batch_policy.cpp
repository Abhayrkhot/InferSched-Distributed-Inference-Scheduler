#include "infersched/engine/static_batch_policy.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace infersched::engine {

StaticBatchPolicy::StaticBatchPolicy(const std::size_t max_batch_sequences,
                                     const std::size_t max_batch_tokens)
    : max_batch_sequences_(max_batch_sequences),
      max_batch_tokens_(max_batch_tokens) {
  if (max_batch_sequences == 0 || max_batch_tokens == 0) {
    throw std::invalid_argument("static batch limits must be positive");
  }
}

std::vector<std::size_t> StaticBatchPolicy::Select(
    const std::vector<PendingRequest>& pending,
    const FakeClock::time_point now) const {
  std::vector<std::size_t> ordered(pending.size());
  std::iota(ordered.begin(), ordered.end(), 0);
  std::stable_sort(ordered.begin(), ordered.end(), [&](const auto lhs, const auto rhs) {
    const auto lhs_arrival = std::min(pending[lhs].arrival_time, now);
    const auto rhs_arrival = std::min(pending[rhs].arrival_time, now);
    if (lhs_arrival != rhs_arrival) {
      return lhs_arrival < rhs_arrival;
    }
    return pending[lhs].request_id < pending[rhs].request_id;
  });

  std::vector<std::size_t> selected;
  selected.reserve(std::min(max_batch_sequences_, pending.size()));
  std::size_t token_count = 0;
  for (const std::size_t index : ordered) {
    if (pending[index].arrival_time > now) {
      continue;
    }
    if (pending[index].max_output_tokens > max_batch_tokens_ ||
        pending[index].prompt_tokens >
            max_batch_tokens_ - pending[index].max_output_tokens) {
      continue;
    }
    const std::size_t request_tokens =
        pending[index].prompt_tokens + pending[index].max_output_tokens;
    if (selected.size() == max_batch_sequences_) {
      break;
    }
    if (request_tokens > max_batch_tokens_ - token_count) {
      continue;
    }
    selected.push_back(index);
    token_count += request_tokens;
  }
  return selected;
}

}  // namespace infersched::engine
