#include "infersched/engine/deterministic_engine.hpp"

#include <algorithm>
#include <limits>
#include <string_view>
#include <utility>

#include "infersched/engine/execution_state.hpp"
#include "infersched/engine/result_cache_key.hpp"

namespace infersched::engine {
namespace {

std::size_t BlocksForTokens(const std::size_t tokens,
                            const std::size_t block_size) {
  return tokens / block_size + static_cast<std::size_t>(tokens % block_size != 0);
}

void AppendKeyPart(std::string& key, const std::string_view part) {
  key += std::to_string(part.size());
  key.push_back(':');
  key.append(part);
  key.push_back('|');
}

}  // namespace

DeterministicEngine::DeterministicEngine(EngineConfig config)
    : config_(config),
      cost_model_(config.cost_model, config.seed),
      kv_cache_(config.kv_blocks, config.kv_block_size_tokens),
      prefix_cache_(config.prefix_cache_entries, kv_cache_),
      result_cache_(config.result_cache_entries),
      policy_(config.max_batch_sequences, config.max_batch_tokens) {}

void DeterministicEngine::Submit(EngineRequest request) {
  pending_.push_back(std::move(request));
}

bool DeterministicEngine::RunOneBatch() {
  const bool rejected_impossible = RejectImpossibleReadyRequests();
  std::vector<PendingRequest> policy_input;
  policy_input.reserve(pending_.size());
  for (const auto& request : pending_) {
    policy_input.push_back(PendingRequest{
        .request_id = request.request_id,
        .prompt_tokens = request.prompt_tokens,
        .max_output_tokens = request.max_output_tokens,
        .priority = request.priority,
        .arrival_time = request.arrival_time,
    });
  }
  const auto selected = policy_.Select(policy_input, clock_.Now());
  if (selected.empty()) {
    return rejected_impossible;
  }

  struct ActiveRequest {
    std::size_t pending_index;
    ExecutionStateMachine state;
    std::size_t remaining_tokens;
    bool prefix_cache_hit;
  };

  std::vector<ActiveRequest> active;
  std::vector<std::size_t> completed_indices;
  std::size_t uncached_prompt_tokens = 0;

  for (const std::size_t index : selected) {
    const auto& request = pending_[index];
    const std::string result_key = ResultKey(request);
    if (const auto cached = result_cache_.Get(result_key); cached.has_value()) {
      results_.push_back(EngineResult{.request_id = request.request_id,
                                      .output = *cached,
                                      .completed_at = clock_.Now(),
                                      .result_cache_hit = true,
                                      .prefix_cache_hit = false});
      completed_indices.push_back(index);
      continue;
    }

    if (request.prompt_tokens >
        std::numeric_limits<std::size_t>::max() - request.max_output_tokens) {
      continue;
    }
    const std::size_t prompt_blocks =
        BlocksForTokens(request.prompt_tokens, config_.kv_block_size_tokens);
    const std::size_t required_blocks = BlocksForTokens(
        request.prompt_tokens + request.max_output_tokens,
        config_.kv_block_size_tokens);
    auto shared = prefix_cache_.Get(PrefixKey(request));
    if (shared.has_value() && shared->size() > prompt_blocks) {
      shared.reset();
    }
    const std::span<const PagedKvCache::BlockId> shared_blocks =
        shared.has_value() ? std::span<const PagedKvCache::BlockId>{*shared}
                           : std::span<const PagedKvCache::BlockId>{};
    if (kv_cache_.Allocate(request.request_id, required_blocks, shared_blocks) !=
        PagedKvCache::AllocationStatus::kAllocated) {
      continue;
    }

    ExecutionStateMachine state;
    static_cast<void>(state.TransitionTo(ExecutionState::kPrefill));
    const std::size_t cached_tokens =
        shared_blocks.size() >= prompt_blocks
            ? request.prompt_tokens
            : shared_blocks.size() * config_.kv_block_size_tokens;
    uncached_prompt_tokens += request.prompt_tokens - cached_tokens;
    active.push_back(ActiveRequest{.pending_index = index,
                                   .state = state,
                                   .remaining_tokens = request.max_output_tokens,
                                   .prefix_cache_hit = !shared_blocks.empty()});
  }

  if (!active.empty()) {
    clock_.Advance(cost_model_.Prefill(uncached_prompt_tokens));
  }
  for (auto& item : active) {
    const auto& request = pending_[item.pending_index];
    static_cast<void>(item.state.TransitionTo(ExecutionState::kDecode));
    if (!item.prefix_cache_hit && request.prompt_tokens > 0) {
      const std::size_t prompt_blocks =
          BlocksForTokens(request.prompt_tokens, config_.kv_block_size_tokens);
      const auto all_blocks = kv_cache_.BlocksFor(request.request_id);
      const auto prefix_blocks = all_blocks.first(std::min(prompt_blocks, all_blocks.size()));
      static_cast<void>(prefix_cache_.Put(PrefixKey(request), prefix_blocks));
    }
  }

  while (std::ranges::any_of(active, [](const auto& item) {
    return item.remaining_tokens > 0;
  })) {
    const std::size_t active_sequences = static_cast<std::size_t>(std::ranges::count_if(
        active, [](const auto& item) { return item.remaining_tokens > 0; }));
    clock_.Advance(cost_model_.DecodeStep(active_sequences));
    for (auto& item : active) {
      if (item.remaining_tokens > 0) {
        --item.remaining_tokens;
      }
    }
  }

  for (auto& item : active) {
    const auto& request = pending_[item.pending_index];
    static_cast<void>(item.state.TransitionTo(ExecutionState::kCompleted));
    const std::string output = SimulatedOutput(request);
    result_cache_.Put(ResultKey(request), output);
    static_cast<void>(kv_cache_.Free(request.request_id));
    results_.push_back(EngineResult{.request_id = request.request_id,
                                    .output = output,
                                    .completed_at = clock_.Now(),
                                    .result_cache_hit = false,
                                    .prefix_cache_hit = item.prefix_cache_hit});
    completed_indices.push_back(item.pending_index);
  }

  std::ranges::sort(completed_indices, std::greater<>{});
  for (const std::size_t index : completed_indices) {
    pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(index));
  }
  return rejected_impossible || !completed_indices.empty();
}

void DeterministicEngine::RunUntilIdle() {
  while (!pending_.empty()) {
    if (RunOneBatch()) {
      continue;
    }
    std::optional<FakeClock::time_point> next_future_arrival;
    for (const auto& request : pending_) {
      if (request.arrival_time > clock_.Now() &&
          (!next_future_arrival.has_value() ||
           request.arrival_time < *next_future_arrival)) {
        next_future_arrival = request.arrival_time;
      }
    }
    if (next_future_arrival.has_value()) {
      clock_.Advance(*next_future_arrival - clock_.Now());
      continue;
    }
    break;  // Remaining requests cannot be admitted with the current resources.
  }
}

void DeterministicEngine::ClearCaches() noexcept {
  prefix_cache_.Clear();
}

std::string DeterministicEngine::ResultKey(const EngineRequest& request) const {
  return BuildResultCacheKey(ResultCacheKeyParts{
      .model_revision = request.model_revision,
      .tokenizer_revision = request.tokenizer_revision,
      .prompt_hash = request.prompt_hash,
      .max_output_tokens = request.max_output_tokens,
      .temperature_micros = request.temperature_micros,
      .top_p_micros = request.top_p_micros,
      .seed = request.sampling_seed,
  });
}

std::string DeterministicEngine::PrefixKey(const EngineRequest& request) const {
  std::string key;
  AppendKeyPart(key, request.model_revision);
  AppendKeyPart(key, request.tokenizer_revision);
  AppendKeyPart(key, request.prompt_hash);
  key += std::to_string(request.prompt_tokens);
  return key;
}

std::string DeterministicEngine::SimulatedOutput(
    const EngineRequest& request) const {
  return "simulated:" + request.model_revision + ':' + request.prompt_hash + ':' +
         std::to_string(request.max_output_tokens) + ':' +
         std::to_string(request.sampling_seed);
}

std::optional<std::string_view> DeterministicEngine::ImpossibleReason(
    const EngineRequest& request) const noexcept {
  if (request.prompt_tokens >
      std::numeric_limits<std::size_t>::max() - request.max_output_tokens) {
    return "token_count_overflow";
  }
  const std::size_t total_tokens =
      request.prompt_tokens + request.max_output_tokens;
  if (total_tokens > config_.max_batch_tokens) {
    return "exceeds_max_batch_tokens";
  }
  if (BlocksForTokens(total_tokens, config_.kv_block_size_tokens) >
      kv_cache_.total_blocks()) {
    return "exceeds_kv_capacity";
  }
  return std::nullopt;
}

bool DeterministicEngine::RejectImpossibleReadyRequests() {
  bool rejected = false;
  for (std::size_t index = 0; index < pending_.size();) {
    const auto reason = ImpossibleReason(pending_[index]);
    if (pending_[index].arrival_time > clock_.Now() || !reason.has_value()) {
      ++index;
      continue;
    }

    ExecutionStateMachine state;
    static_cast<void>(state.TransitionTo(ExecutionState::kFailed));
    results_.push_back(EngineResult{
        .request_id = pending_[index].request_id,
        .output = {},
        .completed_at = clock_.Now(),
        .result_cache_hit = false,
        .prefix_cache_hit = false,
        .succeeded = false,
        .failure_reason = std::string(*reason),
    });
    pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(index));
    rejected = true;
  }
  return rejected;
}

}  // namespace infersched::engine
