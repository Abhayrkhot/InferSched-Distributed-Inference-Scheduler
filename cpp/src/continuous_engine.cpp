#include "infersched/engine/continuous_engine.hpp"

#include <algorithm>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

#include "infersched/engine/result_cache_key.hpp"

namespace infersched::engine {
namespace {

std::size_t BlocksForTokens(const std::size_t tokens,
                            const std::size_t block_size) {
  return tokens / block_size + static_cast<std::size_t>(tokens % block_size != 0);
}

void AppendPart(std::string& output, const std::string_view value) {
  output += std::to_string(value.size());
  output.push_back(':');
  output.append(value);
  output.push_back('|');
}

}  // namespace

ContinuousEngine::ContinuousEngine(EngineConfig config)
    : config_(config),
      cost_model_(config.cost_model, config.seed),
      kv_cache_(config.kv_blocks, config.kv_block_size_tokens),
      prefix_cache_(config.prefix_cache_entries, kv_cache_),
      result_cache_(config.result_cache_entries) {
  if (config.max_batch_sequences == 0 || config.max_batch_tokens == 0 ||
      config.max_prefill_sequences_per_iteration == 0) {
    throw std::invalid_argument("continuous batching limits must be positive");
  }
}

void ContinuousEngine::Submit(EngineRequest request) {
  pending_.Push(std::move(request));
  stats_.peak_pending_requests =
      std::max(stats_.peak_pending_requests, pending_.size());
}

bool ContinuousEngine::RunOneIteration() {
  if (active_.empty()) {
    preempted_requests_.clear();
  }
  bool progressed = AdmitReady();
  CompleteFinished();
  if (active_.empty()) {
    return progressed;
  }
  if (!PrepareDecodeBlocks()) {
    return progressed;
  }

  const auto decode_cost = cost_model_.DecodeStep(active_.size());
  clock_.Advance(decode_cost);
  stats_.simulated_busy_time += decode_cost;
  for (auto& active : active_) {
    ++active.generated_tokens;
    if (active.generated_tokens == 1) {
      active.first_token_at = clock_.Now();
    }
  }
  ++stats_.iterations;
  CompleteFinished();
  return true;
}

void ContinuousEngine::RunUntilIdle() {
  while (!pending_.empty() || !active_.empty()) {
    if (RunOneIteration()) {
      continue;
    }
    const auto next_arrival = pending_.NextArrival();
    if (next_arrival.has_value()) {
      clock_.Advance(*next_arrival - clock_.Now());
      continue;
    }
    break;
  }
}

void ContinuousEngine::ClearCaches() noexcept { prefix_cache_.Clear(); }

bool ContinuousEngine::AdmitReady() {
  bool admitted_any = false;
  std::size_t prefill_budget = 0;
  std::size_t prefill_sequences = 0;
  const std::size_t scan_limit = pending_.size();
  std::vector<EngineRequest> deferred;
  deferred.reserve(scan_limit);
  for (std::size_t scanned = 0;
       scanned < scan_limit && active_.size() < config_.max_batch_sequences;
       ++scanned) {
    auto candidate_value = pending_.PopReady(clock_.Now());
    if (!candidate_value.has_value()) {
      break;
    }
    EngineRequest candidate = std::move(*candidate_value);
    if (const auto reason = ImpossibleReason(candidate); reason.has_value()) {
      results_.push_back(EngineResult{.request_id = candidate.request_id,
                                      .completed_at = clock_.Now(),
                                      .succeeded = false,
                                      .failure_reason = std::string(*reason)});
      admitted_any = true;
      continue;
    }
    if (!active_.empty() && preempted_requests_.contains(candidate.request_id)) {
      deferred.push_back(std::move(candidate));
      continue;
    }
    if (prefill_sequences == config_.max_prefill_sequences_per_iteration) {
      deferred.push_back(std::move(candidate));
      break;
    }
    if (
        active_.size() > config_.max_batch_tokens - prefill_budget ||
        candidate.prompt_tokens >
            config_.max_batch_tokens - prefill_budget - active_.size()) {
      deferred.push_back(std::move(candidate));
      continue;
    }

    const std::string result_key = ResultKey(candidate);
    ++stats_.result_cache_requests;
    if (const auto cached = result_cache_.Get(result_key); cached.has_value()) {
      ++stats_.result_cache_request_hits;
      results_.push_back(EngineResult{.request_id = candidate.request_id,
                                      .output = *cached,
                                      .completed_at = clock_.Now(),
                                      .first_token_at = clock_.Now(),
                                      .result_cache_hit = true});
      admitted_any = true;
      continue;
    }

    const std::size_t prompt_blocks =
        BlocksForTokens(candidate.prompt_tokens, config_.kv_block_size_tokens);
    const auto candidate_hashes = std::span<const std::string>{
        candidate.prompt_block_hashes}.first(
        std::min(candidate.prompt_block_hashes.size(), prompt_blocks));
    std::optional<RadixPrefixCache::Match> match;
    if (!candidate_hashes.empty()) {
      match = prefix_cache_.GetLongest(PrefixNamespace(candidate),
                                       candidate_hashes);
    }

    PagedKvCache::AllocationStatus status =
        PagedKvCache::AllocationStatus::kInsufficientBlocks;
    while (true) {
      const std::span<const PagedKvCache::BlockId> shared =
          match.has_value()
              ? std::span<const PagedKvCache::BlockId>{match->blocks}
              : std::span<const PagedKvCache::BlockId>{};
      status = kv_cache_.Allocate(candidate.request_id, prompt_blocks, shared);
      if (status != PagedKvCache::AllocationStatus::kInsufficientBlocks ||
          !prefix_cache_.EvictOldest()) {
        break;
      }
      match = candidate_hashes.empty()
                  ? std::nullopt
                  : prefix_cache_.GetLongest(PrefixNamespace(candidate),
                                             candidate_hashes);
    }
    if (status != PagedKvCache::AllocationStatus::kAllocated) {
      deferred.push_back(std::move(candidate));
      continue;
    }

    if (!candidate_hashes.empty() &&
        prefix_accounted_requests_.insert(candidate.request_id).second) {
      ++stats_.prefix_cache_requests;
      if (match.has_value()) {
        ++stats_.prefix_cache_request_hits;
      }
    }

    ActiveRequest active{.request = std::move(candidate),
                         .generated_tokens = 0,
                         .prefix_cache_hit = match.has_value()};
    static_cast<void>(active.state.TransitionTo(ExecutionState::kPrefill));
    const std::size_t matched_blocks =
        match.has_value() ? match->matched_block_count : 0;
    const std::size_t cached_tokens =
        matched_blocks >= prompt_blocks
            ? active.request.prompt_tokens
            : matched_blocks * config_.kv_block_size_tokens;
    const auto prefill_cost =
        cost_model_.Prefill(active.request.prompt_tokens - cached_tokens);
    clock_.Advance(prefill_cost);
    stats_.simulated_busy_time += prefill_cost;
    static_cast<void>(active.state.TransitionTo(ExecutionState::kDecode));

    if (!active.request.prompt_block_hashes.empty()) {
      const auto blocks = kv_cache_.BlocksFor(active.request.request_id);
      const auto hashes = std::span<const std::string>{
          active.request.prompt_block_hashes}.first(
          std::min(active.request.prompt_block_hashes.size(), prompt_blocks));
      static_cast<void>(prefix_cache_.PutBlockPrefixes(
          PrefixNamespace(active.request), hashes,
          blocks.first(std::min(blocks.size(), prompt_blocks))));
    }
    prefill_budget += active.request.prompt_tokens;
    active_.push_back(std::move(active));
    ++prefill_sequences;
    ++stats_.admissions;
    stats_.peak_active_sequences =
        std::max(stats_.peak_active_sequences, active_.size());
    admitted_any = true;
  }
  for (auto& request : deferred) {
    pending_.Push(std::move(request));
  }
  return admitted_any;
}

bool ContinuousEngine::PrepareDecodeBlocks() {
  std::size_t index = 0;
  while (index < active_.size()) {
    auto& active = active_[index];
    const std::size_t next_total_tokens =
        active.request.prompt_tokens + active.generated_tokens + 1;
    const std::size_t needed =
        BlocksForTokens(next_total_tokens, config_.kv_block_size_tokens);
    const std::size_t current =
        kv_cache_.BlocksFor(active.request.request_id).size();
    if (needed <= current) {
      ++index;
      continue;
    }

    auto status = kv_cache_.AppendBlocks(active.request.request_id, needed - current);
    while (status == PagedKvCache::AllocationStatus::kInsufficientBlocks &&
           prefix_cache_.EvictOldest()) {
      status = kv_cache_.AppendBlocks(active.request.request_id, needed - current);
    }
    if (status == PagedKvCache::AllocationStatus::kAllocated) {
      ++index;
      continue;
    }

    // active_ is non-empty here, so the documented victim policy always
    // yields a sequence (possibly the requester itself).
    Preempt(SelectVictim());
    index = 0;  // victim removal changes indices; revalidate every sequence.
  }
  return !active_.empty();
}

void ContinuousEngine::CompleteFinished() {
  for (std::size_t index = 0; index < active_.size();) {
    auto& active = active_[index];
    if (active.generated_tokens < active.request.max_output_tokens) {
      ++index;
      continue;
    }
    static_cast<void>(active.state.TransitionTo(ExecutionState::kCompleted));
    const std::string output = SimulatedOutput(active.request);
    result_cache_.Put(ResultKey(active.request), output);
    static_cast<void>(kv_cache_.Free(active.request.request_id));
    results_.push_back(EngineResult{.request_id = active.request.request_id,
                                    .output = output,
                                    .completed_at = clock_.Now(),
                                    .first_token_at =
                                        active.generated_tokens == 0
                                            ? clock_.Now()
                                            : active.first_token_at,
                                    .prefix_cache_hit = active.prefix_cache_hit});
    active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(index));
  }
}

void ContinuousEngine::Preempt(const std::size_t active_index) {
  ActiveRequest victim = std::move(active_[active_index]);
  static_cast<void>(victim.state.TransitionTo(ExecutionState::kPreempted));
  static_cast<void>(victim.state.TransitionTo(ExecutionState::kWaitingAdmission));
  static_cast<void>(kv_cache_.Free(victim.request.request_id));
  victim.request.arrival_time = clock_.Now();
  preempted_requests_.insert(victim.request.request_id);
  pending_.Push(std::move(victim.request));
  active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(active_index));
  ++stats_.preemptions;
}

std::size_t ContinuousEngine::SelectVictim() const {
  std::size_t victim = 0;
  for (std::size_t index = 1; index < active_.size(); ++index) {
    const auto& candidate = active_[index];
    const auto& selected = active_[victim];
    const std::size_t candidate_remaining =
        candidate.request.max_output_tokens - candidate.generated_tokens;
    const std::size_t selected_remaining =
        selected.request.max_output_tokens - selected.generated_tokens;
    if (candidate.request.priority < selected.request.priority ||
        (candidate.request.priority == selected.request.priority &&
         candidate_remaining > selected_remaining)) {
      victim = index;
    }
  }
  return victim;
}

std::optional<std::string_view> ContinuousEngine::ImpossibleReason(
    const EngineRequest& request) const noexcept {
  if (request.prompt_tokens >
      std::numeric_limits<std::size_t>::max() - request.max_output_tokens) {
    return "token_count_overflow";
  }
  if (request.prompt_tokens > config_.max_batch_tokens) {
    return "exceeds_max_batch_tokens";
  }
  if (BlocksForTokens(request.prompt_tokens + request.max_output_tokens,
                      config_.kv_block_size_tokens) > kv_cache_.total_blocks()) {
    return "exceeds_kv_capacity";
  }
  return std::nullopt;
}

std::string ContinuousEngine::ResultKey(const EngineRequest& request) const {
  return BuildResultCacheKey(ResultCacheKeyParts{
      .model_revision = request.model_revision,
      .tokenizer_revision = request.tokenizer_revision,
      .prompt_hash = request.prompt_hash,
      .max_output_tokens = request.max_output_tokens,
      .temperature_micros = request.temperature_micros,
      .top_p_micros = request.top_p_micros,
      .seed = request.sampling_seed});
}

std::string ContinuousEngine::PrefixNamespace(
    const EngineRequest& request) const {
  std::string output;
  AppendPart(output, request.model_revision);
  AppendPart(output, request.tokenizer_revision);
  return output;
}

std::string ContinuousEngine::SimulatedOutput(
    const EngineRequest& request) const {
  return "simulated:" + request.model_revision + ':' + request.prompt_hash + ':' +
         std::to_string(request.max_output_tokens) + ':' +
         std::to_string(request.sampling_seed);
}

}  // namespace infersched::engine
