#pragma once

#include <cstddef>
#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "infersched/engine/cost_model.hpp"
#include "infersched/engine/deterministic_engine.hpp"
#include "infersched/engine/execution_state.hpp"
#include "infersched/engine/fake_clock.hpp"
#include "infersched/engine/paged_kv_cache.hpp"
#include "infersched/engine/radix_prefix_cache.hpp"
#include "infersched/engine/result_cache.hpp"
#include "infersched/engine/request_queue.hpp"

namespace infersched::engine {

struct ContinuousEngineStats {
  std::size_t iterations{};
  std::size_t admissions{};  // Admission events; includes post-preemption re-admission.
  std::size_t preemptions{};
  std::size_t peak_active_sequences{};
  std::size_t prefix_cache_requests{};
  std::size_t prefix_cache_request_hits{};
  std::size_t result_cache_requests{};
  std::size_t result_cache_request_hits{};
  std::size_t peak_pending_requests{};
  std::chrono::nanoseconds simulated_busy_time{};

  // Compute accounting (all in tokens of modeled work).
  std::chrono::nanoseconds simulated_prefill_time{};
  std::chrono::nanoseconds simulated_decode_time{};
  // Sum over decode iterations of the batch size: decode tokens computed.
  std::size_t decode_token_slots{};
  // Output tokens of requests that completed successfully.
  std::size_t useful_output_tokens{};
  // Decode tokens thrown away: preempted sequences restart from scratch, and
  // cancelled sequences produce nothing anyone accepts.
  std::size_t wasted_decode_tokens{};
  std::size_t prefill_tokens_computed{};  // Uncached prompt tokens prefilled.
  std::size_t prefill_tokens_reused{};    // Served from the prefix cache.
  std::size_t wasted_prefill_tokens{};    // Prefill of preempted/cancelled work.
  std::size_t cancellations{};
  std::size_t completed_requests{};

  // Mean sequences per decode step: how full the batch was while decoding.
  [[nodiscard]] double mean_batch_occupancy() const noexcept {
    return iterations == 0 ? 0.0
                           : static_cast<double>(decode_token_slots) /
                                 static_cast<double>(iterations);
  }
  // Useful output tokens per second of modeled compute.
  [[nodiscard]] double useful_tokens_per_busy_second() const noexcept {
    const double busy = std::chrono::duration<double>(simulated_busy_time).count();
    return busy == 0.0 ? 0.0 : static_cast<double>(useful_output_tokens) / busy;
  }
};

class ContinuousEngine {
 public:
  explicit ContinuousEngine(EngineConfig config);

  void Submit(EngineRequest request);
  [[nodiscard]] bool RunOneIteration();
  void RunUntilIdle();
  void ClearCaches() noexcept;
  // Removes a queued or running request, frees its KV blocks, and records a
  // failed result with `reason`. Returns false when the id is not live.
  bool Cancel(std::string_view request_id, std::string_view reason);

  [[nodiscard]] const std::vector<EngineResult>& results() const noexcept {
    return results_;
  }
  // Moves out results recorded since the last call. A long-running Engine
  // drains results through this so the history does not grow without bound.
  [[nodiscard]] std::vector<EngineResult> TakeResults();
  // Request ids that produced their first decoded token since the last call
  // (the TTFT marker the network layer streams as PREFILL_DONE).
  [[nodiscard]] std::vector<std::string> TakeFirstTokens();
  [[nodiscard]] bool idle() const noexcept {
    return pending_.empty() && active_.empty();
  }
  [[nodiscard]] const ContinuousEngineStats& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] const FakeClock& clock() const noexcept { return clock_; }
  [[nodiscard]] const PagedKvCache& kv_cache() const noexcept { return kv_cache_; }
  [[nodiscard]] std::size_t pending_count() const noexcept { return pending_.size(); }
  [[nodiscard]] std::size_t active_count() const noexcept { return active_.size(); }
  [[nodiscard]] std::size_t prefix_cache_hits() const noexcept {
    return stats_.prefix_cache_request_hits;
  }
  [[nodiscard]] std::size_t prefix_cache_misses() const noexcept {
    return stats_.prefix_cache_requests - stats_.prefix_cache_request_hits;
  }

 private:
  struct ActiveRequest {
    EngineRequest request;
    ExecutionStateMachine state;
    std::size_t generated_tokens{};
    bool prefix_cache_hit{};
    FakeClock::time_point first_token_at{};
    std::size_t prefilled_tokens{};  // Uncached prompt tokens this admission computed.
  };

  [[nodiscard]] bool AdmitReady();
  [[nodiscard]] bool PrepareDecodeBlocks();
  void CompleteFinished();
  void Preempt(std::size_t active_index);
  [[nodiscard]] std::size_t SelectVictim() const;
  [[nodiscard]] std::optional<std::string_view> ImpossibleReason(
      const EngineRequest& request) const noexcept;
  [[nodiscard]] std::string ResultKey(const EngineRequest& request) const;
  [[nodiscard]] std::string PrefixNamespace(const EngineRequest& request) const;
  [[nodiscard]] std::string SimulatedOutput(const EngineRequest& request) const;
  void ForgetRequest(const std::string& request_id);

  EngineConfig config_;
  FakeClock clock_;
  DeterministicCostModel cost_model_;
  PagedKvCache kv_cache_;
  RadixPrefixCache prefix_cache_;
  ResultCache result_cache_;
  RequestQueue pending_;
  std::vector<ActiveRequest> active_;
  std::vector<EngineResult> results_;
  std::vector<std::string> first_tokens_;
  ContinuousEngineStats stats_;
  std::unordered_set<std::string> preempted_requests_;
  std::unordered_set<std::string> prefix_accounted_requests_;
};

}  // namespace infersched::engine
