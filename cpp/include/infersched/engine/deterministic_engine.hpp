#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "infersched/engine/cost_model.hpp"
#include "infersched/engine/fake_clock.hpp"
#include "infersched/engine/paged_kv_cache.hpp"
#include "infersched/engine/prefix_cache.hpp"
#include "infersched/engine/result_cache.hpp"
#include "infersched/engine/static_batch_policy.hpp"

namespace infersched::engine {

struct EngineConfig {
  std::size_t kv_blocks{256};
  std::size_t kv_block_size_tokens{16};
  std::size_t max_batch_sequences{8};
  std::size_t max_batch_tokens{4096};
  std::size_t max_prefill_sequences_per_iteration{1};
  std::size_t result_cache_entries{128};
  std::size_t prefix_cache_entries{128};
  std::uint64_t seed{1};
  CostModelConfig cost_model{};
};

struct EngineRequest {
  std::string request_id;
  std::string model_revision;
  std::string tokenizer_revision;
  std::string prompt_hash;
  std::vector<std::string> prompt_block_hashes;
  std::size_t prompt_tokens{};
  std::size_t max_output_tokens{};
  std::uint32_t priority{};
  std::uint64_t sampling_seed{};
  std::uint64_t temperature_micros{};
  std::uint64_t top_p_micros{1'000'000};
  FakeClock::time_point arrival_time{};
};

struct EngineResult {
  std::string request_id;
  std::string output;
  FakeClock::time_point completed_at{};
  FakeClock::time_point first_token_at{};
  bool result_cache_hit{};
  bool prefix_cache_hit{};
  bool succeeded{true};
  std::string failure_reason;
};

class DeterministicEngine {
 public:
  explicit DeterministicEngine(EngineConfig config);

  void Submit(EngineRequest request);
  [[nodiscard]] bool RunOneBatch();
  void RunUntilIdle();
  void ClearCaches() noexcept;

  [[nodiscard]] const std::vector<EngineResult>& results() const noexcept {
    return results_;
  }
  [[nodiscard]] std::size_t pending_count() const noexcept { return pending_.size(); }
  [[nodiscard]] const FakeClock& clock() const noexcept { return clock_; }
  [[nodiscard]] const PagedKvCache& kv_cache() const noexcept { return kv_cache_; }
  [[nodiscard]] std::size_t result_cache_hits() const noexcept {
    return result_cache_.hits();
  }
  [[nodiscard]] std::size_t prefix_cache_hits() const noexcept {
    return prefix_cache_.hits();
  }

 private:
  [[nodiscard]] std::string ResultKey(const EngineRequest& request) const;
  [[nodiscard]] std::string PrefixKey(const EngineRequest& request) const;
  [[nodiscard]] std::string SimulatedOutput(const EngineRequest& request) const;
  [[nodiscard]] std::optional<std::string_view> ImpossibleReason(
      const EngineRequest& request) const noexcept;
  [[nodiscard]] bool RejectImpossibleReadyRequests();

  EngineConfig config_;
  FakeClock clock_;
  DeterministicCostModel cost_model_;
  PagedKvCache kv_cache_;
  PrefixCache prefix_cache_;
  ResultCache result_cache_;
  StaticBatchPolicy policy_;
  std::vector<EngineRequest> pending_;
  std::vector<EngineResult> results_;
};

}  // namespace infersched::engine
