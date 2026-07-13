#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>

namespace infersched::engine {

struct CostModelConfig {
  std::chrono::nanoseconds prefill_fixed{10'000};
  std::chrono::nanoseconds prefill_per_token{1'000};
  std::chrono::nanoseconds decode_fixed{5'000};
  std::chrono::nanoseconds decode_per_sequence{2'000};
  std::uint32_t jitter_basis_points{0};  // Uniform +/-; 100 = 1%.
};

class DeterministicCostModel {
 public:
  DeterministicCostModel(CostModelConfig config, std::uint64_t seed);

  [[nodiscard]] std::chrono::nanoseconds Prefill(
      std::size_t uncached_prompt_tokens);
  [[nodiscard]] std::chrono::nanoseconds DecodeStep(
      std::size_t active_sequences);

 private:
  [[nodiscard]] std::chrono::nanoseconds ApplyJitter(
      std::chrono::nanoseconds base);

  CostModelConfig config_;
  std::mt19937_64 random_;
};

}  // namespace infersched::engine
