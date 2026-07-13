#include "infersched/engine/cost_model.hpp"

#include <limits>
#include <stdexcept>

namespace infersched::engine {
namespace {

std::chrono::nanoseconds SaturatingCost(const std::chrono::nanoseconds fixed,
                                        const std::chrono::nanoseconds per_unit,
                                        const std::size_t units) {
  using Rep = std::chrono::nanoseconds::rep;
  const Rep maximum = std::numeric_limits<Rep>::max();
  if (per_unit.count() > 0 &&
      units > static_cast<std::size_t>((maximum - fixed.count()) /
                                      per_unit.count())) {
    return std::chrono::nanoseconds::max();
  }
  return fixed + per_unit * static_cast<Rep>(units);
}

}  // namespace

DeterministicCostModel::DeterministicCostModel(CostModelConfig config,
                                               const std::uint64_t seed)
    : config_(config), random_(seed) {
  if (config.prefill_fixed.count() < 0 ||
      config.prefill_per_token.count() < 0 || config.decode_fixed.count() < 0 ||
      config.decode_per_sequence.count() < 0 ||
      config.jitter_basis_points > 10'000) {
    throw std::invalid_argument("invalid cost model configuration");
  }
}

std::chrono::nanoseconds DeterministicCostModel::Prefill(
    const std::size_t uncached_prompt_tokens) {
  return ApplyJitter(SaturatingCost(config_.prefill_fixed,
                                    config_.prefill_per_token,
                                    uncached_prompt_tokens));
}

std::chrono::nanoseconds DeterministicCostModel::DecodeStep(
    const std::size_t active_sequences) {
  return ApplyJitter(SaturatingCost(config_.decode_fixed,
                                    config_.decode_per_sequence,
                                    active_sequences));
}

std::chrono::nanoseconds DeterministicCostModel::ApplyJitter(
    const std::chrono::nanoseconds base) {
  if (config_.jitter_basis_points == 0 || base == std::chrono::nanoseconds::max()) {
    return base;
  }
  const auto magnitude = static_cast<std::int64_t>(config_.jitter_basis_points);
  std::uniform_int_distribution<std::int64_t> distribution(-magnitude, magnitude);
  const std::int64_t adjustment = distribution(random_);
  const std::int64_t quotient = base.count() / 10'000;
  const std::int64_t remainder = base.count() % 10'000;
  const std::int64_t magnitude_adjustment = adjustment < 0 ? -adjustment : adjustment;
  const std::int64_t delta = quotient * magnitude_adjustment +
                             (remainder * magnitude_adjustment) / 10'000;
  if (adjustment < 0) {
    return std::chrono::nanoseconds{base.count() - delta};
  }
  if (delta > std::numeric_limits<std::int64_t>::max() - base.count()) {
    return std::chrono::nanoseconds::max();
  }
  return std::chrono::nanoseconds{base.count() + delta};
}

}  // namespace infersched::engine
