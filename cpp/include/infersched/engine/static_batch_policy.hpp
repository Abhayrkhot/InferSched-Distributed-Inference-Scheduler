#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "infersched/engine/fake_clock.hpp"

namespace infersched::engine {

struct PendingRequest {
  std::string request_id;
  std::size_t prompt_tokens{};
  std::size_t max_output_tokens{};
  std::uint32_t priority{};
  FakeClock::time_point arrival_time{};
};

class SchedulingPolicy {
 public:
  virtual ~SchedulingPolicy() = default;
  [[nodiscard]] virtual std::vector<std::size_t> Select(
      const std::vector<PendingRequest>& pending,
      FakeClock::time_point now) const = 0;
};

class StaticBatchPolicy final : public SchedulingPolicy {
 public:
  StaticBatchPolicy(std::size_t max_batch_sequences,
                    std::size_t max_batch_tokens);

  // FCFS baseline. Selection is deterministic for identical input and clock.
  [[nodiscard]] std::vector<std::size_t> Select(
      const std::vector<PendingRequest>& pending,
      FakeClock::time_point now) const override;

 private:
  std::size_t max_batch_sequences_;
  std::size_t max_batch_tokens_;
};

}  // namespace infersched::engine
