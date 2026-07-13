#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace infersched::metrics {

// Lock-free log2 histogram for scheduler latency telemetry. Values are in
// integer microseconds; Percentile returns the bucket upper bound.
class Histogram {
 public:
  static constexpr std::size_t kBucketCount = 64;

  void Record(std::uint64_t value_microseconds) noexcept;
  [[nodiscard]] std::uint64_t Count() const noexcept;
  [[nodiscard]] std::uint64_t Percentile(double percentile) const noexcept;

 private:
  std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
};

}  // namespace infersched::metrics
