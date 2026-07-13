#include "infersched/metrics/histogram.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace infersched::metrics {

void Histogram::Record(const std::uint64_t value_microseconds) noexcept {
  const std::size_t bucket =
      value_microseconds == 0
          ? 0
          : std::min<std::size_t>(
                static_cast<std::size_t>(std::bit_width(value_microseconds)),
                                  kBucketCount - 1);
  buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Histogram::Count() const noexcept {
  std::uint64_t total = 0;
  for (const auto& bucket : buckets_) {
    total += bucket.load(std::memory_order_relaxed);
  }
  return total;
}

std::uint64_t Histogram::Percentile(const double percentile) const noexcept {
  const std::uint64_t total = Count();
  if (total == 0) {
    return 0;
  }
  const double clamped = std::clamp(percentile, 0.0, 100.0);
  const auto target = static_cast<std::uint64_t>(
      std::ceil((clamped / 100.0) * static_cast<double>(total)));
  std::uint64_t cumulative = 0;
  for (std::size_t index = 0; index < kBucketCount; ++index) {
    cumulative += buckets_[index].load(std::memory_order_relaxed);
    if (cumulative >= std::max<std::uint64_t>(target, 1)) {
      if (index == kBucketCount - 1) {
        return std::numeric_limits<std::uint64_t>::max();
      }
      return std::uint64_t{1} << index;
    }
  }
  return std::numeric_limits<std::uint64_t>::max();
}

}  // namespace infersched::metrics
