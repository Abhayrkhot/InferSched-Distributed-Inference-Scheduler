#pragma once

#include <chrono>

namespace infersched::engine {

class FakeClock {
 public:
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<FakeClock, duration>;
  static constexpr bool is_steady = true;

  [[nodiscard]] time_point Now() const noexcept { return now_; }

  void Advance(const duration delta) noexcept {
    if (delta > duration::zero()) {
      now_ += delta;
    }
  }

 private:
  time_point now_{};
};

}  // namespace infersched::engine
