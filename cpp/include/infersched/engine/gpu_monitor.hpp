#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace infersched::engine {

// Samples NVIDIA GPU 0 through NVML (loaded with dlopen, so CPU-only builds and
// machines without a driver simply report available = false).
//
// NVML "utilization.gpu" is the fraction of the sample period during which at
// least one kernel was running. It says the GPU was busy, not how much of its
// compute or bandwidth the kernels used; the benchmark reports it alongside
// tokens/s and per-step timings rather than as an efficiency figure.
class GpuMonitor {
 public:
  struct Summary {
    bool available{};
    std::string name;
    std::size_t samples{};
    double mean_utilization_pct{};
    double mean_memory_utilization_pct{};  // Memory-controller busy fraction.
    double mean_power_w{};
    std::uint64_t peak_used_bytes{};
    std::uint64_t total_bytes{};
  };

  explicit GpuMonitor(std::chrono::milliseconds period = std::chrono::milliseconds(100));
  ~GpuMonitor();
  GpuMonitor(const GpuMonitor&) = delete;
  GpuMonitor& operator=(const GpuMonitor&) = delete;

  [[nodiscard]] bool available() const noexcept { return device_ != nullptr; }
  using TimePoint = std::chrono::steady_clock::time_point;
  void Start();                 // Clears old samples and begins sampling.
  [[nodiscard]] Summary Stop();  // Stops sampling and summarizes every sample.
  // Summary of the samples taken in [from, to] (sampling may continue).
  [[nodiscard]] Summary Summarize(TimePoint from, TimePoint to) const;

 private:
  void Loop();
  void TakeSample();

  std::chrono::milliseconds period_;
  void* library_{};
  void* device_{};
  void* get_utilization_{};
  void* get_memory_{};
  void* get_power_{};
  std::string name_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  struct Sample {
    TimePoint at;
    double utilization{};
    double memory_utilization{};
    double power_w{-1.0};
    std::uint64_t used{};
  };
  mutable std::mutex mutex_;
  std::vector<Sample> samples_;
  std::uint64_t total_{};
};

}  // namespace infersched::engine
