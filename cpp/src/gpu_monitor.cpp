#include "infersched/engine/gpu_monitor.hpp"

#include <dlfcn.h>

#include <algorithm>

namespace infersched::engine {
namespace {

// Minimal NVML ABI (stable since NVML v2); avoids a build dependency on nvml.h.
struct NvmlUtilization {
  unsigned int gpu;
  unsigned int memory;
};
struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};
using NvmlDevice = void*;
using InitFn = int (*)();
using ShutdownFn = int (*)();
using HandleFn = int (*)(unsigned int, NvmlDevice*);
using NameFn = int (*)(NvmlDevice, char*, unsigned int);
using UtilizationFn = int (*)(NvmlDevice, NvmlUtilization*);
using MemoryFn = int (*)(NvmlDevice, NvmlMemory*);
using PowerFn = int (*)(NvmlDevice, unsigned int*);

template <typename Fn>
Fn Symbol(void* library, const char* name) {
  return reinterpret_cast<Fn>(dlsym(library, name));
}

}  // namespace

GpuMonitor::GpuMonitor(const std::chrono::milliseconds period) : period_(period) {
  library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr) {
    return;
  }
  const auto init = Symbol<InitFn>(library_, "nvmlInit_v2");
  const auto handle = Symbol<HandleFn>(library_, "nvmlDeviceGetHandleByIndex_v2");
  get_utilization_ = reinterpret_cast<void*>(
      Symbol<UtilizationFn>(library_, "nvmlDeviceGetUtilizationRates"));
  get_memory_ = reinterpret_cast<void*>(Symbol<MemoryFn>(library_, "nvmlDeviceGetMemoryInfo"));
  get_power_ = reinterpret_cast<void*>(Symbol<PowerFn>(library_, "nvmlDeviceGetPowerUsage"));
  NvmlDevice device = nullptr;
  if (init == nullptr || handle == nullptr || get_utilization_ == nullptr ||
      get_memory_ == nullptr || init() != 0 || handle(0, &device) != 0) {
    dlclose(library_);
    library_ = nullptr;
    return;
  }
  device_ = device;
  if (const auto name = Symbol<NameFn>(library_, "nvmlDeviceGetName"); name != nullptr) {
    char buffer[96] = {};
    if (name(device, buffer, sizeof(buffer)) == 0) {
      name_ = buffer;
    }
  }
}

GpuMonitor::~GpuMonitor() {
  if (running_.load()) {
    static_cast<void>(Stop());
  }
  if (library_ != nullptr) {
    if (const auto shutdown = Symbol<ShutdownFn>(library_, "nvmlShutdown"); shutdown != nullptr) {
      shutdown();
    }
    dlclose(library_);
  }
}

void GpuMonitor::Start() {
  if (!available() || running_.load()) {
    return;
  }
  {
    std::lock_guard lock(mutex_);
    samples_.clear();
  }
  running_.store(true);
  thread_ = std::thread(&GpuMonitor::Loop, this);
}

GpuMonitor::Summary GpuMonitor::Stop() {
  if (running_.exchange(false) && thread_.joinable()) {
    thread_.join();
  }
  return Summarize(TimePoint::min(), TimePoint::max());
}

GpuMonitor::Summary GpuMonitor::Summarize(const TimePoint from, const TimePoint to) const {
  Summary summary;
  summary.available = available();
  summary.name = name_;
  std::lock_guard lock(mutex_);
  summary.total_bytes = total_;
  double utilization = 0.0;
  double memory_utilization = 0.0;
  double power = 0.0;
  std::size_t power_samples = 0;
  for (const Sample& sample : samples_) {
    if (sample.at < from || sample.at > to) {
      continue;
    }
    ++summary.samples;
    utilization += sample.utilization;
    memory_utilization += sample.memory_utilization;
    if (sample.power_w >= 0.0) {
      power += sample.power_w;
      ++power_samples;
    }
    summary.peak_used_bytes = std::max(summary.peak_used_bytes, sample.used);
  }
  if (summary.samples > 0) {
    summary.mean_utilization_pct = utilization / static_cast<double>(summary.samples);
    summary.mean_memory_utilization_pct =
        memory_utilization / static_cast<double>(summary.samples);
  }
  if (power_samples > 0) {
    summary.mean_power_w = power / static_cast<double>(power_samples);
  }
  return summary;
}

void GpuMonitor::Loop() {
  while (running_.load()) {
    TakeSample();
    std::this_thread::sleep_for(period_);
  }
}

void GpuMonitor::TakeSample() {
  NvmlUtilization utilization{};
  NvmlMemory memory{};
  unsigned int milliwatts = 0;
  if (reinterpret_cast<UtilizationFn>(get_utilization_)(device_, &utilization) != 0) {
    return;
  }
  const bool memory_ok = reinterpret_cast<MemoryFn>(get_memory_)(device_, &memory) == 0;
  const bool power_ok = get_power_ != nullptr &&
                        reinterpret_cast<PowerFn>(get_power_)(device_, &milliwatts) == 0;
  std::lock_guard lock(mutex_);
  samples_.push_back(Sample{.at = std::chrono::steady_clock::now(),
                            .utilization = static_cast<double>(utilization.gpu),
                            .memory_utilization = static_cast<double>(utilization.memory),
                            .power_w = power_ok ? milliwatts / 1000.0 : -1.0,
                            .used = memory_ok ? memory.used : 0});
  if (memory_ok) {
    total_ = memory.total;
  }
}

}  // namespace infersched::engine
