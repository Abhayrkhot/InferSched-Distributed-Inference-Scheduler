#include <atomic>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "infersched/concurrency/mpsc_ring.hpp"
#include "infersched/concurrency/sharded_registry.hpp"
#include "infersched/engine/continuous_engine.hpp"
#include "infersched/engine/deterministic_engine.hpp"
#include "infersched/metrics/histogram.hpp"

namespace {

struct Arguments {
  std::string mode{"scheduler"};
  std::string engine{"continuous"};
  std::string queue{"mpsc"};
  std::string registry{"sharded"};
  std::size_t requests{100'000};
  std::size_t producers{4};
  std::size_t qps{500};
  std::uint64_t seed{7};
  bool telemetry{true};
  std::string cache{"none"};
};

infersched::engine::EngineConfig BenchmarkConfig();
infersched::engine::EngineRequest MakeRequest(std::size_t index);

std::size_t ParseSize(const std::string_view value, const char* name) {
  std::size_t parsed = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || parsed == 0) {
    std::cerr << "invalid " << name << ": " << value << '\n';
    std::exit(EXIT_FAILURE);
  }
  return parsed;
}

Arguments ParseArguments(const int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (index + 1 >= argc) {
      std::cerr << "missing value for " << argument << '\n';
      std::exit(EXIT_FAILURE);
    }
    const std::string_view value{argv[++index]};
    if (argument == "--mode") {
      arguments.mode = value;
    } else if (argument == "--engine") {
      arguments.engine = value;
    } else if (argument == "--queue") {
      arguments.queue = value;
    } else if (argument == "--registry") {
      arguments.registry = value;
    } else if (argument == "--requests") {
      arguments.requests = ParseSize(value, "request count");
    } else if (argument == "--producers") {
      arguments.producers = ParseSize(value, "producer count");
    } else if (argument == "--qps") {
      arguments.qps = ParseSize(value, "QPS");
    } else if (argument == "--seed") {
      arguments.seed = ParseSize(value, "seed");
    } else if (argument == "--telemetry") {
      arguments.telemetry = value == "enabled";
    } else if (argument == "--cache") {
      arguments.cache = value;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  return arguments;
}

void RunOpenLoop(const Arguments& arguments) {
  const auto benchmark_start = std::chrono::steady_clock::now();
  auto config = BenchmarkConfig();
  config.seed = arguments.seed;
  // Phenomenological nonzero model: 100 us fixed + 10 us/prompt token,
  // then 50 us fixed + 10 us/active sequence for every decode step.
  config.cost_model.prefill_fixed = std::chrono::microseconds(100);
  config.cost_model.prefill_per_token = std::chrono::microseconds(10);
  config.cost_model.decode_fixed = std::chrono::microseconds(50);
  config.cost_model.decode_per_sequence = std::chrono::microseconds(10);
  config.cost_model.jitter_basis_points = 200;
  config.kv_blocks = 16'384;
  config.prefix_cache_entries = 1024;
  infersched::engine::ContinuousEngine engine(config);

  std::mt19937_64 random(arguments.seed);
  std::exponential_distribution<double> interarrival(
      static_cast<double>(arguments.qps));
  std::unordered_map<std::string, infersched::engine::FakeClock::time_point>
      arrivals;
  auto arrival = infersched::engine::FakeClock::time_point{};
  for (std::size_t index = 0; index < arguments.requests; ++index) {
    if (index != 0) {
      arrival += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(interarrival(random)));
    }
    auto request = MakeRequest(index);
    if ((arguments.cache == "result" || arguments.cache == "both") &&
        index % 5 == 4) {
      const std::size_t origin = index - 4;
      request.prompt_hash = "prompt-" + std::to_string(origin);
      request.prompt_tokens = 16 + (origin % 113);
      request.max_output_tokens = 1 + (origin % 16);
      request.sampling_seed = origin;
    }
    if (arguments.cache == "prefix" || arguments.cache == "both") {
      const std::string group = std::to_string(index / 5);
      request.prompt_tokens = 64;
      request.prompt_block_hashes = {"shared-a-" + group,
                                     "shared-b-" + group,
                                     "shared-c-" + group,
                                     "unique-" + std::to_string(index)};
    }
    request.arrival_time = arrival;
    request.priority = index % 10 == 0 ? 0U : 1U;
    arrivals.emplace(request.request_id, arrival);
    engine.Submit(std::move(request));
  }
  engine.RunUntilIdle();

  infersched::metrics::Histogram ttft;
  infersched::metrics::Histogram end_to_end;
  infersched::metrics::Histogram low_priority;
  for (const auto& result : engine.results()) {
    const auto submitted = arrivals.at(result.request_id);
    const auto ttft_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             result.first_token_at - submitted)
                             .count();
    const auto e2e_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            result.completed_at - submitted)
                            .count();
    if (arguments.telemetry) {
      ttft.Record(
          static_cast<std::uint64_t>(std::max<std::int64_t>(ttft_us, 0)));
      end_to_end.Record(
          static_cast<std::uint64_t>(std::max<std::int64_t>(e2e_us, 0)));
    }
    if (arguments.telemetry && result.request_id.ends_with("0")) {
      low_priority.Record(
          static_cast<std::uint64_t>(std::max<std::int64_t>(ttft_us, 0)));
    }
  }
  const double makespan_seconds =
      std::chrono::duration<double>(engine.clock().Now().time_since_epoch())
          .count();
  const double busy_seconds =
      std::chrono::duration<double>(engine.stats().simulated_busy_time).count();
  const double utilization =
      makespan_seconds == 0.0 ? 0.0 : busy_seconds / makespan_seconds;
  const auto drain_lag = std::chrono::duration_cast<std::chrono::microseconds>(
      engine.clock().Now() - arrival);
  const bool complete = engine.results().size() == arguments.requests;
  engine.ClearCaches();
  const bool zero_leak = engine.kv_cache().allocated_blocks() == 0;
  const double benchmark_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - benchmark_start)
          .count();
  std::cout << "{\"mode\":\"open_loop\",\"requests\":"
            << arguments.requests << ",\"offered_qps\":" << arguments.qps
            << ",\"telemetry\":" << std::boolalpha << arguments.telemetry
            << ",\"benchmark_wall_ms\":" << benchmark_wall_ms
            << ",\"achieved_qps\":"
            << static_cast<double>(arguments.requests) / makespan_seconds
            << ",\"ttft_p50_us\":" << ttft.Percentile(50)
            << ",\"ttft_p90_us\":" << ttft.Percentile(90)
            << ",\"ttft_p99_us\":" << ttft.Percentile(99)
            << ",\"e2e_p50_us\":" << end_to_end.Percentile(50)
            << ",\"e2e_p90_us\":" << end_to_end.Percentile(90)
            << ",\"e2e_p99_us\":" << end_to_end.Percentile(99)
            << ",\"low_priority_ttft_p99_us\":"
            << low_priority.Percentile(99)
            << ",\"engine_utilization\":" << utilization
            << ",\"scheduled_event_depth\":"
            << engine.stats().peak_pending_requests
            << ",\"drain_lag_us\":" << drain_lag.count()
            << ",\"prefix_hit_rate\":"
            << (engine.stats().prefix_cache_requests == 0
                    ? 0.0
                    : static_cast<double>(engine.prefix_cache_hits()) /
                          static_cast<double>(engine.stats().prefix_cache_requests))
            << ",\"result_hit_rate\":"
            << (engine.stats().result_cache_requests == 0
                    ? 0.0
                    : static_cast<double>(
                          engine.stats().result_cache_request_hits) /
                          static_cast<double>(engine.stats().result_cache_requests))
            << ",\"complete\":" << std::boolalpha << complete
            << ",\"zero_kv_leak\":" << zero_leak << "}\n";
  if (!complete || !zero_leak) {
    std::exit(EXIT_FAILURE);
  }
}

infersched::engine::EngineConfig BenchmarkConfig() {
  infersched::engine::EngineConfig config;
  config.kv_blocks = 4096;
  config.kv_block_size_tokens = 16;
  config.max_batch_sequences = 64;
  config.max_batch_tokens = 8192;
  config.max_prefill_sequences_per_iteration = 64;
  config.result_cache_entries = 64;
  // The scheduler-path baseline uses unique prompts and intentionally keeps
  // cache residency negligible; cache effects are benchmarked separately.
  config.prefix_cache_entries = 1;
  config.cost_model.prefill_fixed = std::chrono::nanoseconds::zero();
  config.cost_model.prefill_per_token = std::chrono::nanoseconds::zero();
  config.cost_model.decode_fixed = std::chrono::nanoseconds::zero();
  config.cost_model.decode_per_sequence = std::chrono::nanoseconds::zero();
  return config;
}

infersched::engine::EngineRequest MakeRequest(const std::size_t index) {
  return infersched::engine::EngineRequest{
      .request_id = "request-" + std::to_string(index),
      .model_revision = "benchmark-model",
      .tokenizer_revision = "benchmark-tokenizer",
      .prompt_hash = "prompt-" + std::to_string(index),
      .prompt_tokens = 16 + (index % 113),
      .max_output_tokens = 1 + (index % 16),
      .sampling_seed = index,
  };
}

template <typename Engine>
void RunScheduler(const Arguments& arguments) {
  Engine engine(BenchmarkConfig());
  std::uint64_t token_count = 0;
  for (std::size_t index = 0; index < arguments.requests; ++index) {
    auto request = MakeRequest(index);
    token_count += request.prompt_tokens + request.max_output_tokens;
    engine.Submit(std::move(request));
  }
  const auto start = std::chrono::steady_clock::now();
  engine.RunUntilIdle();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double seconds = std::chrono::duration<double>(elapsed).count();
  const bool complete = engine.results().size() == arguments.requests;
  engine.ClearCaches();
  const bool zero_leak = engine.kv_cache().allocated_blocks() == 0;
  std::cout << "{\"mode\":\"scheduler\",\"engine\":\""
            << arguments.engine << "\",\"requests\":" << arguments.requests
            << ",\"seconds\":" << seconds
            << ",\"requests_per_second\":"
            << static_cast<double>(arguments.requests) / seconds
            << ",\"tokens_per_second\":"
            << static_cast<double>(token_count) / seconds
            << ",\"complete\":" << std::boolalpha << complete
            << ",\"zero_kv_leak\":" << zero_leak << "}\n";
  if (!complete || !zero_leak) {
    std::exit(EXIT_FAILURE);
  }
}

class MutexQueue {
 public:
  void Push(const std::uint64_t value) {
    std::lock_guard lock(mutex_);
    values_.push_back(value);
  }
  bool Pop(std::uint64_t& value) {
    std::lock_guard lock(mutex_);
    if (values_.empty()) {
      return false;
    }
    value = values_.front();
    values_.pop_front();
    return true;
  }

 private:
  std::mutex mutex_;
  std::deque<std::uint64_t> values_;
};

template <typename Push, typename Pop>
void RunQueueWorkers(const Arguments& arguments, Push push, Pop pop) {
  std::atomic<bool> start{false};
  std::vector<std::thread> producers;
  producers.reserve(arguments.producers);
  const std::size_t per_producer =
      (arguments.requests + arguments.producers - 1) / arguments.producers;
  for (std::size_t producer = 0; producer < arguments.producers; ++producer) {
    producers.emplace_back([&, producer] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const std::size_t begin = producer * per_producer;
      const std::size_t end = std::min(begin + per_producer, arguments.requests);
      for (std::size_t value = begin; value < end; ++value) {
        while (!push(static_cast<std::uint64_t>(value))) {
          std::this_thread::yield();
        }
      }
    });
  }
  const auto begin_time = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  std::size_t consumed = 0;
  std::uint64_t checksum = 0;
  while (consumed < arguments.requests) {
    std::uint64_t value = 0;
    if (!pop(value)) {
      std::this_thread::yield();
      continue;
    }
    checksum += value;
    ++consumed;
  }
  for (auto& producer : producers) {
    producer.join();
  }
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - begin_time)
                             .count();
  const std::uint64_t expected =
      static_cast<std::uint64_t>(arguments.requests) *
      static_cast<std::uint64_t>(arguments.requests - 1) / 2;
  std::cout << "{\"mode\":\"queue\",\"queue\":\"" << arguments.queue
            << "\",\"events\":" << arguments.requests
            << ",\"producers\":" << arguments.producers
            << ",\"seconds\":" << seconds << ",\"events_per_second\":"
            << static_cast<double>(arguments.requests) / seconds
            << ",\"checksum_ok\":" << std::boolalpha
            << (checksum == expected) << "}\n";
  if (checksum != expected) {
    std::exit(EXIT_FAILURE);
  }
}

void RunQueue(const Arguments& arguments) {
  if (arguments.queue == "mpsc") {
    infersched::concurrency::MpscRing<std::uint64_t> queue(65'536);
    RunQueueWorkers(
        arguments,
        [&](const std::uint64_t value) { return queue.TryPush(value); },
        [&](std::uint64_t& value) {
          auto result = queue.TryPop();
          if (!result.has_value()) {
            return false;
          }
          value = *result;
          return true;
        });
    return;
  }
  if (arguments.queue == "mutex") {
    MutexQueue queue;
    RunQueueWorkers(
        arguments,
        [&](const std::uint64_t value) {
          queue.Push(value);
          return true;
        },
        [&](std::uint64_t& value) { return queue.Pop(value); });
    return;
  }
  std::cerr << "queue must be mpsc or mutex\n";
  std::exit(EXIT_FAILURE);
}

class GlobalRegistry {
 public:
  void Upsert(std::string key, const std::uint64_t value) {
    std::lock_guard lock(mutex_);
    values_.insert_or_assign(std::move(key), value);
  }
  std::size_t size() const {
    std::lock_guard lock(mutex_);
    return values_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::uint64_t> values_;
};

template <typename Registry>
void RunRegistryImpl(const Arguments& arguments, Registry& registry) {
  std::atomic<bool> start{false};
  std::vector<std::thread> writers;
  writers.reserve(arguments.producers);
  const std::size_t per_producer =
      (arguments.requests + arguments.producers - 1) / arguments.producers;
  for (std::size_t producer = 0; producer < arguments.producers; ++producer) {
    writers.emplace_back([&, producer] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const std::size_t begin = producer * per_producer;
      const std::size_t end = std::min(begin + per_producer, arguments.requests);
      for (std::size_t value = begin; value < end; ++value) {
        registry.Upsert("request-" + std::to_string(value), value);
      }
    });
  }
  const auto begin_time = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  for (auto& writer : writers) {
    writer.join();
  }
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - begin_time)
                             .count();
  const bool complete = registry.size() == arguments.requests;
  std::cout << "{\"mode\":\"registry\",\"registry\":\""
            << arguments.registry << "\",\"operations\":" << arguments.requests
            << ",\"writers\":" << arguments.producers
            << ",\"seconds\":" << seconds
            << ",\"operations_per_second\":"
            << static_cast<double>(arguments.requests) / seconds
            << ",\"complete\":" << std::boolalpha << complete << "}\n";
  if (!complete) {
    std::exit(EXIT_FAILURE);
  }
}

void RunRegistry(const Arguments& arguments) {
  if (arguments.registry == "global") {
    GlobalRegistry registry;
    RunRegistryImpl(arguments, registry);
    return;
  }
  if (arguments.registry == "sharded") {
    infersched::concurrency::ShardedRegistry<std::uint64_t> registry(32);
    RunRegistryImpl(arguments, registry);
    return;
  }
  std::cerr << "registry must be global or sharded\n";
  std::exit(EXIT_FAILURE);
}

}  // namespace

int main(const int argc, char** argv) {
  const Arguments arguments = ParseArguments(argc, argv);
  if (arguments.mode == "scheduler") {
    if (arguments.engine == "continuous") {
      RunScheduler<infersched::engine::ContinuousEngine>(arguments);
      return EXIT_SUCCESS;
    }
    if (arguments.engine == "static") {
      RunScheduler<infersched::engine::DeterministicEngine>(arguments);
      return EXIT_SUCCESS;
    }
    std::cerr << "engine must be continuous or static\n";
    return EXIT_FAILURE;
  }
  if (arguments.mode == "open_loop") {
    RunOpenLoop(arguments);
    return EXIT_SUCCESS;
  }
  if (arguments.mode == "queue") {
    RunQueue(arguments);
    return EXIT_SUCCESS;
  }
  if (arguments.mode == "registry") {
    RunRegistry(arguments);
    return EXIT_SUCCESS;
  }
  std::cerr << "mode must be scheduler, open_loop, queue, or registry\n";
  return EXIT_FAILURE;
}
