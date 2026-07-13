#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "infersched/concurrency/mpsc_ring.hpp"
#include "infersched/concurrency/sharded_registry.hpp"
#include "infersched/metrics/histogram.hpp"

namespace {

TEST(MpscRing, MultipleProducersDeliverEveryValueExactlyOnce) {
  constexpr std::size_t kProducerCount = 4;
  constexpr std::size_t kValuesPerProducer = 10'000;
  constexpr std::size_t kTotal = kProducerCount * kValuesPerProducer;
  infersched::concurrency::MpscRing<std::size_t> queue(1024);
  std::atomic<bool> start{false};
  std::vector<std::thread> producers;
  producers.reserve(kProducerCount);

  for (std::size_t producer = 0; producer < kProducerCount; ++producer) {
    producers.emplace_back([&, producer] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t index = 0; index < kValuesPerProducer; ++index) {
        const std::size_t value = producer * kValuesPerProducer + index;
        while (!queue.TryPush(value)) {
          std::this_thread::yield();
        }
      }
    });
  }

  std::vector<bool> seen(kTotal, false);
  start.store(true, std::memory_order_release);
  std::size_t consumed = 0;
  while (consumed < kTotal) {
    const auto value = queue.TryPop();
    if (!value.has_value()) {
      std::this_thread::yield();
      continue;
    }
    ASSERT_LT(*value, kTotal);
    ASSERT_FALSE(seen[*value]);
    seen[*value] = true;
    ++consumed;
  }
  for (auto& producer : producers) {
    producer.join();
  }
  EXPECT_FALSE(queue.TryPop().has_value());
}

TEST(ShardedRegistry, ConcurrentWritersRemainVisible) {
  constexpr std::size_t kThreadCount = 8;
  constexpr std::size_t kValuesPerThread = 1'000;
  infersched::concurrency::ShardedRegistry<std::uint64_t> registry(32);
  std::vector<std::thread> writers;
  writers.reserve(kThreadCount);
  for (std::size_t thread = 0; thread < kThreadCount; ++thread) {
    writers.emplace_back([&, thread] {
      for (std::size_t index = 0; index < kValuesPerThread; ++index) {
        const std::uint64_t value = thread * kValuesPerThread + index;
        registry.Upsert("request-" + std::to_string(value), value);
      }
    });
  }
  for (auto& writer : writers) {
    writer.join();
  }

  EXPECT_EQ(registry.size(), kThreadCount * kValuesPerThread);
  for (std::uint64_t value = 0; value < kThreadCount * kValuesPerThread;
       value += 997) {
    EXPECT_EQ(registry.Get("request-" + std::to_string(value)), value);
  }
}

TEST(Histogram, ReportsLogBucketPercentiles) {
  infersched::metrics::Histogram histogram;
  for (std::uint64_t value = 1; value <= 100; ++value) {
    histogram.Record(value);
  }
  EXPECT_EQ(histogram.Count(), 100u);
  EXPECT_GE(histogram.Percentile(50.0), 50u);
  EXPECT_LE(histogram.Percentile(50.0), 64u);
  EXPECT_GE(histogram.Percentile(99.0), 99u);
  EXPECT_LE(histogram.Percentile(99.0), 128u);
}

}  // namespace
