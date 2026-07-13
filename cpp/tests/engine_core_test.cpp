#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "infersched/engine/execution_state.hpp"
#include "infersched/engine/deterministic_engine.hpp"
#include "infersched/engine/continuous_engine.hpp"
#include "infersched/engine/fake_clock.hpp"
#include "infersched/engine/paged_kv_cache.hpp"
#include "infersched/engine/result_cache.hpp"
#include "infersched/engine/result_cache_key.hpp"
#include "infersched/engine/radix_prefix_cache.hpp"
#include "infersched/engine/request_queue.hpp"
#include "infersched/engine/static_batch_policy.hpp"
#include "infersched/distributed/router_service.hpp"

namespace {

using infersched::engine::ExecutionState;
using infersched::engine::ExecutionStateMachine;
using infersched::engine::DeterministicEngine;
using infersched::engine::ContinuousEngine;
using infersched::engine::EngineConfig;
using infersched::engine::EngineRequest;
using infersched::engine::EngineResult;
using infersched::engine::FakeClock;
using infersched::engine::PagedKvCache;
using infersched::engine::PendingRequest;
using infersched::engine::ResultCache;
using infersched::engine::StaticBatchPolicy;

TEST(ExecutionState, AcceptsOnlyDocumentedTransitions) {
  ExecutionStateMachine machine;
  EXPECT_FALSE(machine.TransitionTo(ExecutionState::kDecode));
  EXPECT_TRUE(machine.TransitionTo(ExecutionState::kPrefill));
  EXPECT_TRUE(machine.TransitionTo(ExecutionState::kPreempted));
  EXPECT_TRUE(machine.TransitionTo(ExecutionState::kWaitingAdmission));
  EXPECT_TRUE(machine.TransitionTo(ExecutionState::kPrefill));
  EXPECT_TRUE(machine.TransitionTo(ExecutionState::kDecode));
  EXPECT_TRUE(machine.TransitionTo(ExecutionState::kCompleted));
  EXPECT_FALSE(machine.TransitionTo(ExecutionState::kFailed));
  EXPECT_EQ(machine.state(), ExecutionState::kCompleted);
}

TEST(ExecutionState, FailureIsTerminalFromEveryLiveState) {
  for (const auto state : {ExecutionState::kWaitingAdmission,
                           ExecutionState::kPrefill, ExecutionState::kDecode,
                           ExecutionState::kPreempted}) {
    ExecutionStateMachine machine;
    if (state == ExecutionState::kPrefill || state == ExecutionState::kDecode) {
      ASSERT_TRUE(machine.TransitionTo(ExecutionState::kPrefill));
    }
    if (state == ExecutionState::kDecode) {
      ASSERT_TRUE(machine.TransitionTo(ExecutionState::kDecode));
    }
    if (state == ExecutionState::kPreempted) {
      ASSERT_TRUE(machine.TransitionTo(ExecutionState::kPrefill));
      ASSERT_TRUE(machine.TransitionTo(ExecutionState::kPreempted));
    }
    ASSERT_EQ(machine.state(), state);
    EXPECT_TRUE(machine.TransitionTo(ExecutionState::kFailed));
    EXPECT_FALSE(machine.TransitionTo(ExecutionState::kWaitingAdmission));
  }
}

TEST(PagedKvCache, AllocationFailureRollsBackCompletely) {
  PagedKvCache cache(/*block_count=*/4, /*block_size_tokens=*/16);
  ASSERT_EQ(cache.Allocate("first", 3),
            PagedKvCache::AllocationStatus::kAllocated);
  const auto free_before = cache.free_blocks();

  EXPECT_EQ(cache.Allocate("too-large", 2),
            PagedKvCache::AllocationStatus::kInsufficientBlocks);
  EXPECT_EQ(cache.free_blocks(), free_before);
  EXPECT_TRUE(cache.BlocksFor("too-large").empty());
  EXPECT_TRUE(cache.CheckInvariants());
}

TEST(PagedKvCache, IncrementalGrowthIsTransactional) {
  PagedKvCache cache(/*block_count=*/4, /*block_size_tokens=*/16);
  ASSERT_EQ(cache.Allocate("sequence", 2),
            PagedKvCache::AllocationStatus::kAllocated);
  ASSERT_EQ(cache.AppendBlocks("sequence", 1),
            PagedKvCache::AllocationStatus::kAllocated);
  EXPECT_EQ(cache.BlocksFor("sequence").size(), 3u);
  EXPECT_EQ(cache.AppendBlocks("sequence", 2),
            PagedKvCache::AllocationStatus::kInsufficientBlocks);
  EXPECT_EQ(cache.BlocksFor("sequence").size(), 3u);
  EXPECT_TRUE(cache.CheckInvariants());
  EXPECT_TRUE(cache.Free("sequence"));
  EXPECT_EQ(cache.allocated_blocks(), 0u);
}

TEST(PagedKvCache, SharedPrefixUsesReferenceCounts) {
  PagedKvCache cache(/*block_count=*/8, /*block_size_tokens=*/16);
  ASSERT_EQ(cache.Allocate("prefix-owner", 3),
            PagedKvCache::AllocationStatus::kAllocated);
  const std::vector<PagedKvCache::BlockId> shared(
      cache.BlocksFor("prefix-owner").begin(),
      cache.BlocksFor("prefix-owner").begin() + 2);

  ASSERT_EQ(cache.Allocate("consumer", 4, shared),
            PagedKvCache::AllocationStatus::kAllocated);
  EXPECT_EQ(cache.RefCount(shared[0]), 2u);
  EXPECT_EQ(cache.RefCount(shared[1]), 2u);
  EXPECT_EQ(cache.allocated_blocks(), 5u);

  EXPECT_TRUE(cache.Free("prefix-owner"));
  EXPECT_EQ(cache.RefCount(shared[0]), 1u);
  EXPECT_TRUE(cache.CheckInvariants());
  EXPECT_TRUE(cache.Free("consumer"));
  EXPECT_EQ(cache.free_blocks(), cache.total_blocks());
  EXPECT_TRUE(cache.CheckInvariants());
}

TEST(PagedKvCache, PrefixPinOutlivesOriginatingSequence) {
  PagedKvCache cache(/*block_count=*/8, /*block_size_tokens=*/16);
  ASSERT_EQ(cache.Allocate("origin", 3),
            PagedKvCache::AllocationStatus::kAllocated);
  const std::vector<PagedKvCache::BlockId> prefix(
      cache.BlocksFor("origin").begin(), cache.BlocksFor("origin").end());
  ASSERT_TRUE(cache.PinPrefix("model:prompt", prefix));

  ASSERT_TRUE(cache.Free("origin"));
  EXPECT_EQ(cache.allocated_blocks(), 3u);
  for (const auto block : prefix) {
    EXPECT_EQ(cache.RefCount(block), 1u);
  }
  EXPECT_TRUE(cache.CheckInvariants());

  EXPECT_TRUE(cache.UnpinPrefix("model:prompt"));
  EXPECT_EQ(cache.allocated_blocks(), 0u);
  EXPECT_TRUE(cache.CheckInvariants());
}

TEST(ResultCache, EvictsLeastRecentlyUsedAndCountsLookups) {
  ResultCache cache(/*capacity=*/2);
  cache.Put("a", "result-a");
  cache.Put("b", "result-b");
  ASSERT_EQ(cache.Get("a"), std::optional<std::string>{"result-a"});
  cache.Put("c", "result-c");

  EXPECT_FALSE(cache.Get("b").has_value());
  EXPECT_EQ(cache.Get("a"), std::optional<std::string>{"result-a"});
  EXPECT_EQ(cache.Get("c"), std::optional<std::string>{"result-c"});
  EXPECT_EQ(cache.hits(), 3u);
  EXPECT_EQ(cache.misses(), 1u);
}

TEST(ResultCache, TypedKeyIsUnambiguousAndIncludesSamplingState) {
  using infersched::engine::BuildResultCacheKey;
  using infersched::engine::ResultCacheKeyParts;
  const auto first = BuildResultCacheKey(ResultCacheKeyParts{
      .model_revision = "ab", .tokenizer_revision = "c", .prompt_hash = "d",
      .max_output_tokens = 8, .temperature_micros = 700'000,
      .top_p_micros = 900'000, .seed = 1});
  const auto boundary_changed = BuildResultCacheKey(ResultCacheKeyParts{
      .model_revision = "a", .tokenizer_revision = "bc", .prompt_hash = "d",
      .max_output_tokens = 8, .temperature_micros = 700'000,
      .top_p_micros = 900'000, .seed = 1});
  const auto seed_changed = BuildResultCacheKey(ResultCacheKeyParts{
      .model_revision = "ab", .tokenizer_revision = "c", .prompt_hash = "d",
      .max_output_tokens = 8, .temperature_micros = 700'000,
      .top_p_micros = 900'000, .seed = 2});
  EXPECT_NE(first, boundary_changed);
  EXPECT_NE(first, seed_changed);
}

TEST(PrefixCache, FindsLongestBlockAlignedPrefixAcrossDifferentPrompts) {
  PagedKvCache kv(/*block_count=*/16, /*block_size_tokens=*/4);
  infersched::engine::PrefixCache cache(/*capacity=*/8, kv);
  ASSERT_EQ(kv.Allocate("origin", 3),
            PagedKvCache::AllocationStatus::kAllocated);
  const std::vector<std::string> origin_hashes{"A", "B", "C"};
  ASSERT_EQ(cache.PutBlockPrefixes("model-tokenizer", origin_hashes,
                                  kv.BlocksFor("origin")),
            3u);

  const std::vector<std::string> related_hashes{"A", "B", "different"};
  const auto match = cache.GetLongest("model-tokenizer", related_hashes);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->matched_block_count, 2u);
  EXPECT_EQ(match->blocks.size(), 2u);
  EXPECT_EQ(cache.hits(), 1u);

  cache.Clear();
  EXPECT_TRUE(kv.Free("origin"));
  EXPECT_EQ(kv.allocated_blocks(), 0u);
  EXPECT_TRUE(kv.CheckInvariants());
}

TEST(RadixPrefixCache, StoresOneNodePerBlockAndFindsLongestPrefix) {
  PagedKvCache kv(/*block_count=*/32, /*block_size_tokens=*/4);
  infersched::engine::RadixPrefixCache cache(/*node_capacity=*/16, kv);
  ASSERT_EQ(kv.Allocate("origin", 4),
            PagedKvCache::AllocationStatus::kAllocated);
  const std::vector<std::string> hashes{"A", "B", "C", "D"};
  EXPECT_EQ(cache.PutBlockPrefixes("namespace", hashes,
                                  kv.BlocksFor("origin")),
            4u);
  EXPECT_EQ(cache.size(), 4u);

  const std::vector<std::string> related{"A", "B", "X"};
  const auto match = cache.GetLongest("namespace", related);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->matched_block_count, 2u);
  EXPECT_EQ(match->blocks.size(), 2u);

  cache.Clear();
  EXPECT_TRUE(kv.Free("origin"));
  EXPECT_EQ(kv.allocated_blocks(), 0u);
  EXPECT_TRUE(kv.CheckInvariants());
}

TEST(RequestQueue, OrdersReadyRequestsWithoutScanningFutureWork) {
  infersched::engine::RequestQueue queue;
  const auto now = FakeClock::time_point{};
  queue.Push(EngineRequest{.request_id = "future",
                           .arrival_time = now + std::chrono::seconds(1)});
  queue.Push(EngineRequest{.request_id = "b", .arrival_time = now});
  queue.Push(EngineRequest{.request_id = "a", .arrival_time = now});

  ASSERT_EQ(queue.PopReady(now)->request_id, "a");
  ASSERT_EQ(queue.PopReady(now)->request_id, "b");
  EXPECT_FALSE(queue.PopReady(now).has_value());
  ASSERT_TRUE(queue.NextArrival().has_value());
  EXPECT_EQ(*queue.NextArrival(), now + std::chrono::seconds(1));
}

TEST(PagedKvCache, SeededRandomOperationsPreserveInvariantsAndLeakNothing) {
  PagedKvCache cache(/*block_count=*/64, /*block_size_tokens=*/16);
  std::mt19937 random(/*seed=*/0xC0FFEEu);
  std::vector<std::string> live;

  for (std::size_t step = 0; step < 10'000; ++step) {
    const bool should_allocate = live.empty() || (random() % 100u) < 60u;
    if (should_allocate) {
      const std::string id = "seq-" + std::to_string(step);
      const auto blocks = static_cast<std::size_t>((random() % 8u) + 1u);
      if (cache.Allocate(id, blocks) == PagedKvCache::AllocationStatus::kAllocated) {
        live.push_back(id);
      }
    } else {
      const auto index = static_cast<std::size_t>(random()) % live.size();
      ASSERT_TRUE(cache.Free(live[index]));
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
    }
    ASSERT_TRUE(cache.CheckInvariants()) << "step=" << step;
  }

  for (const auto& id : live) {
    ASSERT_TRUE(cache.Free(id));
  }
  EXPECT_EQ(cache.allocated_blocks(), 0u);
  EXPECT_EQ(cache.free_blocks(), cache.total_blocks());
  EXPECT_TRUE(cache.CheckInvariants());
}

TEST(StaticBatchPolicy, IsDeterministicFcfsAndRespectsBothLimits) {
  FakeClock clock;
  StaticBatchPolicy policy(/*max_batch_sequences=*/2,
                           /*max_batch_tokens=*/12);
  const auto start = clock.Now();
  std::vector<PendingRequest> pending{
      {.request_id = "later", .prompt_tokens = 2, .max_output_tokens = 2,
       .arrival_time = start + std::chrono::milliseconds(2)},
      {.request_id = "b", .prompt_tokens = 3, .max_output_tokens = 3,
       .arrival_time = start},
      {.request_id = "a", .prompt_tokens = 3, .max_output_tokens = 3,
       .arrival_time = start},
      {.request_id = "too-large", .prompt_tokens = 20,
       .max_output_tokens = 1, .arrival_time = start},
  };

  clock.Advance(std::chrono::milliseconds(1));
  const auto first = policy.Select(pending, clock.Now());
  const auto second = policy.Select(pending, clock.Now());
  EXPECT_EQ(first, second);
  EXPECT_EQ(first, (std::vector<std::size_t>{2, 1}));
}

TEST(DeterministicEngine, ComposesBatchingCostAndDistinctCaches) {
  EngineConfig config;
  config.kv_blocks = 32;
  config.kv_block_size_tokens = 4;
  config.max_batch_sequences = 2;
  config.max_batch_tokens = 64;
  config.seed = 42;
  config.cost_model.jitter_basis_points = 50;

  DeterministicEngine engine(config);
  const EngineRequest first{.request_id = "first",
                            .model_revision = "model-v1",
                            .tokenizer_revision = "tok-v1",
                            .prompt_hash = "shared-prompt",
                            .prompt_tokens = 8,
                            .max_output_tokens = 3,
                            .sampling_seed = 11};
  EngineRequest second = first;
  second.request_id = "second";
  second.sampling_seed = 12;  // result miss, prefix hit
  EngineRequest third = second;
  third.request_id = "third";  // exact result key hit

  engine.Submit(first);
  engine.RunUntilIdle();
  const auto after_first = engine.clock().Now();
  engine.Submit(second);
  engine.RunUntilIdle();
  const auto after_second = engine.clock().Now();
  engine.Submit(third);
  engine.RunUntilIdle();

  ASSERT_EQ(engine.results().size(), 3u);
  EXPECT_FALSE(engine.results()[0].prefix_cache_hit);
  EXPECT_TRUE(engine.results()[1].prefix_cache_hit);
  EXPECT_TRUE(engine.results()[2].result_cache_hit);
  EXPECT_GT(after_second, after_first);
  EXPECT_EQ(engine.clock().Now(), after_second);  // result hit skips simulated compute
  EXPECT_EQ(engine.prefix_cache_hits(), 1u);
  EXPECT_EQ(engine.result_cache_hits(), 1u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());

  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(DeterministicEngine, SameSeedAndInputProduceIdenticalTimeline) {
  EngineConfig config;
  config.seed = 1234;
  config.cost_model.jitter_basis_points = 250;
  DeterministicEngine first(config);
  DeterministicEngine second(config);

  for (std::size_t index = 0; index < 6; ++index) {
    EngineRequest request{.request_id = "request-" + std::to_string(index),
                          .model_revision = "model",
                          .tokenizer_revision = "tokenizer",
                          .prompt_hash = "prompt-" + std::to_string(index),
                          .prompt_tokens = 4 + index,
                          .max_output_tokens = 2 + index,
                          .sampling_seed = index};
    first.Submit(request);
    second.Submit(std::move(request));
  }
  first.RunUntilIdle();
  second.RunUntilIdle();

  ASSERT_EQ(first.results().size(), second.results().size());
  for (std::size_t index = 0; index < first.results().size(); ++index) {
    EXPECT_EQ(first.results()[index].request_id, second.results()[index].request_id);
    EXPECT_EQ(first.results()[index].output, second.results()[index].output);
    EXPECT_EQ(first.results()[index].completed_at,
              second.results()[index].completed_at);
  }
  EXPECT_EQ(first.clock().Now(), second.clock().Now());
}

TEST(DeterministicEngine, RejectsImpossibleRequestAndStillRunsFutureWork) {
  EngineConfig config;
  config.kv_blocks = 4;
  config.kv_block_size_tokens = 4;
  config.max_batch_tokens = 32;
  DeterministicEngine engine(config);

  engine.Submit(EngineRequest{.request_id = "impossible",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "too-large",
                              .prompt_tokens = 17,
                              .max_output_tokens = 1,
                              .arrival_time = engine.clock().Now()});
  engine.Submit(EngineRequest{.request_id = "too-wide-for-batch",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "too-wide",
                              .prompt_tokens = 33,
                              .max_output_tokens = 0,
                              .arrival_time = engine.clock().Now()});
  engine.Submit(EngineRequest{
      .request_id = "future-valid",
      .model_revision = "model",
      .tokenizer_revision = "tokenizer",
      .prompt_hash = "valid",
      .prompt_tokens = 4,
      .max_output_tokens = 2,
      .arrival_time = engine.clock().Now() + std::chrono::milliseconds(5)});

  engine.RunUntilIdle();

  ASSERT_EQ(engine.results().size(), 3u);
  EXPECT_EQ(engine.results()[0].request_id, "impossible");
  EXPECT_FALSE(engine.results()[0].succeeded);
  EXPECT_EQ(engine.results()[0].failure_reason, "exceeds_kv_capacity");
  EXPECT_EQ(engine.results()[1].request_id, "too-wide-for-batch");
  EXPECT_FALSE(engine.results()[1].succeeded);
  EXPECT_EQ(engine.results()[1].failure_reason, "exceeds_max_batch_tokens");
  EXPECT_EQ(engine.results()[2].request_id, "future-valid");
  EXPECT_TRUE(engine.results()[2].succeeded);
  EXPECT_GE(engine.results()[2].completed_at.time_since_epoch(),
            std::chrono::milliseconds(5));
  EXPECT_EQ(engine.pending_count(), 0u);

  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(ContinuousEngine, AdmitsNewRequestWhileLongDecodeIsActive) {
  EngineConfig config;
  config.kv_blocks = 32;
  config.kv_block_size_tokens = 4;
  config.max_batch_sequences = 2;
  config.max_batch_tokens = 32;
  config.cost_model.prefill_fixed = std::chrono::microseconds(10);
  config.cost_model.prefill_per_token = std::chrono::microseconds(1);
  config.cost_model.decode_fixed = std::chrono::microseconds(5);
  config.cost_model.decode_per_sequence = std::chrono::microseconds(1);
  ContinuousEngine engine(config);

  engine.Submit(EngineRequest{.request_id = "long",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "long-prompt",
                              .prompt_tokens = 4,
                              .max_output_tokens = 8});
  engine.Submit(EngineRequest{
      .request_id = "late-short",
      .model_revision = "model",
      .tokenizer_revision = "tokenizer",
      .prompt_hash = "short-prompt",
      .prompt_tokens = 4,
      .max_output_tokens = 1,
      .arrival_time = engine.clock().Now() + std::chrono::microseconds(15)});

  engine.RunUntilIdle();
  ASSERT_EQ(engine.results().size(), 2u);
  EXPECT_EQ(engine.results()[0].request_id, "late-short");
  EXPECT_EQ(engine.results()[1].request_id, "long");
  EXPECT_EQ(engine.stats().peak_active_sequences, 2u);
  EXPECT_EQ(engine.stats().admissions, 2u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
}

TEST(ContinuousEngine, FinishesLateShortRequestBeforeStaticBaseline) {
  EngineConfig config;
  config.kv_blocks = 32;
  config.kv_block_size_tokens = 4;
  config.max_batch_sequences = 2;
  config.max_batch_tokens = 32;
  config.cost_model.prefill_fixed = std::chrono::microseconds(10);
  config.cost_model.prefill_per_token = std::chrono::microseconds(1);
  config.cost_model.decode_fixed = std::chrono::microseconds(5);
  config.cost_model.decode_per_sequence = std::chrono::microseconds(1);

  const EngineRequest long_request{.request_id = "long",
                                   .model_revision = "model",
                                   .tokenizer_revision = "tokenizer",
                                   .prompt_hash = "long-prompt",
                                   .prompt_tokens = 4,
                                   .max_output_tokens = 8};
  const EngineRequest late_short{
      .request_id = "late-short",
      .model_revision = "model",
      .tokenizer_revision = "tokenizer",
      .prompt_hash = "short-prompt",
      .prompt_tokens = 4,
      .max_output_tokens = 1,
      .arrival_time = FakeClock::time_point{std::chrono::microseconds(15)}};

  DeterministicEngine static_engine(config);
  static_engine.Submit(long_request);
  static_engine.Submit(late_short);
  static_engine.RunUntilIdle();

  ContinuousEngine continuous_engine(config);
  continuous_engine.Submit(long_request);
  continuous_engine.Submit(late_short);
  continuous_engine.RunUntilIdle();

  const auto static_short = std::ranges::find(
      static_engine.results(), "late-short", &EngineResult::request_id);
  const auto continuous_short = std::ranges::find(
      continuous_engine.results(), "late-short", &EngineResult::request_id);
  ASSERT_NE(static_short, static_engine.results().end());
  ASSERT_NE(continuous_short, continuous_engine.results().end());
  EXPECT_LT(continuous_short->completed_at, static_short->completed_at);

  static_engine.ClearCaches();
  continuous_engine.ClearCaches();
  EXPECT_EQ(static_engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_EQ(continuous_engine.kv_cache().allocated_blocks(), 0u);
}

TEST(ContinuousEngine, PreemptsUnderDecodePressureAndEventuallyCompletesAll) {
  EngineConfig config;
  config.kv_blocks = 5;
  config.kv_block_size_tokens = 2;
  config.max_batch_sequences = 2;
  config.max_batch_tokens = 16;
  ContinuousEngine engine(config);

  engine.Submit(EngineRequest{.request_id = "high-priority",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "high",
                              .prompt_tokens = 2,
                              .max_output_tokens = 6,
                              .priority = 10});
  engine.Submit(EngineRequest{.request_id = "low-priority",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "low",
                              .prompt_tokens = 2,
                              .max_output_tokens = 6,
                              .priority = 1});

  engine.RunUntilIdle();
  ASSERT_EQ(engine.results().size(), 2u);
  EXPECT_GT(engine.stats().preemptions, 0u);
  EXPECT_EQ(engine.results()[0].request_id, "high-priority");
  EXPECT_EQ(engine.pending_count(), 0u);
  EXPECT_EQ(engine.active_count(), 0u);
  EXPECT_TRUE(std::ranges::all_of(engine.results(),
                                  [](const auto& result) { return result.succeeded; }));
  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(ContinuousEngine, SeededWorkloadPreservesKvInvariantsEveryIteration) {
  EngineConfig config;
  config.kv_blocks = 48;
  config.kv_block_size_tokens = 4;
  config.max_batch_sequences = 6;
  config.max_batch_tokens = 64;
  config.max_prefill_sequences_per_iteration = 2;
  config.seed = 0xBADC0DEu;
  config.cost_model.jitter_basis_points = 100;
  ContinuousEngine engine(config);
  std::mt19937 random(0xBADC0DEu);

  for (std::size_t index = 0; index < 100; ++index) {
    engine.Submit(EngineRequest{
        .request_id = "continuous-" + std::to_string(index),
        .model_revision = "model",
        .tokenizer_revision = "tokenizer",
        .prompt_hash = "prompt-" + std::to_string(index),
        .prompt_tokens = static_cast<std::size_t>((random() % 12u) + 1u),
        .max_output_tokens = static_cast<std::size_t>((random() % 16u) + 1u),
        .priority = static_cast<std::uint32_t>(random() % 4u),
        .sampling_seed = index});
  }

  std::size_t safety_iterations = 0;
  while (engine.pending_count() > 0 || engine.active_count() > 0) {
    ASSERT_TRUE(engine.RunOneIteration());
    ASSERT_TRUE(engine.kv_cache().CheckInvariants());
    ASSERT_LT(++safety_iterations, 10'000u);
  }
  EXPECT_EQ(engine.results().size(), 100u);
  EXPECT_TRUE(std::ranges::all_of(engine.results(),
                                  [](const auto& result) { return result.succeeded; }));
  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(ContinuousEngine, ReusesTruePrefixAcrossDifferentPrompts) {
  EngineConfig config;
  config.kv_blocks = 32;
  config.kv_block_size_tokens = 4;
  config.max_batch_sequences = 1;
  ContinuousEngine engine(config);

  engine.Submit(EngineRequest{.request_id = "origin",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "whole-prompt-one",
                              .prompt_block_hashes = {"A", "B", "C"},
                              .prompt_tokens = 12,
                              .max_output_tokens = 1});
  engine.RunUntilIdle();
  engine.Submit(EngineRequest{.request_id = "related",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "whole-prompt-two",
                              .prompt_block_hashes = {"A", "B", "D"},
                              .prompt_tokens = 12,
                              .max_output_tokens = 1});
  engine.RunUntilIdle();

  ASSERT_EQ(engine.results().size(), 2u);
  EXPECT_FALSE(engine.results()[0].prefix_cache_hit);
  EXPECT_TRUE(engine.results()[1].prefix_cache_hit);
  EXPECT_EQ(engine.prefix_cache_hits(), 1u);
  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(ContinuousEngine, PrefixHitMetricCountsRequestsNotEvictionRetries) {
  EngineConfig config;
  config.kv_blocks = 3;
  config.kv_block_size_tokens = 2;
  config.max_batch_sequences = 1;
  config.prefix_cache_entries = 8;
  ContinuousEngine engine(config);

  engine.Submit(EngineRequest{.request_id = "origin",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "ABC",
                              .prompt_block_hashes = {"A", "B", "C"},
                              .prompt_tokens = 6,
                              .max_output_tokens = 0});
  engine.RunUntilIdle();
  engine.Submit(EngineRequest{.request_id = "related-under-pressure",
                              .model_revision = "model",
                              .tokenizer_revision = "tokenizer",
                              .prompt_hash = "ABD",
                              .prompt_block_hashes = {"A", "B", "D"},
                              .prompt_tokens = 6,
                              .max_output_tokens = 0});
  engine.RunUntilIdle();

  ASSERT_EQ(engine.results().size(), 2u);
  EXPECT_EQ(engine.stats().prefix_cache_requests, 2u);
  EXPECT_EQ(engine.prefix_cache_hits(), 1u);
  EXPECT_EQ(engine.prefix_cache_misses(), 1u);
  EXPECT_TRUE(engine.results()[1].prefix_cache_hit);
  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(RouterService, ExpiredHeartbeatLeaseCannotRemainLive) {
  infersched::distributed::RouterService service(std::chrono::milliseconds(20));
  infersched::v1::RegisterRequest request;
  request.set_engine_id("lease-engine");
  request.set_incarnation_id("lease-incarnation");
  request.set_endpoint("127.0.0.1:1");
  infersched::v1::RegisterReply register_reply;
  ASSERT_TRUE(service.Register(nullptr, &request, &register_reply).ok());
  ASSERT_TRUE(register_reply.accepted());
  EXPECT_EQ(register_reply.assigned_lease_ttl_ms(), 20u);
  ASSERT_TRUE(service.WaitForEngine(std::chrono::milliseconds(1)).has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(service.WaitForEngine(std::chrono::milliseconds(1)).has_value());
  infersched::v1::HeartbeatRequest heartbeat;
  heartbeat.set_engine_id("lease-engine");
  heartbeat.set_incarnation_id("lease-incarnation");
  infersched::v1::HeartbeatReply heartbeat_reply;
  ASSERT_TRUE(service.Heartbeat(nullptr, &heartbeat, &heartbeat_reply).ok());
  EXPECT_FALSE(heartbeat_reply.lease_ok());
}

}  // namespace
