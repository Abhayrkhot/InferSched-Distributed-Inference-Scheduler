// The StepExecutor contract, checked without a model: the engine must hand
// an executor a consistent view of which sequences run, which KV blocks are
// alive, and when they die. A real-model executor relies on exactly this to
// keep llama.cpp's KV cells in step with the scheduler's PagedKvCache.
#include <chrono>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <gtest/gtest.h>

#include "infersched/engine/continuous_engine.hpp"
#include "infersched/engine/prompt_blocks.hpp"

namespace infersched::engine {
namespace {

class RecordingExecutor final : public StepExecutor {
 public:
  PrefillResult Prefill(const PrefillWork& work) override {
    ++prefills;
    EXPECT_TRUE(live.insert(work.request->request_id).second)
        << "prefill of a sequence that is already running";
    EXPECT_LE(work.matched_blocks, work.blocks.size());
    for (std::size_t index = 0; index < work.matched_blocks; ++index) {
      // A prefix match may only reference blocks whose KV still exists.
      EXPECT_TRUE(alive.contains(work.blocks[index])) << "match on a freed block";
    }
    for (const auto block : work.blocks) {
      alive.insert(block);
    }
    // Reuse less than claimed, as a real executor does when it must
    // recompute; the engine has to account with this number.
    const std::size_t reused = work.cached_tokens / 2;
    claimed_tokens += work.cached_tokens;
    reused_tokens += reused;
    return PrefillResult{.compute = std::chrono::microseconds(5), .reused_tokens = reused};
  }

  std::chrono::nanoseconds Decode(const std::size_t batch_size,
                                  const std::span<const std::string_view> sequences) override {
    EXPECT_EQ(batch_size, sequences.size());
    std::unordered_set<std::string_view> unique;
    for (const auto id : sequences) {
      EXPECT_TRUE(live.contains(std::string(id))) << "decode of a released sequence";
      EXPECT_TRUE(unique.insert(id).second) << "sequence decoded twice in one step";
    }
    ++decodes;
    return std::chrono::microseconds(10);
  }

  void Release(const std::string& request_id, const std::span<const PagedKvCache::BlockId> blocks,
               const PagedKvCache& kv) override {
    EXPECT_EQ(live.erase(request_id), 1U) << "release of a sequence that is not running";
    for (const auto block : blocks) {
      // Blocks nobody references any more were reported before Release.
      if (kv.RefCount(block) == 0) {
        EXPECT_FALSE(alive.contains(block));
      }
    }
    ++releases;
  }

  void OnBlockFreed(const PagedKvCache::BlockId block) noexcept override {
    alive.erase(block);
    ++frees;
  }

  std::string Output(const EngineRequest& request) override {
    EXPECT_TRUE(live.contains(request.request_id));
    return "out:" + request.request_id;
  }
  bool tracks_sequences() const noexcept override { return true; }
  std::string_view name() const noexcept override { return "recording"; }

  std::set<std::string> live;
  std::unordered_set<PagedKvCache::BlockId> alive;
  std::size_t prefills{};
  std::size_t decodes{};
  std::size_t releases{};
  std::size_t frees{};
  std::size_t claimed_tokens{};
  std::size_t reused_tokens{};
};

TEST(ExecutorContract, EngineKeepsExecutorInStepUnderPreemptionCancelAndEviction) {
  EngineConfig config;
  config.kv_blocks = 48;  // Small enough to force preemption.
  config.max_batch_sequences = 6;
  config.max_prefill_sequences_per_iteration = 3;
  config.prefix_cache_entries = 24;  // Small enough to force radix eviction.
  config.result_cache_entries = 1;
  auto owned = std::make_unique<RecordingExecutor>();
  RecordingExecutor& executor = *owned;
  ContinuousEngine engine(config, std::move(owned));

  std::mt19937_64 random(11);
  std::uniform_int_distribution<std::size_t> prompt(8, 120);
  std::uniform_int_distribution<std::size_t> output(1, 90);
  std::uniform_int_distribution<int> prefix(0, 3);
  std::size_t submitted = 0;
  std::size_t cancel_attempts = 0;
  for (int round = 0; round < 60; ++round) {
    for (int burst = 0; burst < 3; ++burst) {
      EngineRequest request;
      request.request_id = "r" + std::to_string(submitted);
      request.model_revision = "m";
      request.tokenizer_revision = "t";
      request.prompt_hash = "h" + std::to_string(submitted);
      request.prompt_tokens = prompt(random);
      request.max_output_tokens = output(random);
      const int shared = prefix(random);
      request.prompt_block_hashes = PromptBlockHashes(
          request.prompt_hash, shared == 0 ? "" : "sys" + std::to_string(shared), 48,
          request.prompt_tokens, config.kv_block_size_tokens);
      request.arrival_time = engine.clock().Now();
      engine.Submit(std::move(request));
      ++submitted;
    }
    for (int step = 0; step < 4; ++step) {
      static_cast<void>(engine.RunOneIteration());
    }
    if (round % 5 == 0) {
      ++cancel_attempts;
      static_cast<void>(engine.Cancel("r" + std::to_string(submitted - 2), "client_cancelled"));
    }
  }
  engine.RunUntilIdle();

  EXPECT_TRUE(engine.idle());
  EXPECT_TRUE(executor.live.empty());
  EXPECT_EQ(executor.prefills, executor.releases);
  EXPECT_GT(engine.stats().preemptions, 0U);
  EXPECT_GT(engine.stats().cancellations, 0U);
  EXPECT_GT(executor.claimed_tokens, 0U);
  // Engine statistics use the executor's reuse, not the prefix cache's claim.
  EXPECT_EQ(engine.stats().prefill_tokens_reused, executor.reused_tokens);

  engine.ClearCaches();
  EXPECT_EQ(engine.kv_cache().free_blocks(), engine.kv_cache().total_blocks());
  EXPECT_TRUE(executor.alive.empty()) << executor.alive.size() << " blocks never freed";
  EXPECT_GT(executor.frees, 0U);
  static_cast<void>(cancel_attempts);
}

TEST(ExecutorContract, CostModelExecutorIsTheDefault) {
  ContinuousEngine engine(EngineConfig{});
  EXPECT_EQ(engine.executor().name(), "cost-model");
  EXPECT_FALSE(engine.executor().tracks_sequences());
  EXPECT_FALSE(engine.executor().measured());
}

}  // namespace
}  // namespace infersched::engine
