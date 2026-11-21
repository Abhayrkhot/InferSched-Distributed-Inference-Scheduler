// Real-model tests: ContinuousEngine scheduling llama.cpp. Set
// INFERSCHED_TEST_MODEL to a GGUF file (CI uses the 260K-parameter
// tinyllamas/stories260K.gguf on CPU); without it every test is skipped.
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "infersched/engine/continuous_engine.hpp"
#include "infersched/engine/llama_executor.hpp"
#include "infersched/engine/prompt_blocks.hpp"

namespace infersched::engine {
namespace {

std::string ModelPath() {
  const char* path = std::getenv("INFERSCHED_TEST_MODEL");
  return path == nullptr ? std::string{} : std::string(path);
}

#define REQUIRE_MODEL()                                          \
  if (ModelPath().empty()) {                                     \
    GTEST_SKIP() << "set INFERSCHED_TEST_MODEL to a GGUF model"; \
  }

struct Rig {
  LlamaExecutor* executor{};
  std::unique_ptr<ContinuousEngine> engine;
  std::map<std::string, std::vector<std::int32_t>> outputs;
};

std::unique_ptr<Rig> MakeRig(const EngineConfig& config, const std::size_t holders = 8,
                             const bool backend_sampling = false) {
  auto rig = std::make_unique<Rig>();
  LlamaExecutorConfig llama;
  llama.model_path = ModelPath();
  llama.gpu_layers = 0;
  llama.kv_tokens = config.kv_blocks * config.kv_block_size_tokens;
  llama.block_size_tokens = config.kv_block_size_tokens;
  llama.max_sequences = config.max_batch_sequences;
  llama.holder_sequences = holders;
  llama.max_batch_tokens = config.max_batch_tokens;
  llama.flash_attention = false;
  llama.threads = 2;
  llama.backend_sampling = backend_sampling;
  auto executor = std::make_unique<LlamaExecutor>(llama);
  rig->executor = executor.get();
  rig->executor->SetCompletionObserver(
      [outputs = &rig->outputs](const std::string& id, const std::vector<std::int32_t>& tokens) {
        (*outputs)[id] = tokens;
      });
  rig->engine = std::make_unique<ContinuousEngine>(config, std::move(executor));
  return rig;
}

EngineRequest Request(const std::string& id, const std::size_t prompt, const std::size_t output,
                      const std::string& prefix = {}, const std::size_t prefix_tokens = 0) {
  EngineRequest request;
  request.request_id = id;
  request.model_revision = "m";
  request.tokenizer_revision = "t";
  request.prompt_hash = "hash-" + id;
  request.prompt_tokens = prompt;
  request.max_output_tokens = output;
  request.prompt_block_hashes = PromptBlockHashes(request.prompt_hash, prefix, prefix_tokens,
                                                  prompt, 16);
  return request;
}

EngineConfig SmallConfig() {
  EngineConfig config;
  config.kv_blocks = 64;
  config.max_batch_sequences = 4;
  config.max_batch_tokens = 512;
  config.max_prefill_sequences_per_iteration = 1;
  config.prefix_cache_entries = 64;
  config.result_cache_entries = 1;
  return config;
}

void ExpectAllKvReleased(Rig& rig) {
  rig.engine->ClearCaches();
  EXPECT_EQ(rig.engine->kv_cache().free_blocks(), rig.engine->kv_cache().total_blocks());
  EXPECT_EQ(rig.executor->running_sequences(), 0U);
  EXPECT_EQ(rig.executor->parked_blocks(), 0U);
  EXPECT_EQ(rig.executor->tracked_blocks(), 0U);
  EXPECT_TRUE(rig.executor->KvEmpty()) << "llama.cpp still holds KV cells";
}

TEST(LlamaExecutor, RunsScheduledBatchesAndReleasesAllKv) {
  REQUIRE_MODEL();
  auto rig = MakeRig(SmallConfig());
  std::size_t prompt_total = 0;
  const std::vector<std::pair<std::size_t, std::size_t>> shapes = {
      {10, 5}, {37, 20}, {64, 1}, {5, 30}, {90, 12}, {48, 16}};
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    rig->engine->Submit(Request("r" + std::to_string(index), shapes[index].first,
                                shapes[index].second));
    prompt_total += shapes[index].first;
  }
  rig->engine->RunUntilIdle();

  const auto results = rig->engine->TakeResults();
  ASSERT_EQ(results.size(), shapes.size());
  for (const auto& result : results) {
    EXPECT_TRUE(result.succeeded) << result.failure_reason;
    EXPECT_FALSE(result.output.empty());
  }
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    EXPECT_EQ(rig->outputs.at("r" + std::to_string(index)).size(), shapes[index].second);
  }
  EXPECT_EQ(rig->executor->stats().prompt_tokens_computed, prompt_total);
  EXPECT_GT(rig->engine->stats().mean_batch_occupancy(), 1.0);  // Real batching happened.
  EXPECT_TRUE(rig->executor->measured());
  EXPECT_GT(rig->engine->stats().simulated_busy_time.count(), 0);
  ExpectAllKvReleased(*rig);
}

TEST(LlamaExecutor, PrefixReuseSharesRealKvAndMatchesRecompute) {
  REQUIRE_MODEL();
  // Engine 1: A computes the shared 48-token prefix; B reuses its KV.
  auto shared = MakeRig(SmallConfig());
  shared->engine->Submit(Request("a", 60, 12, "sys", 48));
  shared->engine->RunUntilIdle();
  // A left the batch; the radix cache still holds its blocks, so their cells
  // were parked in holder sequences instead of being freed.
  EXPECT_GE(shared->executor->stats().blocks_parked, 3U);
  shared->engine->Submit(Request("b", 70, 12, "sys", 48));
  shared->engine->RunUntilIdle();
  EXPECT_EQ(shared->executor->stats().prompt_tokens_copied, 48U);
  EXPECT_EQ(shared->engine->stats().prefill_tokens_reused, 48U);
  EXPECT_EQ(shared->executor->stats().prompt_tokens_computed, 60U + (70U - 48U));

  // Engine 2: B alone computes everything. Same prompt tokens, so the reused
  // KV must produce the same greedy continuation.
  auto alone = MakeRig(SmallConfig());
  alone->engine->Submit(Request("b", 70, 12, "sys", 48));
  alone->engine->RunUntilIdle();
  EXPECT_EQ(alone->executor->stats().prompt_tokens_copied, 0U);
  EXPECT_EQ(shared->outputs.at("b"), alone->outputs.at("b"));
  ExpectAllKvReleased(*shared);
  ExpectAllKvReleased(*alone);
}

TEST(LlamaExecutor, BackendSamplingPicksTheSameGreedyTokens) {
  REQUIRE_MODEL();
  auto host = MakeRig(SmallConfig(), 8, false);
  auto device = MakeRig(SmallConfig(), 8, true);
  for (auto* rig : {host.get(), device.get()}) {
    rig->engine->Submit(Request("x", 40, 24));
    rig->engine->Submit(Request("y", 70, 16));
    rig->engine->RunUntilIdle();
  }
  EXPECT_EQ(host->outputs.at("x"), device->outputs.at("x"));
  EXPECT_EQ(host->outputs.at("y"), device->outputs.at("y"));
  ExpectAllKvReleased(*device);
}

TEST(LlamaExecutor, PreemptionUnderKvPressureStillCompletesEveryToken) {
  REQUIRE_MODEL();
  EngineConfig config = SmallConfig();
  config.kv_blocks = 12;  // Each request needs 7 blocks; three cannot coexist.
  config.max_prefill_sequences_per_iteration = 3;
  auto rig = MakeRig(config, 0);
  for (int index = 0; index < 3; ++index) {
    rig->engine->Submit(Request("p" + std::to_string(index), 40, 60));
  }
  rig->engine->RunUntilIdle();
  EXPECT_GT(rig->engine->stats().preemptions, 0U);
  EXPECT_GT(rig->engine->stats().wasted_decode_tokens, 0U);
  for (int index = 0; index < 3; ++index) {
    EXPECT_EQ(rig->outputs.at("p" + std::to_string(index)).size(), 60U);
  }
  ExpectAllKvReleased(*rig);
}

TEST(LlamaExecutor, SameIterationSharerWaitsOneStepThenReusesKv) {
  REQUIRE_MODEL();
  EngineConfig config = SmallConfig();
  config.max_prefill_sequences_per_iteration = 2;
  auto rig = MakeRig(config);
  rig->engine->Submit(Request("first", 64, 8, "sys", 48));
  rig->engine->Submit(Request("second", 64, 8, "sys", 48));
  rig->engine->RunUntilIdle();
  // Both arrive together. The second's prefix is computed by the first's
  // admission step, so the engine admits it one iteration later and it shares
  // that KV, rather than recomputing it into cells the scheduler never counted.
  EXPECT_EQ(rig->executor->stats().reuse_not_ready, 0U);
  EXPECT_EQ(rig->executor->stats().prompt_tokens_copied, 48U);
  EXPECT_EQ(rig->engine->stats().prefill_tokens_reused, 48U);
  EXPECT_EQ(rig->outputs.at("first").size(), 8U);
  EXPECT_EQ(rig->outputs.at("second").size(), 8U);
  ExpectAllKvReleased(*rig);
}

TEST(LlamaExecutor, CancellationFreesARunningSequence) {
  REQUIRE_MODEL();
  auto rig = MakeRig(SmallConfig());
  rig->engine->Submit(Request("long", 32, 200));
  for (int step = 0; step < 5; ++step) {
    static_cast<void>(rig->engine->RunOneIteration());
  }
  ASSERT_EQ(rig->executor->running_sequences(), 1U);
  EXPECT_TRUE(rig->engine->Cancel("long", "client_cancelled"));
  EXPECT_EQ(rig->executor->running_sequences(), 0U);
  rig->engine->Submit(Request("after", 20, 4));
  rig->engine->RunUntilIdle();
  EXPECT_EQ(rig->outputs.at("after").size(), 4U);
  ExpectAllKvReleased(*rig);
}

TEST(LlamaExecutor, PrefixCacheEvictionDropsParkedKv) {
  REQUIRE_MODEL();
  EngineConfig config = SmallConfig();
  config.prefix_cache_entries = 6;  // Radix nodes; forces evictions.
  auto rig = MakeRig(config, 4);
  for (int index = 0; index < 12; ++index) {
    rig->engine->Submit(Request("e" + std::to_string(index), 50, 4,
                                "sys" + std::to_string(index % 3), 32));
    rig->engine->RunUntilIdle();
    // Parked blocks can never outnumber what the radix cache still holds.
    EXPECT_LE(rig->executor->parked_blocks(), config.prefix_cache_entries);
  }
  EXPECT_GT(rig->executor->stats().blocks_parked, 0U);
  ExpectAllKvReleased(*rig);
}

}  // namespace
}  // namespace infersched::engine
