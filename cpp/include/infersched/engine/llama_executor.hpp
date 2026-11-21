#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "infersched/engine/step_executor.hpp"

struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_sampler;

namespace infersched::engine {

struct LlamaExecutorConfig {
  std::string model_path;
  int gpu_layers{999};              // Layers offloaded to the GPU; 0 = CPU only.
  std::size_t kv_tokens{4096};      // KV cells: EngineConfig kv_blocks * block size.
  std::size_t block_size_tokens{16};
  std::size_t max_sequences{8};     // Running sequences: EngineConfig max_batch_sequences.
  std::size_t holder_sequences{0};  // Extra sequence ids that keep prefix-cache KV alive.
  std::size_t max_batch_tokens{4096};
  std::size_t ubatch_tokens{512};
  bool flash_attention{true};
  int threads{4};
  // Greedy-sample on the device (llama.cpp's backend sampling) instead of
  // copying every sequence's logits to the host and taking the argmax there.
  bool backend_sampling{false};
};

struct LlamaExecutorStats {
  std::size_t decode_calls{};
  std::size_t batch_tokens{};            // Tokens submitted to llama_decode.
  std::size_t prompt_tokens_computed{};  // Prompt tokens run through the model.
  std::size_t prompt_tokens_copied{};    // Prompt tokens whose KV was shared via seq_cp.
  std::size_t blocks_parked{};           // Prefix blocks moved into holder sequences.
  std::size_t blocks_dropped{};          // Prefix blocks whose KV could not be kept.
  std::size_t reuse_not_ready{};         // Matches whose KV was still being computed.
  std::chrono::nanoseconds model_time{};   // Inside llama_decode and logits readback.
  std::chrono::nanoseconds sample_time{};  // Greedy argmax on the host.
};

// One llama_decode call, for calibrating the cost model against hardware.
struct LlamaStepRecord {
  std::size_t prompt_tokens{};   // Prompt tokens computed in the step.
  std::size_t decode_tokens{};   // Sequences that decoded one token.
  std::size_t context_tokens{};  // Sum of KV positions the decode tokens attend to.
  std::chrono::nanoseconds duration{};
};

// Runs a GGUF model with llama.cpp on exactly the batches ContinuousEngine
// schedules. Every scheduler iteration becomes one llama_decode call that
// carries the prompt tokens of newly admitted sequences and one token for
// each running sequence (prefill and decode fused, as llama-server and vLLM's
// chunked prefill do). Sampling is greedy and EOS is ignored, so every request
// produces exactly max_output_tokens tokens (the vLLM benchmark --ignore-eos
// convention), which keeps runs comparable across schedulers.
//
// KV cache: one unified llama.cpp cache of kv_tokens cells. The scheduler's
// PagedKvCache decides admission and preemption, so llama.cpp never runs out
// of cells. Prefix-cache hits share KV cells through llama_memory_seq_cp
// (metadata only, no copy). When the sequence that computed a cached block
// leaves the batch, the block is parked in a "holder" sequence so the KV
// survives until the radix prefix cache evicts it.
//
// Prompts are synthetic token ids derived deterministically from the
// request's block hashes, so requests that share a prefix_id share identical
// prefix tokens (and hence reusable KV), while every other prompt differs.
class LlamaExecutor final : public StepExecutor {
 public:
  explicit LlamaExecutor(LlamaExecutorConfig config);
  ~LlamaExecutor() override;

  [[nodiscard]] PrefillResult Prefill(const PrefillWork& work) override;
  [[nodiscard]] std::chrono::nanoseconds Decode(
      std::size_t batch_size, std::span<const std::string_view> sequences) override;
  void Release(const std::string& request_id,
               std::span<const PagedKvCache::BlockId> blocks,
               const PagedKvCache& kv) override;
  void OnBlockFreed(PagedKvCache::BlockId block) noexcept override;
  [[nodiscard]] std::string Output(const EngineRequest& request) override;
  [[nodiscard]] bool tracks_sequences() const noexcept override { return true; }
  [[nodiscard]] bool measured() const noexcept override { return true; }
  [[nodiscard]] bool shares_only_computed_kv() const noexcept override { return true; }
  [[nodiscard]] std::string_view name() const noexcept override { return "llama.cpp"; }

  // Prompt token ids the executor feeds for `request`.
  [[nodiscard]] std::vector<std::int32_t> PromptTokens(const EngineRequest& request) const;
  // Called with each completed sequence's generated token ids (for tests and
  // for dumping a workload's reference outputs).
  void SetCompletionObserver(
      std::function<void(const std::string&, const std::vector<std::int32_t>&)> observer) {
    completion_observer_ = std::move(observer);
  }

  [[nodiscard]] const LlamaExecutorStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const LlamaStepRecord& last_step() const noexcept { return last_step_; }
  [[nodiscard]] std::size_t running_sequences() const noexcept { return sequences_.size(); }
  [[nodiscard]] std::size_t parked_blocks() const noexcept;
  [[nodiscard]] std::size_t tracked_blocks() const noexcept { return blocks_.size(); }
  // True when no llama.cpp sequence (running or holder) holds any KV cell.
  [[nodiscard]] bool KvEmpty() const;
  [[nodiscard]] std::int32_t vocab_size() const noexcept { return vocab_size_; }
  [[nodiscard]] std::string model_description() const;
  [[nodiscard]] std::uint64_t model_bytes() const;
  [[nodiscard]] std::uint64_t model_parameters() const;
  [[nodiscard]] const LlamaExecutorConfig& config() const noexcept { return config_; }

 private:
  struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view value) const noexcept {
      return std::hash<std::string_view>{}(value);
    }
  };
  struct Sequence {
    std::int32_t id{};
    std::vector<std::int32_t> prompt;
    std::vector<std::int32_t> generated;
    std::size_t compute_from{};  // First prompt position the model must compute.
    bool prompt_pending{};       // Prompt not yet submitted to llama_decode.
    std::vector<PagedKvCache::BlockId> prompt_blocks;
  };
  // Where a cacheable prompt block's KV lives.
  struct BlockKv {
    std::int32_t source{-1};  // Sequence whose cells hold the block; -1 = dropped.
    std::uint32_t index{};    // Block index within the prompt.
    std::uint32_t valid{};    // Prompt tokens in the block (< block size for a tail).
    std::vector<std::int32_t> copies;  // Running sequences sharing all its cells.
  };
  static constexpr PagedKvCache::BlockId kNoBlock = ~PagedKvCache::BlockId{0};

  [[nodiscard]] std::int32_t FindHolder(std::uint32_t index) const noexcept;
  // True when `sequence`'s cells hold computed KV (holders always do; a
  // running sequence only after its prompt went through llama_decode).
  [[nodiscard]] bool Computed(std::int32_t sequence) const noexcept;
  [[nodiscard]] bool IsHolder(std::int32_t sequence) const noexcept {
    return sequence >= static_cast<std::int32_t>(config_.max_sequences);
  }

  LlamaExecutorConfig config_;
  llama_model* model_{};
  llama_context* context_{};
  const llama_vocab* vocab_{};
  std::int32_t vocab_size_{};
  std::int32_t token_low_{};
  std::int32_t token_high_{};
  void* batch_{};  // llama_batch, kept opaque to avoid exposing llama.h.
  std::size_t batch_capacity_{};
  std::unordered_map<std::string, Sequence, StringHash, std::equal_to<>> sequences_;
  std::unordered_map<PagedKvCache::BlockId, BlockKv> blocks_;
  std::vector<std::int32_t> free_running_;
  std::vector<Sequence*> running_by_id_;  // Indexed by running sequence id.
  // holder_slots_[h][i]: block parked in holder h at prompt block index i.
  std::vector<std::vector<PagedKvCache::BlockId>> holder_slots_;
  std::vector<std::pair<Sequence*, std::int32_t>> outputs_;
  std::vector<llama_sampler*> samplers_;  // One greedy chain per running id.
  LlamaExecutorStats stats_;
  LlamaStepRecord last_step_;
  std::function<void(const std::string&, const std::vector<std::int32_t>&)>
      completion_observer_;
};

}  // namespace infersched::engine
