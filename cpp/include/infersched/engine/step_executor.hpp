#pragma once

#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "infersched/engine/cost_model.hpp"
#include "infersched/engine/paged_kv_cache.hpp"

namespace infersched::engine {

struct EngineRequest;

// The scheduler decides *what* runs each iteration (admission, batch
// membership, preemption, KV block placement, prefix reuse). A StepExecutor
// performs that work. The default executor charges the phenomenological cost
// model; the llama.cpp executor runs a real model on the same decisions.

// One admitted sequence's prompt work.
struct PrefillWork {
  const EngineRequest* request{};
  // Every KV block the sequence holds, in position order.
  std::span<const PagedKvCache::BlockId> blocks;
  // Leading blocks the radix prefix cache matched (their KV already exists).
  std::size_t matched_blocks{};
  // Prompt tokens the prefix cache says are already computed.
  std::size_t cached_tokens{};
};

struct PrefillResult {
  // Compute charged at admission time. A fused executor returns zero here and
  // charges the prompt work inside the next Decode call.
  std::chrono::nanoseconds compute{};
  // Prompt tokens actually reused. At most cached_tokens; lower when the
  // executor must recompute (a real model needs the last prompt token's
  // logits, and a cached block whose KV was dropped is recomputed).
  std::size_t reused_tokens{};
};

class StepExecutor {
 public:
  StepExecutor() = default;
  StepExecutor(const StepExecutor&) = delete;
  StepExecutor& operator=(const StepExecutor&) = delete;
  virtual ~StepExecutor() = default;

  [[nodiscard]] virtual PrefillResult Prefill(const PrefillWork& work) = 0;

  // One iteration in which each of the `batch_size` active sequences produces
  // one output token. `sequences` lists their request ids when
  // tracks_sequences() is true and is empty otherwise.
  [[nodiscard]] virtual std::chrono::nanoseconds Decode(
      std::size_t batch_size, std::span<const std::string_view> sequences) = 0;

  // The sequence left the batch (completed, cancelled or preempted) after the
  // PagedKvCache released its blocks. `blocks` are the blocks it held; any
  // that `kv` still references (prefix cache, other sequences) must keep
  // their KV. Only called when tracks_sequences() is true.
  virtual void Release(const std::string& request_id,
                       std::span<const PagedKvCache::BlockId> blocks,
                       const PagedKvCache& kv) {
    static_cast<void>(request_id);
    static_cast<void>(blocks);
    static_cast<void>(kv);
  }
  // A block's reference count reached zero. Only called when
  // tracks_sequences() is true.
  virtual void OnBlockFreed(PagedKvCache::BlockId block) noexcept {
    static_cast<void>(block);
  }

  // Output text of a sequence that just completed.
  [[nodiscard]] virtual std::string Output(const EngineRequest& request) = 0;

  // True when the executor keeps per-sequence state (ids, block lifetimes).
  [[nodiscard]] virtual bool tracks_sequences() const noexcept { return false; }
  // True when a prefix match may only use KV that has actually been
  // computed. The engine then defers a request whose match includes blocks
  // admitted in the same iteration (they are computed by that iteration's
  // step) instead of letting it share KV that does not exist yet.
  [[nodiscard]] virtual bool shares_only_computed_kv() const noexcept { return false; }
  // True when compute durations are measured wall time, not modeled time.
  [[nodiscard]] virtual bool measured() const noexcept { return false; }
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

// The original behaviour: charge the deterministic cost model.
class CostModelExecutor final : public StepExecutor {
 public:
  CostModelExecutor(CostModelConfig config, std::uint64_t seed)
      : cost_model_(config, seed) {}

  [[nodiscard]] PrefillResult Prefill(const PrefillWork& work) override;
  [[nodiscard]] std::chrono::nanoseconds Decode(
      std::size_t batch_size, std::span<const std::string_view> sequences) override;
  [[nodiscard]] std::string Output(const EngineRequest& request) override;
  [[nodiscard]] std::string_view name() const noexcept override { return "cost-model"; }

 private:
  DeterministicCostModel cost_model_;
};

}  // namespace infersched::engine
