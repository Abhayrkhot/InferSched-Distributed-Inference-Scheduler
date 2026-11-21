#include "infersched/engine/step_executor.hpp"

#include "infersched/engine/deterministic_engine.hpp"

namespace infersched::engine {

PrefillResult CostModelExecutor::Prefill(const PrefillWork& work) {
  const std::size_t prompt = work.request->prompt_tokens;
  const std::size_t cached = work.cached_tokens < prompt ? work.cached_tokens : prompt;
  return PrefillResult{.compute = cost_model_.Prefill(prompt - cached),
                       .reused_tokens = cached};
}

std::chrono::nanoseconds CostModelExecutor::Decode(
    const std::size_t batch_size, const std::span<const std::string_view> sequences) {
  static_cast<void>(sequences);
  return cost_model_.DecodeStep(batch_size);
}

std::string CostModelExecutor::Output(const EngineRequest& request) {
  return "simulated:" + request.model_revision + ':' + request.prompt_hash + ':' +
         std::to_string(request.max_output_tokens) + ':' +
         std::to_string(request.sampling_seed);
}

}  // namespace infersched::engine
