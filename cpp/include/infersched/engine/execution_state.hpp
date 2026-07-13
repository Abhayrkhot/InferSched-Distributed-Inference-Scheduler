#pragma once

#include <string_view>

namespace infersched::engine {

enum class ExecutionState {
  kWaitingAdmission,
  kPrefill,
  kDecode,
  kPreempted,
  kCompleted,
  kFailed,
};

[[nodiscard]] bool CanTransition(ExecutionState from,
                                 ExecutionState to) noexcept;
[[nodiscard]] bool IsTerminal(ExecutionState state) noexcept;
[[nodiscard]] std::string_view ToString(ExecutionState state) noexcept;

class ExecutionStateMachine {
 public:
  [[nodiscard]] ExecutionState state() const noexcept { return state_; }
  [[nodiscard]] bool TransitionTo(ExecutionState next) noexcept;

 private:
  ExecutionState state_{ExecutionState::kWaitingAdmission};
};

}  // namespace infersched::engine
