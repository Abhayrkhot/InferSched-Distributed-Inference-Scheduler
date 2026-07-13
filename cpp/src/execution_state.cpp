#include "infersched/engine/execution_state.hpp"

namespace infersched::engine {

bool CanTransition(const ExecutionState from, const ExecutionState to) noexcept {
  if (from == ExecutionState::kCompleted || from == ExecutionState::kFailed) {
    return false;
  }
  if (to == ExecutionState::kFailed) {
    return true;
  }

  switch (from) {
    case ExecutionState::kWaitingAdmission:
      return to == ExecutionState::kPrefill;
    case ExecutionState::kPrefill:
      return to == ExecutionState::kDecode || to == ExecutionState::kPreempted;
    case ExecutionState::kDecode:
      return to == ExecutionState::kCompleted ||
             to == ExecutionState::kPreempted;
    case ExecutionState::kPreempted:
      return to == ExecutionState::kWaitingAdmission;
    case ExecutionState::kCompleted:
    case ExecutionState::kFailed:
      return false;
  }
  return false;
}

bool IsTerminal(const ExecutionState state) noexcept {
  return state == ExecutionState::kCompleted || state == ExecutionState::kFailed;
}

std::string_view ToString(const ExecutionState state) noexcept {
  switch (state) {
    case ExecutionState::kWaitingAdmission:
      return "WAITING_ADMISSION";
    case ExecutionState::kPrefill:
      return "PREFILL";
    case ExecutionState::kDecode:
      return "DECODE";
    case ExecutionState::kPreempted:
      return "PREEMPTED";
    case ExecutionState::kCompleted:
      return "COMPLETED";
    case ExecutionState::kFailed:
      return "FAILED";
  }
  return "UNKNOWN";
}

bool ExecutionStateMachine::TransitionTo(const ExecutionState next) noexcept {
  if (!CanTransition(state_, next)) {
    return false;
  }
  state_ = next;
  return true;
}

}  // namespace infersched::engine
