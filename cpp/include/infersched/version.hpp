#pragma once

#include <string_view>

namespace infersched {

// Semantic version of the InferSched runtime.
inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 7;  // Phase 7 — control plane and measurement.
inline constexpr int kVersionPatch = 0;

// Human-readable version string, e.g. "0.0.0".
std::string_view Version() noexcept;

}  // namespace infersched
