#include "infersched/version.hpp"

namespace infersched {

std::string_view Version() noexcept {
  // Kept in sync with the constants in version.hpp; stringified here so the
  // literal has static storage duration and a stable string_view is safe.
  return "0.7.0";
}

}  // namespace infersched
