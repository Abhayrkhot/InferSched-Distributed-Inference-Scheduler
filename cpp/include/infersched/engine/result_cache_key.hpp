#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace infersched::engine {

struct ResultCacheKeyParts {
  std::string_view model_revision;
  std::string_view tokenizer_revision;
  std::string_view prompt_hash;
  std::uint64_t max_output_tokens{};
  std::uint64_t temperature_micros{};
  std::uint64_t top_p_micros{};
  std::uint64_t seed{};
};

[[nodiscard]] std::string BuildResultCacheKey(const ResultCacheKeyParts& parts);

}  // namespace infersched::engine
