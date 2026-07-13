#include "infersched/engine/result_cache_key.hpp"

#include <string>

namespace infersched::engine {
namespace {

void AppendPart(std::string& output, const std::string_view value) {
  output += std::to_string(value.size());
  output.push_back(':');
  output.append(value);
  output.push_back('|');
}

}  // namespace

std::string BuildResultCacheKey(const ResultCacheKeyParts& parts) {
  std::string key;
  AppendPart(key, parts.model_revision);
  AppendPart(key, parts.tokenizer_revision);
  AppendPart(key, parts.prompt_hash);
  key += std::to_string(parts.max_output_tokens) + '|';
  key += std::to_string(parts.temperature_micros) + '|';
  key += std::to_string(parts.top_p_micros) + '|';
  key += std::to_string(parts.seed);
  return key;
}

}  // namespace infersched::engine
