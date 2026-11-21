#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace infersched::engine {

// Block hashes for a simulated prompt: full blocks inside the shared prefix
// hash by (prefix_id, block index), so requests sharing a prefix share KV
// blocks; every other block hashes by the request's own prompt hash.
inline std::vector<std::string> PromptBlockHashes(const std::string_view prompt_hash,
                                                  const std::string_view prefix_id,
                                                  const std::size_t prefix_tokens,
                                                  const std::size_t prompt_tokens,
                                                  const std::size_t block_size) {
  const std::size_t blocks = (prompt_tokens + block_size - 1) / block_size;
  const std::size_t shared_tokens =
      prefix_id.empty() ? 0 : std::min(prefix_tokens, prompt_tokens);
  std::vector<std::string> hashes;
  hashes.reserve(blocks);
  for (std::size_t index = 0; index < blocks; ++index) {
    std::string hash = (index + 1) * block_size <= shared_tokens ? "p:" : "u:";
    hash += (index + 1) * block_size <= shared_tokens ? prefix_id : prompt_hash;
    hash += ':';
    hash += std::to_string(index);
    hashes.push_back(std::move(hash));
  }
  return hashes;
}

}  // namespace infersched::engine
