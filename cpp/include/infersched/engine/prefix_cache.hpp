#pragma once

#include <cstddef>
#include <list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "infersched/engine/paged_kv_cache.hpp"

namespace infersched::engine {

class PrefixCache {
 public:
  struct Match {
    std::vector<PagedKvCache::BlockId> blocks;
    std::size_t matched_block_count{};
  };

  PrefixCache(std::size_t capacity, PagedKvCache& kv_cache);
  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;

  [[nodiscard]] bool Put(std::string key,
                         std::span<const PagedKvCache::BlockId> blocks);
  [[nodiscard]] std::optional<std::vector<PagedKvCache::BlockId>> Get(
      std::string_view key);
  [[nodiscard]] std::optional<Match> GetLongest(
      std::string_view cache_namespace,
      std::span<const std::string> prompt_block_hashes);
  [[nodiscard]] std::size_t PutBlockPrefixes(
      std::string_view cache_namespace,
      std::span<const std::string> prompt_block_hashes,
      std::span<const PagedKvCache::BlockId> blocks);
  [[nodiscard]] bool EvictOldest() noexcept;
  void Clear() noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
  [[nodiscard]] std::size_t misses() const noexcept { return misses_; }

 private:
  struct Entry {
    std::vector<PagedKvCache::BlockId> blocks;
    std::list<std::string>::iterator recency;
  };

  std::size_t capacity_;
  PagedKvCache& kv_cache_;
  std::list<std::string> recency_;
  std::unordered_map<std::string, Entry> entries_;
  std::size_t hits_{};
  std::size_t misses_{};

  [[nodiscard]] static std::string BlockPrefixKey(
      std::string_view cache_namespace,
      std::span<const std::string> prompt_block_hashes);
};

}  // namespace infersched::engine
