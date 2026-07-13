#include "infersched/engine/prefix_cache.hpp"

#include <stdexcept>
#include <utility>

namespace infersched::engine {

PrefixCache::PrefixCache(const std::size_t capacity, PagedKvCache& kv_cache)
    : capacity_(capacity), kv_cache_(kv_cache) {
  if (capacity == 0) {
    throw std::invalid_argument("prefix cache capacity must be positive");
  }
}

PrefixCache::~PrefixCache() { Clear(); }

bool PrefixCache::Put(
    std::string key, const std::span<const PagedKvCache::BlockId> blocks) {
  const auto existing = entries_.find(key);
  if (existing != entries_.end()) {
    recency_.splice(recency_.begin(), recency_, existing->second.recency);
    return existing->second.blocks ==
           std::vector<PagedKvCache::BlockId>(blocks.begin(), blocks.end());
  }
  if (!kv_cache_.PinPrefix(key, blocks)) {
    return false;
  }
  if (entries_.size() == capacity_) {
    const std::string evicted = recency_.back();
    static_cast<void>(kv_cache_.UnpinPrefix(evicted));
    entries_.erase(evicted);
    recency_.pop_back();
  }
  recency_.push_front(std::move(key));
  entries_.emplace(recency_.front(),
                   Entry{std::vector<PagedKvCache::BlockId>(blocks.begin(), blocks.end()),
                         recency_.begin()});
  return true;
}

std::optional<std::vector<PagedKvCache::BlockId>> PrefixCache::Get(
    const std::string_view key) {
  const auto iterator = entries_.find(std::string(key));
  if (iterator == entries_.end()) {
    ++misses_;
    return std::nullopt;
  }
  ++hits_;
  recency_.splice(recency_.begin(), recency_, iterator->second.recency);
  return iterator->second.blocks;
}

std::optional<PrefixCache::Match> PrefixCache::GetLongest(
    const std::string_view cache_namespace,
    const std::span<const std::string> prompt_block_hashes) {
  for (std::size_t count = prompt_block_hashes.size(); count > 0; --count) {
    const std::string key =
        BlockPrefixKey(cache_namespace, prompt_block_hashes.first(count));
    const auto iterator = entries_.find(key);
    if (iterator == entries_.end()) {
      continue;
    }
    ++hits_;
    recency_.splice(recency_.begin(), recency_, iterator->second.recency);
    return Match{.blocks = iterator->second.blocks,
                 .matched_block_count = count};
  }
  ++misses_;
  return std::nullopt;
}

std::size_t PrefixCache::PutBlockPrefixes(
    const std::string_view cache_namespace,
    const std::span<const std::string> prompt_block_hashes,
    const std::span<const PagedKvCache::BlockId> blocks) {
  const std::size_t count = std::min(prompt_block_hashes.size(), blocks.size());
  std::size_t inserted = 0;
  for (std::size_t length = 1; length <= count; ++length) {
    if (Put(BlockPrefixKey(cache_namespace, prompt_block_hashes.first(length)),
            blocks.first(length))) {
      ++inserted;
    }
  }
  return inserted;
}

bool PrefixCache::EvictOldest() noexcept {
  if (recency_.empty()) {
    return false;
  }
  const std::string evicted = recency_.back();
  static_cast<void>(kv_cache_.UnpinPrefix(evicted));
  entries_.erase(evicted);
  recency_.pop_back();
  return true;
}

void PrefixCache::Clear() noexcept {
  for (const auto& key : recency_) {
    static_cast<void>(kv_cache_.UnpinPrefix(key));
  }
  entries_.clear();
  recency_.clear();
}

std::string PrefixCache::BlockPrefixKey(
    const std::string_view cache_namespace,
    const std::span<const std::string> prompt_block_hashes) {
  std::string key{"blocks|"};
  key += std::to_string(cache_namespace.size());
  key.push_back(':');
  key.append(cache_namespace);
  for (const auto& hash : prompt_block_hashes) {
    key.push_back('|');
    key += std::to_string(hash.size());
    key.push_back(':');
    key.append(hash);
  }
  return key;
}

}  // namespace infersched::engine
