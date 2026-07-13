#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "infersched/engine/paged_kv_cache.hpp"

namespace infersched::engine {

// One node per cached prompt block. Unlike the P1/P2 materialized-prefix map,
// a P-block prompt requires O(P) nodes and block-id storage rather than O(P^2).
class RadixPrefixCache {
 public:
  struct Match {
    std::vector<PagedKvCache::BlockId> blocks;
    std::size_t matched_block_count{};
  };

  RadixPrefixCache(std::size_t node_capacity, PagedKvCache& kv_cache);
  ~RadixPrefixCache();
  RadixPrefixCache(const RadixPrefixCache&) = delete;
  RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;

  [[nodiscard]] std::optional<Match> GetLongest(
      std::string_view cache_namespace,
      std::span<const std::string> prompt_block_hashes);
  [[nodiscard]] std::size_t PutBlockPrefixes(
      std::string_view cache_namespace,
      std::span<const std::string> prompt_block_hashes,
      std::span<const PagedKvCache::BlockId> blocks);
  [[nodiscard]] bool EvictOldest() noexcept;
  void Clear() noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return node_count_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  struct Node {
    PagedKvCache::BlockId block{};
    std::string pin_key;
    std::uint64_t last_access{};
    std::unordered_map<std::string, std::unique_ptr<Node>> children;
  };
  struct Root {
    std::unordered_map<std::string, std::unique_ptr<Node>> children;
  };
  struct LeafCandidate {
    std::unordered_map<std::string, std::unique_ptr<Node>>* parent{};
    std::string key;
    std::uint64_t last_access{};
  };

  void FindOldestLeaf(
      std::unordered_map<std::string, std::unique_ptr<Node>>& children,
      std::optional<LeafCandidate>& candidate) noexcept;
  void UnpinTree(
      std::unordered_map<std::string, std::unique_ptr<Node>>& children) noexcept;

  std::size_t capacity_;
  PagedKvCache& kv_cache_;
  std::unordered_map<std::string, Root> roots_;
  std::size_t node_count_{};
  std::uint64_t access_clock_{};
  std::uint64_t next_pin_id_{};
};

}  // namespace infersched::engine
