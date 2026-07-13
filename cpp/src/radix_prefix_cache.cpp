#include "infersched/engine/radix_prefix_cache.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace infersched::engine {

RadixPrefixCache::RadixPrefixCache(const std::size_t node_capacity,
                                   PagedKvCache& kv_cache)
    : capacity_(node_capacity), kv_cache_(kv_cache) {
  if (node_capacity == 0) {
    throw std::invalid_argument("radix prefix cache capacity must be positive");
  }
}

RadixPrefixCache::~RadixPrefixCache() { Clear(); }

std::optional<RadixPrefixCache::Match> RadixPrefixCache::GetLongest(
    const std::string_view cache_namespace,
    const std::span<const std::string> prompt_block_hashes) {
  const auto root = roots_.find(std::string(cache_namespace));
  if (root == roots_.end()) {
    return std::nullopt;
  }
  auto* children = &root->second.children;
  Match match;
  match.blocks.reserve(prompt_block_hashes.size());
  for (const auto& hash : prompt_block_hashes) {
    const auto node = children->find(hash);
    if (node == children->end()) {
      break;
    }
    node->second->last_access = ++access_clock_;
    match.blocks.push_back(node->second->block);
    children = &node->second->children;
  }
  match.matched_block_count = match.blocks.size();
  return match.blocks.empty() ? std::nullopt
                              : std::optional<Match>{std::move(match)};
}

std::size_t RadixPrefixCache::PutBlockPrefixes(
    const std::string_view cache_namespace,
    const std::span<const std::string> prompt_block_hashes,
    const std::span<const PagedKvCache::BlockId> blocks) {
  const std::size_t count = std::min(prompt_block_hashes.size(), blocks.size());
  auto* children = &roots_[std::string(cache_namespace)].children;
  std::size_t inserted = 0;
  for (std::size_t index = 0; index < count; ++index) {
    auto existing = children->find(prompt_block_hashes[index]);
    if (existing != children->end()) {
      existing->second->last_access = ++access_clock_;
      children = &existing->second->children;
      continue;
    }

    const std::string pin_key =
        "radix-prefix-" + std::to_string(next_pin_id_++);
    const std::array<PagedKvCache::BlockId, 1> block{blocks[index]};
    if (!kv_cache_.PinPrefix(pin_key, block)) {
      break;
    }
    auto node = std::make_unique<Node>();
    node->block = blocks[index];
    node->pin_key = pin_key;
    node->last_access = ++access_clock_;
    auto [iterator, was_inserted] = children->emplace(
        prompt_block_hashes[index], std::move(node));
    static_cast<void>(was_inserted);
    children = &iterator->second->children;
    ++node_count_;
    ++inserted;
  }
  while (node_count_ > capacity_) {
    if (!EvictOldest()) {
      break;
    }
  }
  return inserted;
}

bool RadixPrefixCache::EvictOldest() noexcept {
  std::optional<LeafCandidate> candidate;
  for (auto& [cache_namespace, root] : roots_) {
    static_cast<void>(cache_namespace);
    FindOldestLeaf(root.children, candidate);
  }
  if (!candidate.has_value()) {
    return false;
  }
  const auto node = candidate->parent->find(candidate->key);
  if (node == candidate->parent->end()) {
    return false;
  }
  static_cast<void>(kv_cache_.UnpinPrefix(node->second->pin_key));
  candidate->parent->erase(node);
  --node_count_;
  return true;
}

void RadixPrefixCache::Clear() noexcept {
  for (auto& [cache_namespace, root] : roots_) {
    static_cast<void>(cache_namespace);
    UnpinTree(root.children);
  }
  roots_.clear();
  node_count_ = 0;
}

void RadixPrefixCache::FindOldestLeaf(
    std::unordered_map<std::string, std::unique_ptr<Node>>& children,
    std::optional<LeafCandidate>& candidate) noexcept {
  for (auto& [key, node] : children) {
    if (!node->children.empty()) {
      FindOldestLeaf(node->children, candidate);
      continue;
    }
    if (!candidate.has_value() || node->last_access < candidate->last_access) {
      candidate = LeafCandidate{.parent = &children,
                                .key = key,
                                .last_access = node->last_access};
    }
  }
}

void RadixPrefixCache::UnpinTree(
    std::unordered_map<std::string, std::unique_ptr<Node>>& children) noexcept {
  for (auto& [key, node] : children) {
    static_cast<void>(key);
    UnpinTree(node->children);
    static_cast<void>(kv_cache_.UnpinPrefix(node->pin_key));
  }
}

}  // namespace infersched::engine
