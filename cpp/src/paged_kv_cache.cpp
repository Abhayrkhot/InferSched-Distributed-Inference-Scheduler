#include "infersched/engine/paged_kv_cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace infersched::engine {

PagedKvCache::PagedKvCache(const std::size_t block_count,
                           const std::size_t block_size_tokens)
    : block_size_tokens_(block_size_tokens), refcounts_(block_count, 0) {
  if (block_count > std::numeric_limits<BlockId>::max()) {
    throw std::invalid_argument("block_count exceeds BlockId range");
  }
  if (block_size_tokens == 0) {
    throw std::invalid_argument("block_size_tokens must be positive");
  }
  free_list_.reserve(block_count);
  for (std::size_t index = block_count; index > 0; --index) {
    free_list_.push_back(static_cast<BlockId>(index - 1));
  }
}

PagedKvCache::AllocationStatus PagedKvCache::Allocate(
    std::string sequence_id, const std::size_t required_blocks,
    const std::span<const BlockId> shared_prefix_blocks) {
  if (sequence_blocks_.contains(sequence_id)) {
    return AllocationStatus::kSequenceAlreadyExists;
  }
  if (shared_prefix_blocks.size() > required_blocks) {
    return AllocationStatus::kInvalidSharedBlock;
  }

  std::unordered_set<BlockId> unique_shared;
  unique_shared.reserve(shared_prefix_blocks.size());
  for (const BlockId block : shared_prefix_blocks) {
    if (block >= refcounts_.size() || refcounts_[block] == 0 ||
        !unique_shared.insert(block).second) {
      return AllocationStatus::kInvalidSharedBlock;
    }
  }

  const std::size_t new_block_count = required_blocks - shared_prefix_blocks.size();
  if (free_list_.size() < new_block_count) {
    return AllocationStatus::kInsufficientBlocks;
  }

  std::vector<BlockId> blocks;
  blocks.reserve(required_blocks);
  blocks.insert(blocks.end(), shared_prefix_blocks.begin(), shared_prefix_blocks.end());
  for (const BlockId block : shared_prefix_blocks) {
    ++refcounts_[block];
  }
  for (std::size_t count = 0; count < new_block_count; ++count) {
    const BlockId block = free_list_.back();
    free_list_.pop_back();
    refcounts_[block] = 1;
    blocks.push_back(block);
  }
  sequence_blocks_.emplace(std::move(sequence_id), std::move(blocks));
  return AllocationStatus::kAllocated;
}

bool PagedKvCache::Free(const std::string_view sequence_id) noexcept {
  const auto iterator = sequence_blocks_.find(std::string(sequence_id));
  if (iterator == sequence_blocks_.end()) {
    return false;
  }
  for (const BlockId block : iterator->second) {
    --refcounts_[block];
    if (refcounts_[block] == 0) {
      free_list_.push_back(block);
    }
  }
  sequence_blocks_.erase(iterator);
  return true;
}

PagedKvCache::AllocationStatus PagedKvCache::AppendBlocks(
    const std::string_view sequence_id, const std::size_t additional_blocks) {
  const auto iterator = sequence_blocks_.find(std::string(sequence_id));
  if (iterator == sequence_blocks_.end()) {
    return AllocationStatus::kInvalidSharedBlock;
  }
  if (free_list_.size() < additional_blocks) {
    return AllocationStatus::kInsufficientBlocks;
  }
  iterator->second.reserve(iterator->second.size() + additional_blocks);
  for (std::size_t count = 0; count < additional_blocks; ++count) {
    const BlockId block = free_list_.back();
    free_list_.pop_back();
    refcounts_[block] = 1;
    iterator->second.push_back(block);
  }
  return AllocationStatus::kAllocated;
}

bool PagedKvCache::PinPrefix(const std::string prefix_key,
                             const std::span<const BlockId> blocks) {
  if (prefix_blocks_.contains(prefix_key) || blocks.empty()) {
    return false;
  }
  std::unordered_set<BlockId> unique;
  unique.reserve(blocks.size());
  for (const BlockId block : blocks) {
    if (block >= refcounts_.size() || refcounts_[block] == 0 ||
        !unique.insert(block).second) {
      return false;
    }
  }
  for (const BlockId block : blocks) {
    ++refcounts_[block];
  }
  prefix_blocks_.emplace(std::move(prefix_key),
                         std::vector<BlockId>(blocks.begin(), blocks.end()));
  return true;
}

bool PagedKvCache::UnpinPrefix(const std::string_view prefix_key) noexcept {
  const auto iterator = prefix_blocks_.find(std::string(prefix_key));
  if (iterator == prefix_blocks_.end()) {
    return false;
  }
  for (const BlockId block : iterator->second) {
    --refcounts_[block];
    if (refcounts_[block] == 0) {
      free_list_.push_back(block);
    }
  }
  prefix_blocks_.erase(iterator);
  return true;
}

std::size_t PagedKvCache::total_blocks() const noexcept { return refcounts_.size(); }
std::size_t PagedKvCache::free_blocks() const noexcept { return free_list_.size(); }
std::size_t PagedKvCache::allocated_blocks() const noexcept {
  return total_blocks() - free_blocks();
}
std::size_t PagedKvCache::block_size_tokens() const noexcept {
  return block_size_tokens_;
}
std::size_t PagedKvCache::sequence_count() const noexcept {
  return sequence_blocks_.size();
}

std::span<const PagedKvCache::BlockId> PagedKvCache::BlocksFor(
    const std::string_view sequence_id) const noexcept {
  const auto iterator = sequence_blocks_.find(std::string(sequence_id));
  return iterator == sequence_blocks_.end()
             ? std::span<const BlockId>{}
             : std::span<const BlockId>{iterator->second};
}

std::uint32_t PagedKvCache::RefCount(const BlockId block) const noexcept {
  return block < refcounts_.size() ? refcounts_[block] : 0;
}

bool PagedKvCache::CheckInvariants() const noexcept {
  if (allocated_blocks() + free_blocks() != total_blocks()) {
    return false;
  }

  std::vector<std::uint32_t> observed_refs(total_blocks(), 0);
  for (const auto& [sequence_id, blocks] : sequence_blocks_) {
    static_cast<void>(sequence_id);
    std::unordered_set<BlockId> unique;
    for (const BlockId block : blocks) {
      if (block >= total_blocks() || !unique.insert(block).second) {
        return false;
      }
      ++observed_refs[block];
    }
  }
  for (const auto& [prefix_key, blocks] : prefix_blocks_) {
    static_cast<void>(prefix_key);
    std::unordered_set<BlockId> unique;
    for (const BlockId block : blocks) {
      if (block >= total_blocks() || !unique.insert(block).second) {
        return false;
      }
      ++observed_refs[block];
    }
  }

  std::vector<bool> observed_free(total_blocks(), false);
  for (const BlockId block : free_list_) {
    if (block >= total_blocks() || observed_free[block]) {
      return false;
    }
    observed_free[block] = true;
  }

  for (std::size_t block = 0; block < total_blocks(); ++block) {
    if (observed_refs[block] != refcounts_[block]) {
      return false;
    }
    if ((refcounts_[block] == 0) != observed_free[block]) {
      return false;
    }
  }
  return true;
}

}  // namespace infersched::engine
