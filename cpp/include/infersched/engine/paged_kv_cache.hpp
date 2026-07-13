#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infersched::engine {

class PagedKvCache {
 public:
  using BlockId = std::uint32_t;

  enum class AllocationStatus {
    kAllocated,
    kSequenceAlreadyExists,
    kInvalidSharedBlock,
    kInsufficientBlocks,
  };

  PagedKvCache(std::size_t block_count, std::size_t block_size_tokens);

  // required_blocks includes shared prefix blocks. Allocation is all-or-nothing.
  [[nodiscard]] AllocationStatus Allocate(
      std::string sequence_id, std::size_t required_blocks,
      std::span<const BlockId> shared_prefix_blocks = {});
  [[nodiscard]] bool Free(std::string_view sequence_id) noexcept;
  [[nodiscard]] AllocationStatus AppendBlocks(std::string_view sequence_id,
                                               std::size_t additional_blocks);
  [[nodiscard]] bool PinPrefix(std::string prefix_key,
                               std::span<const BlockId> blocks);
  [[nodiscard]] bool UnpinPrefix(std::string_view prefix_key) noexcept;

  [[nodiscard]] std::size_t total_blocks() const noexcept;
  [[nodiscard]] std::size_t free_blocks() const noexcept;
  [[nodiscard]] std::size_t allocated_blocks() const noexcept;
  [[nodiscard]] std::size_t block_size_tokens() const noexcept;
  [[nodiscard]] std::size_t sequence_count() const noexcept;
  [[nodiscard]] std::span<const BlockId> BlocksFor(
      std::string_view sequence_id) const noexcept;
  [[nodiscard]] std::uint32_t RefCount(BlockId block) const noexcept;

  // Expensive by design: tests call this after every randomized operation.
  [[nodiscard]] bool CheckInvariants() const noexcept;

 private:
  std::size_t block_size_tokens_;
  std::vector<std::uint32_t> refcounts_;
  std::vector<BlockId> free_list_;
  std::unordered_map<std::string, std::vector<BlockId>> sequence_blocks_;
  std::unordered_map<std::string, std::vector<BlockId>> prefix_blocks_;
};

}  // namespace infersched::engine
