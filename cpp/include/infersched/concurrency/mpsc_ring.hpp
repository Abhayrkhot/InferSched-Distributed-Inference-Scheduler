#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace infersched::concurrency {

// Bounded lock-free multi-producer/single-consumer queue using per-slot
// sequence numbers. Capacity must be a power of two.
template <typename T>
class MpscRing {
 public:
  explicit MpscRing(const std::size_t capacity)
      : capacity_(capacity), mask_(capacity - 1), cells_(new Cell[capacity]) {
    if (capacity < 2 || !std::has_single_bit(capacity)) {
      throw std::invalid_argument("MPSC capacity must be a power of two >= 2");
    }
    for (std::size_t index = 0; index < capacity_; ++index) {
      cells_[index].sequence.store(index, std::memory_order_relaxed);
    }
  }

  MpscRing(const MpscRing&) = delete;
  MpscRing& operator=(const MpscRing&) = delete;

  [[nodiscard]] bool TryPush(T value) {
    std::size_t position = enqueue_position_.load(std::memory_order_relaxed);
    while (true) {
      Cell& cell = cells_[position & mask_];
      const std::size_t sequence = cell.sequence.load(std::memory_order_acquire);
      const auto difference = static_cast<std::intptr_t>(sequence) -
                              static_cast<std::intptr_t>(position);
      if (difference == 0) {
        if (enqueue_position_.compare_exchange_weak(
                position, position + 1, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          cell.value.emplace(std::move(value));
          cell.sequence.store(position + 1, std::memory_order_release);
          return true;
        }
      } else if (difference < 0) {
        return false;
      } else {
        position = enqueue_position_.load(std::memory_order_relaxed);
      }
    }
  }

  [[nodiscard]] std::optional<T> TryPop() {
    Cell& cell = cells_[dequeue_position_ & mask_];
    const std::size_t sequence = cell.sequence.load(std::memory_order_acquire);
    const auto difference = static_cast<std::intptr_t>(sequence) -
                            static_cast<std::intptr_t>(dequeue_position_ + 1);
    if (difference < 0) {
      return std::nullopt;
    }
    if (difference != 0) {
      return std::nullopt;
    }
    T value = std::move(*cell.value);
    cell.value.reset();
    cell.sequence.store(dequeue_position_ + capacity_, std::memory_order_release);
    ++dequeue_position_;
    return value;
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  struct Cell {
    std::atomic<std::size_t> sequence{};
    std::optional<T> value;
  };

  const std::size_t capacity_;
  const std::size_t mask_;
  std::unique_ptr<Cell[]> cells_;
  alignas(64) std::atomic<std::size_t> enqueue_position_{};
  alignas(64) std::size_t dequeue_position_{};
};

}  // namespace infersched::concurrency
