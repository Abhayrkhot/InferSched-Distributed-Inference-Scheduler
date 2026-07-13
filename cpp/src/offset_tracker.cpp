#include "infersched/durable/offset_tracker.hpp"

namespace infersched::durable {

std::int64_t ContiguousOffsetTracker::MarkPublishComplete(
    const std::int64_t offset) {
  if (offset <= committed_offset_) {
    return committed_offset_;
  }
  completed_.insert(offset);
  while (completed_.contains(committed_offset_ + 1)) {
    completed_.erase(committed_offset_ + 1);
    ++committed_offset_;
  }
  return committed_offset_;
}

}  // namespace infersched::durable
