#pragma once

#include <cstdint>
#include <set>

namespace infersched::durable {

class ContiguousOffsetTracker {
 public:
  explicit ContiguousOffsetTracker(std::int64_t committed_offset = -1)
      : committed_offset_(committed_offset) {}

  [[nodiscard]] std::int64_t MarkPublishComplete(std::int64_t offset);
  [[nodiscard]] std::int64_t committed_offset() const {
    return committed_offset_;
  }

 private:
  std::int64_t committed_offset_;
  std::set<std::int64_t> completed_;
};

}  // namespace infersched::durable
