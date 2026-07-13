#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "infersched/durable/durable_store.hpp"

namespace infersched::durable {

enum class IngestOutcome { kInserted, kDuplicate, kRetryParse, kPoisonRecorded };

class RequestIngestor {
 public:
  explicit RequestIngestor(DurableStore& store, std::size_t max_parse_retries = 3)
      : store_(store), max_parse_retries_(max_parse_retries) {}

  [[nodiscard]] IngestOutcome Handle(std::int32_t partition,
                                     std::int64_t offset,
                                     std::string_view payload);

 private:
  struct SourcePosition {
    std::int32_t partition{};
    std::int64_t offset{};
    bool operator==(const SourcePosition&) const = default;
  };
  struct SourcePositionHash {
    [[nodiscard]] std::size_t operator()(const SourcePosition& position) const;
  };

  DurableStore& store_;
  std::size_t max_parse_retries_;
  std::unordered_map<SourcePosition, std::size_t, SourcePositionHash> parse_attempts_;
};

}  // namespace infersched::durable
