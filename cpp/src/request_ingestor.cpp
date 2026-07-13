#include "infersched/durable/request_ingestor.hpp"

#include <functional>

#include "infersched.pb.h"

namespace infersched::durable {

std::size_t RequestIngestor::SourcePositionHash::operator()(
    const SourcePosition& position) const {
  const std::size_t partition = std::hash<std::int32_t>{}(position.partition);
  const std::size_t offset = std::hash<std::int64_t>{}(position.offset);
  return partition ^ (offset + 0x9e3779b9U + (partition << 6U) +
                      (partition >> 2U));
}

IngestOutcome RequestIngestor::Handle(const std::int32_t partition,
                                      const std::int64_t offset,
                                      const std::string_view payload) {
  v1::InferenceRequest request;
  if (!request.ParseFromArray(payload.data(), static_cast<int>(payload.size())) ||
      request.request_id().empty()) {
    const SourcePosition position{.partition = partition, .offset = offset};
    std::size_t& attempts = parse_attempts_[position];
    ++attempts;
    if (attempts < max_parse_retries_) {
      return IngestOutcome::kRetryParse;
    }
    static_cast<void>(
        store_.RecordPoison(partition, offset, payload, "invalid_inference_request"));
    parse_attempts_.erase(position);
    return IngestOutcome::kPoisonRecorded;
  }
  parse_attempts_.erase(SourcePosition{.partition = partition, .offset = offset});
  return store_.Ingest(request.request_id(), partition, offset, payload)
             ? IngestOutcome::kInserted
             : IngestOutcome::kDuplicate;
}

}  // namespace infersched::durable
