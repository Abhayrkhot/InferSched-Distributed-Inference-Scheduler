#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infersched::durable {

struct Fence {
  std::int32_t partition_id{};
  std::uint64_t ownership_epoch{};
  std::uint64_t attempt_id{};
  std::string engine_id;
};

struct Assignment {
  Fence fence;
  std::uint64_t state_version{};
};

struct OutboxRecord {
  std::int64_t id{};
  std::string topic;
  std::string message_key;
  std::string payload;
  std::int32_t source_partition{};
  std::int64_t source_offset{};
};

struct RecoveredRequest {
  std::string request_id;
  std::string state;
  std::int32_t source_partition{};
  std::int64_t source_offset{};
  std::uint64_t state_version{};
  std::string raw_payload;
};

class DurableStore {
 public:
  explicit DurableStore(std::string connection_string);
  ~DurableStore();
  DurableStore(DurableStore&&) noexcept;
  DurableStore& operator=(DurableStore&&) noexcept;
  DurableStore(const DurableStore&) = delete;
  DurableStore& operator=(const DurableStore&) = delete;

  void Migrate();
  [[nodiscard]] std::uint64_t AcquirePartition(std::int32_t partition_id,
                                                std::string_view router_id);
  [[nodiscard]] bool Ingest(std::string_view request_id,
                            std::int32_t source_partition,
                            std::int64_t source_offset,
                            std::string_view raw_payload);
  [[nodiscard]] std::optional<Assignment> AssignAttempt(
      std::string_view request_id, std::int32_t partition_id,
      std::uint64_t ownership_epoch, std::string_view engine_id);
  [[nodiscard]] bool Transition(std::string_view request_id,
                                std::uint64_t expected_state_version,
                                std::string_view expected_state,
                                std::string_view next_state);
  [[nodiscard]] bool Finalize(std::string_view request_id, const Fence& fence,
                              std::string_view result_payload);
  [[nodiscard]] bool RecordPoison(std::int32_t source_partition,
                                  std::int64_t source_offset,
                                  std::string_view raw_payload,
                                  std::string_view error);
  [[nodiscard]] std::vector<OutboxRecord> FetchUnpublished(
      std::size_t limit) const;
  void MarkPublished(std::int64_t outbox_id);
  [[nodiscard]] std::vector<RecoveredRequest> RecoverUnresolved() const;
  [[nodiscard]] bool IsTerminal(std::string_view request_id) const;
  [[nodiscard]] bool IsSourcePublished(std::int32_t source_partition,
                                       std::int64_t source_offset) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace infersched::durable
