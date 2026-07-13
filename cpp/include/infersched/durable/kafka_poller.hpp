#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "infersched/durable/offset_tracker.hpp"
#include <librdkafka/rdkafkacpp.h>

namespace infersched::durable {

struct KafkaMessage {
  std::string payload;
  std::string key;
  std::int32_t partition{};
  std::int64_t offset{};
};

enum class OwnershipEventKind { kAssigned, kRevoked };

struct OwnershipEvent {
  OwnershipEventKind kind{};
  std::vector<std::int32_t> partitions;
};

class KafkaPoller : private RdKafka::RebalanceCb {
 public:
  KafkaPoller(std::string brokers, std::string group_id, std::string topic,
              std::string offset_reset = "earliest");
  ~KafkaPoller();
  KafkaPoller(const KafkaPoller&) = delete;
  KafkaPoller& operator=(const KafkaPoller&) = delete;

  void Start();
  void Stop();
  [[nodiscard]] std::optional<KafkaMessage> Pop(
      std::chrono::milliseconds timeout);
  [[nodiscard]] std::optional<OwnershipEvent> PopOwnershipEvent(
      std::chrono::milliseconds timeout);
  [[nodiscard]] bool Owns(std::int32_t partition) const;
  void MarkPublishComplete(std::int32_t partition, std::int64_t offset);

 private:
  void rebalance_cb(RdKafka::KafkaConsumer* consumer, RdKafka::ErrorCode error,
                    std::vector<RdKafka::TopicPartition*>& partitions) override;
  void PollLoop();

  std::string brokers_;
  std::string group_id_;
  std::string topic_;
  std::string offset_reset_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<KafkaMessage> messages_;
  std::deque<OwnershipEvent> ownership_events_;
  std::deque<std::pair<std::int32_t, std::int64_t>> commits_;
  std::unordered_map<std::int32_t, ContiguousOffsetTracker> trackers_;
  std::unordered_set<std::int32_t> assigned_partitions_;
  bool stop_{};
  std::thread thread_;
};

}  // namespace infersched::durable
