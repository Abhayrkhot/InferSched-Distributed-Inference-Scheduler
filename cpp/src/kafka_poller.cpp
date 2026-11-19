#include "infersched/durable/kafka_poller.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <librdkafka/rdkafkacpp.h>

namespace infersched::durable {

KafkaPoller::KafkaPoller(std::string brokers, std::string group_id,
                         std::string topic, std::string offset_reset)
    : brokers_(std::move(brokers)), group_id_(std::move(group_id)),
      topic_(std::move(topic)), offset_reset_(std::move(offset_reset)) {
  if (offset_reset_ != "earliest" && offset_reset_ != "latest") {
    throw std::invalid_argument("offset reset must be earliest or latest");
  }
}

KafkaPoller::~KafkaPoller() { Stop(); }

void KafkaPoller::Start() {
  std::lock_guard lock(mutex_);
  if (thread_.joinable()) {
    return;
  }
  stop_ = false;
  thread_ = std::thread(&KafkaPoller::PollLoop, this);
}

void KafkaPoller::Stop() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  ready_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::optional<KafkaMessage> KafkaPoller::Pop(
    const std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  ready_.wait_for(lock, timeout, [&] { return stop_ || !messages_.empty(); });
  if (messages_.empty()) {
    return std::nullopt;
  }
  KafkaMessage message = std::move(messages_.front());
  messages_.pop_front();
  return message;
}

std::optional<OwnershipEvent> KafkaPoller::PopOwnershipEvent(
    const std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  ready_.wait_for(lock, timeout,
                  [&] { return stop_ || !ownership_events_.empty(); });
  if (ownership_events_.empty()) {
    return std::nullopt;
  }
  OwnershipEvent event = std::move(ownership_events_.front());
  ownership_events_.pop_front();
  return event;
}

bool KafkaPoller::Owns(const std::int32_t partition) const {
  std::lock_guard lock(mutex_);
  return assigned_partitions_.contains(partition);
}

void KafkaPoller::rebalance_cb(
    RdKafka::KafkaConsumer* consumer, const RdKafka::ErrorCode error,
    std::vector<RdKafka::TopicPartition*>& partitions) {
  std::vector<std::int32_t> ids;
  ids.reserve(partitions.size());
  for (const RdKafka::TopicPartition* partition : partitions) {
    ids.push_back(partition->partition());
  }
  if (error == RdKafka::ERR__ASSIGN_PARTITIONS) {
    static_cast<void>(consumer->assign(partitions));
    {
      std::lock_guard lock(mutex_);
      assigned_partitions_.insert(ids.begin(), ids.end());
      ownership_events_.push_back(
          OwnershipEvent{.kind = OwnershipEventKind::kAssigned,
                         .partitions = std::move(ids)});
    }
  } else {
    {
      std::lock_guard lock(mutex_);
      for (const std::int32_t partition : ids) {
        assigned_partitions_.erase(partition);
        trackers_.erase(partition);
        std::erase_if(messages_, [&](const KafkaMessage& message) {
          return message.partition == partition;
        });
      }
      ownership_events_.push_back(
          OwnershipEvent{.kind = OwnershipEventKind::kRevoked,
                         .partitions = std::move(ids)});
    }
    static_cast<void>(consumer->unassign());
  }
  ready_.notify_all();
}

void KafkaPoller::MarkPublishComplete(const std::int32_t partition,
                                      const std::int64_t offset) {
  {
    std::lock_guard lock(mutex_);
    const auto tracker = trackers_.find(partition);
    if (tracker == trackers_.end()) {
      return;
    }
    const std::int64_t before = tracker->second.committed_offset();
    const std::int64_t safe = tracker->second.MarkPublishComplete(offset);
    if (safe > before) {
      commits_.emplace_back(partition, safe);
    }
  }
  ready_.notify_all();
}

void KafkaPoller::PollLoop() {
  std::string error;
  std::unique_ptr<RdKafka::Conf> config(
      RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  if (config->set("bootstrap.servers", brokers_, error) !=
          RdKafka::Conf::CONF_OK ||
      config->set("group.id", group_id_, error) != RdKafka::Conf::CONF_OK ||
      config->set("enable.auto.commit", "false", error) !=
          RdKafka::Conf::CONF_OK ||
      config->set("session.timeout.ms", "6000", error) !=
          RdKafka::Conf::CONF_OK ||
      config->set("heartbeat.interval.ms", "1000", error) !=
          RdKafka::Conf::CONF_OK ||
      config->set("auto.offset.reset", offset_reset_, error) !=
          RdKafka::Conf::CONF_OK ||
      config->set("rebalance_cb", this, error) != RdKafka::Conf::CONF_OK) {
    return;
  }
  std::unique_ptr<RdKafka::KafkaConsumer> consumer(
      RdKafka::KafkaConsumer::create(config.get(), error));
  if (!consumer || consumer->subscribe({topic_}) != RdKafka::ERR_NO_ERROR) {
    return;
  }

  // Only the highest safe offset per partition matters, so commits are
  // coalesced and flushed at most every kCommitInterval, asynchronously while
  // running (a lost commit only causes a redelivery that ingest deduplicates)
  // and synchronously on shutdown. Per-offset synchronous commits stalled
  // consumption and loaded the group coordinator under sustained traffic.
  constexpr auto kCommitInterval = std::chrono::milliseconds(100);
  std::map<std::int32_t, std::int64_t> latest;
  auto last_commit = std::chrono::steady_clock::now();
  while (true) {
    std::deque<std::pair<std::int32_t, std::int64_t>> commits;
    bool should_stop = false;
    {
      std::lock_guard lock(mutex_);
      should_stop = stop_;
      commits.swap(commits_);
    }
    for (const auto& [partition, offset] : commits) {
      const auto [entry, inserted] = latest.try_emplace(partition, offset);
      if (!inserted) {
        entry->second = std::max(entry->second, offset);
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (!latest.empty() && (should_stop || now - last_commit >= kCommitInterval)) {
      last_commit = now;
      std::vector<RdKafka::TopicPartition*> positions;
      positions.reserve(latest.size());
      for (const auto& [partition, offset] : latest) {
        positions.push_back(RdKafka::TopicPartition::create(topic_, partition, offset + 1));
      }
      if (should_stop) {
        static_cast<void>(consumer->commitSync(positions));
      } else {
        static_cast<void>(consumer->commitAsync(positions));
      }
      RdKafka::TopicPartition::destroy(positions);
      latest.clear();
    }
    if (should_stop) {
      break;
    }

    std::unique_ptr<RdKafka::Message> message(consumer->consume(50));
    if (message->err() != RdKafka::ERR_NO_ERROR) {
      continue;
    }
    const char* payload = static_cast<const char*>(message->payload());
    const std::string* key = message->key();
    {
      std::lock_guard lock(mutex_);
      static_cast<void>(trackers_.try_emplace(
          message->partition(), message->offset() - 1));
      messages_.push_back(KafkaMessage{
          .payload = std::string(payload, message->len()),
          .key = key == nullptr ? std::string{} : *key,
          .partition = message->partition(), .offset = message->offset(),
          .timestamp_ms = message->timestamp().timestamp});
    }
    ready_.notify_one();
  }
  static_cast<void>(consumer->close());
}

}  // namespace infersched::durable
