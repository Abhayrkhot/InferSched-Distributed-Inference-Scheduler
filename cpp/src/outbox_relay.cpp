#include "infersched/durable/outbox_relay.hpp"

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <librdkafka/rdkafkacpp.h>

namespace infersched::durable {

class KafkaOutboxPublisher::Impl final : public RdKafka::DeliveryReportCb {
 public:
  Impl(std::string brokers, const std::chrono::milliseconds publish_timeout)
      : timeout(publish_timeout) {
    std::string error;
    std::unique_ptr<RdKafka::Conf> config(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    if (config->set("bootstrap.servers", brokers, error) !=
        RdKafka::Conf::CONF_OK) {
      throw std::runtime_error(error);
    }
    if (config->set("enable.idempotence", "true", error) !=
        RdKafka::Conf::CONF_OK) {
      throw std::runtime_error(error);
    }
    if (config->set("linger.ms", "1", error) != RdKafka::Conf::CONF_OK) {
      throw std::runtime_error(error);
    }
    if (config->set("dr_cb", this, error) != RdKafka::Conf::CONF_OK) {
      throw std::runtime_error(error);
    }
    producer.reset(RdKafka::Producer::create(config.get(), error));
    if (!producer) {
      throw std::runtime_error(error);
    }
  }

  // Delivery reports arrive on the thread calling poll(), which is always the
  // thread holding `batch_mutex` inside PublishMany. The opaque carries the
  // batch generation so a late report from a timed-out batch can never mark a
  // record of the current batch as delivered.
  void dr_cb(RdKafka::Message& message) override {
    const auto opaque = reinterpret_cast<std::uintptr_t>(message.msg_opaque());
    const std::uint64_t batch = static_cast<std::uint64_t>(opaque) >> 32U;
    const std::size_t slot = static_cast<std::size_t>(opaque & 0xffffffffU);
    if (batch != (generation & 0xffffffffU) || slot == 0 || slot > status.size() ||
        status[slot - 1] != kPending) {
      return;
    }
    status[slot - 1] = message.err() == RdKafka::ERR_NO_ERROR ? kDelivered : kFailed;
    --pending;
  }

  [[nodiscard]] void* Opaque(const std::size_t index) const {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(
        (generation << 32U) | static_cast<std::uint64_t>(index + 1)));
  }

  static constexpr int kPending = 0;
  static constexpr int kDelivered = 1;
  static constexpr int kFailed = 2;

  std::unique_ptr<RdKafka::Producer> producer;
  std::chrono::milliseconds timeout;
  std::mutex batch_mutex;
  std::uint64_t generation{};
  std::vector<int> status;
  std::size_t pending{};
};

KafkaOutboxPublisher::KafkaOutboxPublisher(
    std::string brokers, const std::chrono::milliseconds timeout)
    : impl_(std::make_unique<Impl>(std::move(brokers), timeout)) {}
KafkaOutboxPublisher::~KafkaOutboxPublisher() = default;
KafkaOutboxPublisher::KafkaOutboxPublisher(KafkaOutboxPublisher&&) noexcept =
    default;
KafkaOutboxPublisher& KafkaOutboxPublisher::operator=(
    KafkaOutboxPublisher&&) noexcept = default;

bool KafkaOutboxPublisher::Publish(const OutboxRecord& record) {
  return PublishMany(std::span<const OutboxRecord>(&record, 1)).front();
}

std::vector<bool> KafkaOutboxPublisher::PublishMany(
    const std::span<const OutboxRecord> records) {
  std::lock_guard lock(impl_->batch_mutex);
  if (records.size() >= 0xffffffffU) {
    throw std::invalid_argument("outbox batch too large");
  }
  ++impl_->generation;
  impl_->status.assign(records.size(), Impl::kPending);
  impl_->pending = 0;
  for (std::size_t index = 0; index < records.size(); ++index) {
    const OutboxRecord& record = records[index];
    RdKafka::ErrorCode error = RdKafka::ERR_NO_ERROR;
    // A full local queue is transient: serve delivery reports and retry.
    for (int retry = 0; retry < 100; ++retry) {
      error = impl_->producer->produce(
          record.topic, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
          const_cast<char*>(record.payload.data()), record.payload.size(),
          record.message_key.data(), record.message_key.size(), 0,
          impl_->Opaque(index));
      if (error != RdKafka::ERR__QUEUE_FULL) {
        break;
      }
      impl_->producer->poll(10);
    }
    if (error == RdKafka::ERR_NO_ERROR) {
      ++impl_->pending;
    } else {
      impl_->status[index] = Impl::kFailed;
    }
  }
  const auto deadline = std::chrono::steady_clock::now() + impl_->timeout;
  while (impl_->pending > 0 && std::chrono::steady_clock::now() < deadline) {
    impl_->producer->poll(10);
  }
  std::vector<bool> delivered(records.size(), false);
  for (std::size_t index = 0; index < records.size(); ++index) {
    delivered[index] = impl_->status[index] == Impl::kDelivered;
  }
  return delivered;
}

std::size_t OutboxRelay::PublishBatch(const std::size_t limit) {
  const std::vector<OutboxRecord> records = store_.FetchUnpublished(limit);
  if (records.empty()) {
    return 0;
  }
  const std::vector<bool> delivered = publisher_.PublishMany(records);
  std::vector<std::int64_t> acknowledged;
  acknowledged.reserve(records.size());
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (delivered[index]) {
      acknowledged.push_back(records[index].id);
    }
  }
  store_.MarkPublishedMany(acknowledged);
  return acknowledged.size();
}

}  // namespace infersched::durable
