#include "infersched/durable/outbox_relay.hpp"

#include <atomic>
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
    if (config->set("dr_cb", this, error) != RdKafka::Conf::CONF_OK) {
      throw std::runtime_error(error);
    }
    producer.reset(RdKafka::Producer::create(config.get(), error));
    if (!producer) {
      throw std::runtime_error(error);
    }
  }

  void dr_cb(RdKafka::Message& message) override {
    delivered.store(message.err() == RdKafka::ERR_NO_ERROR,
                    std::memory_order_release);
    complete.store(true, std::memory_order_release);
  }

  std::unique_ptr<RdKafka::Producer> producer;
  std::chrono::milliseconds timeout;
  std::atomic<bool> complete{};
  std::atomic<bool> delivered{};
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
  impl_->complete.store(false, std::memory_order_release);
  impl_->delivered.store(false, std::memory_order_release);
  const RdKafka::ErrorCode error = impl_->producer->produce(
      record.topic, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
      const_cast<char*>(record.payload.data()), record.payload.size(),
      record.message_key.data(), record.message_key.size(), 0, nullptr);
  if (error != RdKafka::ERR_NO_ERROR) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + impl_->timeout;
  while (!impl_->complete.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    impl_->producer->poll(10);
  }
  return impl_->delivered.load(std::memory_order_acquire);
}

std::size_t OutboxRelay::PublishBatch(const std::size_t limit) {
  std::size_t published = 0;
  for (const OutboxRecord& record : store_.FetchUnpublished(limit)) {
    if (!publisher_.Publish(record)) {
      break;
    }
    store_.MarkPublished(record.id);
    ++published;
  }
  return published;
}

}  // namespace infersched::durable
