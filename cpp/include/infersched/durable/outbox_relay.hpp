#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

#include "infersched/durable/durable_store.hpp"

namespace infersched::durable {

class OutboxPublisher {
 public:
  virtual ~OutboxPublisher() = default;
  [[nodiscard]] virtual bool Publish(const OutboxRecord& record) = 0;
};

class KafkaOutboxPublisher final : public OutboxPublisher {
 public:
  explicit KafkaOutboxPublisher(
      std::string brokers,
      std::chrono::milliseconds timeout = std::chrono::seconds(5));
  ~KafkaOutboxPublisher() override;
  KafkaOutboxPublisher(KafkaOutboxPublisher&&) noexcept;
  KafkaOutboxPublisher& operator=(KafkaOutboxPublisher&&) noexcept;
  KafkaOutboxPublisher(const KafkaOutboxPublisher&) = delete;
  KafkaOutboxPublisher& operator=(const KafkaOutboxPublisher&) = delete;

  [[nodiscard]] bool Publish(const OutboxRecord& record) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class OutboxRelay {
 public:
  OutboxRelay(DurableStore& store, OutboxPublisher& publisher)
      : store_(store), publisher_(publisher) {}

  [[nodiscard]] std::size_t PublishBatch(std::size_t limit);

 private:
  DurableStore& store_;
  OutboxPublisher& publisher_;
};

}  // namespace infersched::durable
