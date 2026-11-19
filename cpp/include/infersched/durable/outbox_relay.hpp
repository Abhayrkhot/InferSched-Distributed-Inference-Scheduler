#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "infersched/durable/durable_store.hpp"

namespace infersched::durable {

class OutboxPublisher {
 public:
  virtual ~OutboxPublisher() = default;
  [[nodiscard]] virtual bool Publish(const OutboxRecord& record) = 0;
  // Publishes a batch and reports per-record broker acknowledgement. The
  // default publishes one record at a time and stops at the first failure.
  [[nodiscard]] virtual std::vector<bool> PublishMany(
      std::span<const OutboxRecord> records) {
    std::vector<bool> delivered(records.size(), false);
    for (std::size_t index = 0; index < records.size(); ++index) {
      delivered[index] = Publish(records[index]);
      if (!delivered[index]) {
        break;
      }
    }
    return delivered;
  }
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
  // Produces the whole batch, then waits once for all delivery reports.
  [[nodiscard]] std::vector<bool> PublishMany(
      std::span<const OutboxRecord> records) override;

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
