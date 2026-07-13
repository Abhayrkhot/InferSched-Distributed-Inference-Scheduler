#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <librdkafka/rdkafkacpp.h>

#include "infersched/durable/durable_store.hpp"
#include "infersched/durable/offset_tracker.hpp"
#include "infersched/durable/outbox_relay.hpp"
#include "infersched/durable/kafka_poller.hpp"
#include "infersched/durable/request_ingestor.hpp"
#include "infersched.pb.h"

namespace {

using infersched::durable::ContiguousOffsetTracker;
using infersched::durable::DurableStore;
using infersched::durable::Fence;
using infersched::durable::KafkaOutboxPublisher;
using infersched::durable::KafkaPoller;
using infersched::durable::OutboxPublisher;
using infersched::durable::OutboxRecord;
using infersched::durable::OutboxRelay;
using infersched::durable::RequestIngestor;

std::string Unique(const std::string& prefix) {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return prefix + "-" + std::to_string(ticks);
}

std::int64_t UniqueOffset() {
  return std::chrono::steady_clock::now().time_since_epoch().count();
}

class RecordingPublisher final : public OutboxPublisher {
 public:
  bool Publish(const OutboxRecord& record) override {
    records.push_back(record);
    return succeed;
  }
  bool succeed{true};
  std::vector<OutboxRecord> records;
};

TEST(ContiguousOffsetTracker, AdvancesOnlyAcrossCompletedPrefix) {
  ContiguousOffsetTracker tracker(9);
  EXPECT_EQ(tracker.MarkPublishComplete(11), 9);
  EXPECT_EQ(tracker.MarkPublishComplete(12), 9);
  EXPECT_EQ(tracker.MarkPublishComplete(10), 12);
  EXPECT_EQ(tracker.MarkPublishComplete(10), 12);
}

class PostgresDurability : public testing::Test {
 protected:
  void SetUp() override {
    try {
      store = std::make_unique<DurableStore>(
          "host=localhost port=55432 dbname=infersched user=infersched "
          "password=infersched connect_timeout=2");
      store->Migrate();
    } catch (const std::exception& error) {
      GTEST_SKIP() << "Postgres integration unavailable: " << error.what();
    }
  }
  std::unique_ptr<DurableStore> store;
};

TEST_F(PostgresDurability, CurrentFenceAloneCanFinalizeAndCreatesOneOutbox) {
  const std::string request_id = Unique("fenced");
  const std::int32_t partition = 700;
  const std::int64_t offset = UniqueOffset();
  const std::uint64_t epoch = store->AcquirePartition(partition, "router-a");
  ASSERT_TRUE(store->Ingest(request_id, partition, offset, "request"));
  EXPECT_FALSE(store->Ingest(request_id, partition, offset, "request"));
  const auto first = store->AssignAttempt(request_id, partition, epoch, "engine-a");
  ASSERT_TRUE(first.has_value());
  const auto current = store->AssignAttempt(request_id, partition, epoch, "engine-b");
  ASSERT_TRUE(current.has_value());
  EXPECT_GT(current->fence.attempt_id, first->fence.attempt_id);

  EXPECT_FALSE(store->Finalize(request_id, first->fence, "late-result"));
  EXPECT_TRUE(store->Finalize(request_id, current->fence, "accepted-result"));
  EXPECT_FALSE(store->Finalize(request_id, current->fence, "duplicate-result"));
  EXPECT_TRUE(store->IsTerminal(request_id));

  const auto rows = store->FetchUnpublished(1000);
  EXPECT_EQ(std::count_if(rows.begin(), rows.end(), [&](const OutboxRecord& row) {
              return row.topic == "inference.results" &&
                     row.message_key == request_id &&
                     row.payload == "accepted-result";
            }), 1);
}

TEST_F(PostgresDurability, OptimisticStateVersionRejectsStaleTransitions) {
  const std::string request_id = Unique("versioned");
  ASSERT_TRUE(store->Ingest(request_id, 704, UniqueOffset(), "request"));
  EXPECT_TRUE(store->Transition(request_id, 0, "RECEIVED", "QUEUED"));
  EXPECT_FALSE(store->Transition(request_id, 0, "RECEIVED", "QUEUED"));
  EXPECT_FALSE(store->Transition(request_id, 1, "RECEIVED", "DISPATCHED"));
  EXPECT_TRUE(store->Transition(request_id, 1, "QUEUED", "CANCELLED"));
  EXPECT_FALSE(store->Transition(request_id, 2, "CANCELLED", "QUEUED"));
}

TEST_F(PostgresDurability, PoisonAndDlqOutboxAreAtomicAndIdempotent) {
  const std::int32_t partition = 701;
  const std::int64_t offset = UniqueOffset();
  EXPECT_TRUE(store->RecordPoison(partition, offset, "not-protobuf", "parse"));
  EXPECT_FALSE(store->RecordPoison(partition, offset, "not-protobuf", "parse"));
  const auto rows = store->FetchUnpublished(1000);
  EXPECT_EQ(std::count_if(rows.begin(), rows.end(), [&](const OutboxRecord& row) {
              return row.topic == "inference.dlq" &&
                     row.source_partition == partition && row.source_offset == offset;
            }), 1);
}

TEST_F(PostgresDurability, BinaryPoisonIsHexEncodedInTextOutbox) {
  const std::int32_t partition = 708;
  const std::int64_t offset = UniqueOffset();
  const std::string payload{"\x00\x80\xff", 3};
  ASSERT_TRUE(store->RecordPoison(partition, offset, payload, "parse"));

  const auto rows = store->FetchUnpublished(1000);
  const auto row = std::find_if(
      rows.begin(), rows.end(), [&](const OutboxRecord& candidate) {
        return candidate.topic == "inference.dlq" &&
               candidate.source_partition == partition &&
               candidate.source_offset == offset;
      });
  ASSERT_NE(row, rows.end());
  EXPECT_EQ(row->payload, "0080ff");
}

TEST_F(PostgresDurability, AtLeastOnceIngestDeduplicatesAndPoisonsAfterBoundedRetries) {
  RequestIngestor ingestor(*store, 3);
  infersched::v1::InferenceRequest request;
  request.set_request_id(Unique("ingested"));
  const std::string payload = request.SerializeAsString();
  const std::int64_t valid_offset = UniqueOffset();
  EXPECT_EQ(ingestor.Handle(707, valid_offset, payload),
            infersched::durable::IngestOutcome::kInserted);
  EXPECT_EQ(ingestor.Handle(707, valid_offset, payload),
            infersched::durable::IngestOutcome::kDuplicate);

  const std::int64_t poison_offset = UniqueOffset();
  EXPECT_EQ(ingestor.Handle(707, poison_offset, "invalid"),
            infersched::durable::IngestOutcome::kRetryParse);
  EXPECT_EQ(ingestor.Handle(707, poison_offset, "invalid"),
            infersched::durable::IngestOutcome::kRetryParse);
  EXPECT_EQ(ingestor.Handle(707, poison_offset, "invalid"),
            infersched::durable::IngestOutcome::kPoisonRecorded);
  const auto rows = store->FetchUnpublished(1000);
  EXPECT_NE(std::find_if(rows.begin(), rows.end(), [&](const OutboxRecord& row) {
              return row.topic == "inference.dlq" &&
                     row.source_partition == 707 &&
                     row.source_offset == poison_offset;
            }), rows.end());
}

TEST_F(PostgresDurability, RestartRecoversOnlyUnresolvedRequests) {
  const std::string request_id = Unique("recover");
  const std::int32_t partition = 702;
  ASSERT_TRUE(store->Ingest(request_id, partition, UniqueOffset(), "request"));
  DurableStore restarted(
      "host=localhost port=55432 dbname=infersched user=infersched "
      "password=infersched connect_timeout=2");
  const auto unresolved = restarted.RecoverUnresolved();
  EXPECT_NE(std::find_if(unresolved.begin(), unresolved.end(),
                         [&](const auto& request) {
                           return request.request_id == request_id;
                         }), unresolved.end());
}

TEST_F(PostgresDurability, RelayMarksOnlyAcknowledgedRowsPublished) {
  const std::string request_id = Unique("relay");
  const std::int32_t partition = 703;
  const std::int64_t offset = UniqueOffset();
  const std::uint64_t epoch = store->AcquirePartition(partition, "router-a");
  ASSERT_TRUE(store->Ingest(request_id, partition, offset, "request"));
  const auto assignment =
      store->AssignAttempt(request_id, partition, epoch, "engine-a");
  ASSERT_TRUE(assignment.has_value());
  ASSERT_TRUE(store->Finalize(request_id, assignment->fence, "result"));

  // Simulate a Router crash after the transaction commits but before publish.
  store.reset();
  DurableStore restarted(
      "host=localhost port=55432 dbname=infersched user=infersched "
      "password=infersched connect_timeout=2");
  RecordingPublisher publisher;
  OutboxRelay relay(restarted, publisher);
  EXPECT_GE(relay.PublishBatch(1000), 1u);
  EXPECT_TRUE(restarted.IsSourcePublished(partition, offset));
  const auto remaining = restarted.FetchUnpublished(1000);
  EXPECT_EQ(std::count_if(remaining.begin(), remaining.end(),
                          [&](const OutboxRecord& row) {
                            return row.message_key == request_id;
                          }), 0);
}

TEST_F(PostgresDurability, RelayRetainsRowsWhenPublishIsNotAcknowledged) {
  const std::int32_t partition = 705;
  const std::int64_t offset = UniqueOffset();
  ASSERT_TRUE(store->RecordPoison(partition, offset, "bad", "parse"));
  RecordingPublisher publisher;
  publisher.succeed = false;
  OutboxRelay relay(*store, publisher);
  EXPECT_EQ(relay.PublishBatch(1000), 0u);
  const auto remaining = store->FetchUnpublished(1000);
  EXPECT_NE(std::find_if(remaining.begin(), remaining.end(),
                         [&](const OutboxRecord& row) {
                           return row.source_partition == partition &&
                                  row.source_offset == offset;
                         }), remaining.end());
}

TEST_F(PostgresDurability, KafkaRelayPublishesBeforeMarkingOutboxComplete) {
  const std::string request_id = Unique("kafka-result");
  const std::int32_t partition = 706;
  const std::uint64_t epoch = store->AcquirePartition(partition, "router-kafka");
  ASSERT_TRUE(store->Ingest(request_id, partition, UniqueOffset(), "request"));
  const auto assignment =
      store->AssignAttempt(request_id, partition, epoch, "engine-kafka");
  ASSERT_TRUE(assignment.has_value());
  ASSERT_TRUE(store->Finalize(request_id, assignment->fence, "kafka-payload"));

  try {
    KafkaOutboxPublisher publisher("localhost:9092");
    OutboxRelay relay(*store, publisher);
    ASSERT_GE(relay.PublishBatch(1000), 1u);
  } catch (const std::exception& error) {
    GTEST_SKIP() << "Kafka integration unavailable: " << error.what();
  }
  const auto remaining = store->FetchUnpublished(1000);
  EXPECT_EQ(std::count_if(remaining.begin(), remaining.end(),
                          [&](const OutboxRecord& row) {
                            return row.message_key == request_id;
                          }), 0);
}

TEST(KafkaDurability, DedicatedPollThreadReceivesPublishedRequest) {
  const std::string key = Unique("poll-key");
  const std::string payload = Unique("poll-payload");
  KafkaPoller poller("localhost:9092", Unique("poll-group"),
                     "inference.requests");
  poller.Start();
  const auto assignment =
      poller.PopOwnershipEvent(std::chrono::seconds(10));
  if (!assignment.has_value()) {
    poller.Stop();
    GTEST_SKIP() << "Kafka integration unavailable or assignment timed out";
  }
  ASSERT_EQ(assignment->kind,
            infersched::durable::OwnershipEventKind::kAssigned);
  KafkaOutboxPublisher publisher("localhost:9092");
  ASSERT_TRUE(publisher.Publish(OutboxRecord{
      .topic = "inference.requests", .message_key = key, .payload = payload}));

  bool received = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!received && std::chrono::steady_clock::now() < deadline) {
    const auto message = poller.Pop(std::chrono::milliseconds(250));
    if (message.has_value() && message->key == key) {
      EXPECT_EQ(message->payload, payload);
      poller.MarkPublishComplete(message->partition, message->offset);
      received = true;
    }
  }
  poller.Stop();
  EXPECT_TRUE(received);
}

TEST(KafkaDurability, ConsumerGroupRebalancesPartitionsAfterRouterStops) {
  const std::string group = Unique("rebalance-group");
  KafkaPoller first("localhost:9092", group, "inference.requests");
  KafkaPoller second("localhost:9092", group, "inference.requests");
  first.Start();
  second.Start();

  std::unordered_set<std::int32_t> first_partitions;
  std::unordered_set<std::int32_t> second_partitions;
  const auto initial_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(12);
  while ((first_partitions.empty() || second_partitions.empty()) &&
         std::chrono::steady_clock::now() < initial_deadline) {
    if (const auto event =
            first.PopOwnershipEvent(std::chrono::milliseconds(100));
        event.has_value() &&
        event->kind == infersched::durable::OwnershipEventKind::kAssigned) {
      first_partitions.insert(event->partitions.begin(), event->partitions.end());
    }
    if (const auto event =
            second.PopOwnershipEvent(std::chrono::milliseconds(100));
        event.has_value() &&
        event->kind == infersched::durable::OwnershipEventKind::kAssigned) {
      second_partitions.insert(event->partitions.begin(), event->partitions.end());
    }
  }
  ASSERT_FALSE(first_partitions.empty());
  ASSERT_FALSE(second_partitions.empty());
  for (const std::int32_t partition : first_partitions) {
    EXPECT_FALSE(second_partitions.contains(partition));
  }

  first.Stop();
  const auto takeover_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(12);
  while (second_partitions.size() < 12 &&
         std::chrono::steady_clock::now() < takeover_deadline) {
    const auto event =
        second.PopOwnershipEvent(std::chrono::milliseconds(250));
    if (event.has_value() &&
        event->kind == infersched::durable::OwnershipEventKind::kAssigned) {
      second_partitions.insert(event->partitions.begin(), event->partitions.end());
    }
  }
  second.Stop();
  EXPECT_EQ(second_partitions.size(), 12u);
}

}  // namespace
