#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <grpcpp/grpcpp.h>

#include "infersched/distributed/router_service.hpp"
#include "infersched/durable/durable_store.hpp"
#include "infersched/durable/kafka_poller.hpp"
#include "infersched/durable/outbox_relay.hpp"
#include "infersched/durable/request_ingestor.hpp"
#include "infersched.grpc.pb.h"

namespace {

std::string Argument(const int argc, char** argv, const std::string& name,
                     const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      return argv[index + 1];
    }
  }
  return fallback;
}

std::string Dispatch(infersched::v1::EngineControl::Stub& stub,
                     const infersched::v1::InferenceRequest& request,
                     const infersched::durable::Fence& fence) {
  infersched::v1::DispatchRequest dispatch;
  *dispatch.mutable_request() = request;
  dispatch.mutable_fence()->set_partition_id(fence.partition_id);
  dispatch.mutable_fence()->set_ownership_epoch(fence.ownership_epoch);
  dispatch.mutable_fence()->set_attempt_id(fence.attempt_id);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  auto reader = stub.Dispatch(&context, dispatch);
  infersched::v1::DispatchEvent event;
  std::string result;
  while (reader->Read(&event)) {
    if (event.has_ack() &&
        event.ack().status() == infersched::v1::DispatchAck::REJECTED) {
      return {};
    }
    if (event.has_progress() &&
        event.progress().kind() == infersched::v1::ProgressEvent::COMPLETED) {
      result = event.progress().result_ref();
    }
  }
  return reader->Finish().ok() ? result : std::string{};
}

std::string ResultEnvelope(const infersched::v1::InferenceRequest& request,
                           const infersched::durable::Fence& fence,
                           const std::string& output) {
  return "{\"request_id\":\"" + request.request_id() +
         "\",\"trace_id\":\"" + request.trace_id() +
         "\",\"span_id\":\"" + request.span_id() +
         "\",\"ownership_epoch\":" +
         std::to_string(fence.ownership_epoch) + ",\"attempt_id\":" +
         std::to_string(fence.attempt_id) + ",\"output\":\"" + output + "\"}";
}

}  // namespace

int main(const int argc, char** argv) {
  const std::string listen = Argument(argc, argv, "--listen", "127.0.0.1:50051");
  const std::string brokers = Argument(argc, argv, "--brokers", "localhost:9092");
  const std::string router_id = Argument(argc, argv, "--router-id", "router-p5");
  const std::string consumer_group =
      Argument(argc, argv, "--consumer-group", "infersched-router-p5");
  const std::string offset_reset =
      Argument(argc, argv, "--offset-reset", "earliest");
  const std::string ready_file = Argument(argc, argv, "--ready-file", "");
  const std::string database = Argument(
      argc, argv, "--database",
      "host=localhost port=55432 dbname=infersched user=infersched password=infersched");
  const std::size_t expected = static_cast<std::size_t>(
      std::stoull(Argument(argc, argv, "--requests", "1")));
  const auto run_seconds = std::chrono::seconds(
      std::stoll(Argument(argc, argv, "--run-seconds", "20")));

  infersched::durable::DurableStore store(database);
  store.Migrate();
  infersched::durable::RequestIngestor ingestor(store);
  infersched::durable::KafkaOutboxPublisher publisher(brokers);
  infersched::durable::OutboxRelay relay(store, publisher);
  infersched::durable::KafkaPoller poller(
      brokers, consumer_group, "inference.requests", offset_reset);

  infersched::distributed::RouterService service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  const auto engine = service.WaitForEngine(std::chrono::seconds(10));
  if (!server || !engine.has_value()) {
    poller.Stop();
    return EXIT_FAILURE;
  }
  auto stub = infersched::v1::EngineControl::NewStub(grpc::CreateChannel(
      engine->endpoint, grpc::InsecureChannelCredentials()));
  std::unordered_map<std::int32_t, std::uint64_t> epochs;
  std::size_t completed = 0;
  const auto execute = [&](const infersched::v1::InferenceRequest& request,
                           const std::int32_t partition) {
    const auto epoch = epochs.find(partition);
    if (epoch == epochs.end() || !poller.Owns(partition)) {
      return false;
    }
    const auto assignment = store.AssignAttempt(
        request.request_id(), partition, epoch->second, engine->engine_id);
    if (!assignment.has_value()) {
      return false;
    }
    const std::string result = Dispatch(*stub, request, assignment->fence);
    return !result.empty() && store.Finalize(
                                  request.request_id(), assignment->fence,
                                  ResultEnvelope(request, assignment->fence, result));
  };

  poller.Start();
  const auto deadline = std::chrono::steady_clock::now() + run_seconds;
  while (completed < expected && std::chrono::steady_clock::now() < deadline) {
    if (const auto ownership =
            poller.PopOwnershipEvent(std::chrono::milliseconds::zero())) {
      if (ownership->kind == infersched::durable::OwnershipEventKind::kRevoked) {
        for (const std::int32_t partition : ownership->partitions) {
          epochs.erase(partition);
        }
      } else {
        for (const std::int32_t partition : ownership->partitions) {
          epochs.insert_or_assign(
              partition, store.AcquirePartition(partition, router_id));
        }
        if (!ready_file.empty()) {
          std::ofstream marker(ready_file, std::ios::trunc);
          if (!marker) {
            std::cerr << "failed to write readiness marker: " << ready_file
                      << '\n';
            poller.Stop();
            server->Shutdown();
            return EXIT_FAILURE;
          }
          marker << "assigned\n";
        }
        const auto unresolved = store.RecoverUnresolved();
        for (const auto& recovered : unresolved) {
          if (!epochs.contains(recovered.source_partition)) {
            continue;
          }
          infersched::v1::InferenceRequest request;
          if (request.ParseFromString(recovered.raw_payload) &&
              execute(request, recovered.source_partition)) {
            static_cast<void>(relay.PublishBatch(1000));
            ++completed;
          }
        }
      }
      continue;
    }
    const auto message = poller.Pop(std::chrono::milliseconds(250));
    if (!message.has_value()) {
      continue;
    }
    if (!poller.Owns(message->partition)) {
      continue;
    }
    const auto outcome =
        ingestor.Handle(message->partition, message->offset, message->payload);
    if (outcome == infersched::durable::IngestOutcome::kRetryParse) {
      continue;
    }
    if (outcome == infersched::durable::IngestOutcome::kPoisonRecorded) {
      static_cast<void>(relay.PublishBatch(1000));
      if (store.IsSourcePublished(message->partition, message->offset)) {
        poller.MarkPublishComplete(message->partition, message->offset);
      }
      continue;
    }
    infersched::v1::InferenceRequest request;
    if (!request.ParseFromString(message->payload)) {
      continue;
    }
    if (store.IsTerminal(request.request_id())) {
      static_cast<void>(relay.PublishBatch(1000));
      if (store.IsSourcePublished(message->partition, message->offset)) {
        poller.MarkPublishComplete(message->partition, message->offset);
      }
      continue;
    }
    if (!execute(request, message->partition)) {
      continue;
    }
    static_cast<void>(relay.PublishBatch(1000));
    if (!store.IsSourcePublished(message->partition, message->offset)) {
      continue;
    }
    poller.MarkPublishComplete(message->partition, message->offset);
    ++completed;
  }
  poller.Stop();
  server->Shutdown();
  std::cout << "{\"completed\":" << completed << "}\n";
  return completed >= expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
