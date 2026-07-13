#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "infersched/distributed/router_service.hpp"
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

bool DispatchOne(infersched::v1::EngineControl::Stub& stub,
                 const std::size_t index, const std::uint64_t attempt,
                 const std::uint64_t ownership_epoch, bool expect_rejection) {
  infersched::v1::DispatchRequest request;
  request.mutable_request()->set_request_id("grpc-request-" +
                                             std::to_string(index));
  request.mutable_request()->set_model_id("benchmark-model");
  request.mutable_request()->set_model_revision("benchmark-model-v1");
  request.mutable_request()->set_tokenizer_revision("tokenizer-v1");
  request.mutable_request()->set_prompt_hash("prompt-" + std::to_string(index));
  request.mutable_request()->set_prompt_tokens(16 +
                                                static_cast<std::uint32_t>(index % 16));
  request.mutable_request()->mutable_sampling()->set_max_output_tokens(4);
  request.mutable_request()->mutable_sampling()->set_seed(index);
  request.mutable_fence()->set_partition_id(0);
  request.mutable_fence()->set_ownership_epoch(ownership_epoch);
  request.mutable_fence()->set_attempt_id(attempt);

  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  auto reader = stub.Dispatch(&context, request);
  infersched::v1::DispatchEvent event;
  bool rejected = false;
  bool completed = false;
  while (reader->Read(&event)) {
    if (event.has_ack()) {
      rejected = event.ack().status() == infersched::v1::DispatchAck::REJECTED;
    } else if (event.has_progress() &&
               event.progress().kind() == infersched::v1::ProgressEvent::COMPLETED) {
      completed = true;
    }
  }
  const grpc::Status status = reader->Finish();
  return status.ok() && (expect_rejection ? rejected : completed && !rejected);
}

}  // namespace

int main(const int argc, char** argv) {
  const std::string listen = Argument(argc, argv, "--listen", "127.0.0.1:50051");
  const std::size_t requests = static_cast<std::size_t>(
      std::stoull(Argument(argc, argv, "--requests", "32")));
  const bool retry_demo = Argument(argc, argv, "--retry-demo", "false") == "true";
  const bool overlapping_retry_demo =
      Argument(argc, argv, "--overlapping-retry-demo", "false") == "true";
  const std::string marker = Argument(argc, argv, "--dispatch-marker", "");

  infersched::distributed::RouterService service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server) {
    std::cerr << "failed to start RouterControl at " << listen << '\n';
    return EXIT_FAILURE;
  }
  const auto engine = service.WaitForEngine(std::chrono::seconds(10));
  if (!engine.has_value()) {
    std::cerr << "no Engine registered\n";
    server->Shutdown();
    return EXIT_FAILURE;
  }

  auto channel = grpc::CreateChannel(engine->endpoint, grpc::InsecureChannelCredentials());
  auto stub = infersched::v1::EngineControl::NewStub(channel);
  if (overlapping_retry_demo) {
    std::atomic<bool> first_completed{false};
    std::thread first_attempt([&] {
      first_completed.store(DispatchOne(*stub, 0, 1, 1, false),
                            std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool overlapping_retry_rejected = DispatchOne(*stub, 0, 2, 1, true);
    first_attempt.join();

    infersched::v1::DrainRequest drain_request;
    drain_request.set_engine_id(engine->engine_id);
    infersched::v1::DrainReply drain_reply;
    grpc::ClientContext drain_context;
    const grpc::Status drain_status =
        stub->Drain(&drain_context, drain_request, &drain_reply);
    server->Shutdown();
    const bool drained = drain_status.ok() && drain_reply.in_flight() == 0;
    const bool success = first_completed.load(std::memory_order_acquire) &&
                         overlapping_retry_rejected && drained;
    std::cout << "{\"first_completed\":" << std::boolalpha
              << first_completed.load(std::memory_order_acquire)
              << ",\"overlapping_retry_rejected\":"
              << overlapping_retry_rejected << ",\"drained\":" << drained
              << "}\n";
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (retry_demo) {
    if (!marker.empty()) {
      std::ofstream marker_file(marker);
      marker_file << "dispatching\n";
    }
    const bool first_attempt_failed = !DispatchOne(*stub, 0, 1, 1, false);
    const auto replacement = service.WaitForNewIncarnation(
        engine->incarnation_id, std::chrono::seconds(10));
    bool retry_completed = false;
    bool drained = false;
    if (replacement.has_value()) {
      auto replacement_channel = grpc::CreateChannel(
          replacement->endpoint, grpc::InsecureChannelCredentials());
      auto replacement_stub = infersched::v1::EngineControl::NewStub(
          replacement_channel);
      retry_completed = DispatchOne(*replacement_stub, 0, 2, 2, false);
      infersched::v1::DrainRequest drain_request;
      drain_request.set_engine_id(replacement->engine_id);
      infersched::v1::DrainReply drain_reply;
      grpc::ClientContext drain_context;
      const grpc::Status drain_status = replacement_stub->Drain(
          &drain_context, drain_request, &drain_reply);
      drained = drain_status.ok() && drain_reply.in_flight() == 0;
    }
    server->Shutdown();
    std::cout << "{\"first_attempt_failed\":" << std::boolalpha
              << first_attempt_failed << ",\"retry_completed\":"
              << retry_completed << ",\"drained\":" << drained << "}\n";
    return first_attempt_failed && retry_completed && drained ? EXIT_SUCCESS
                                                               : EXIT_FAILURE;
  }

  std::atomic<std::size_t> completed{0};
  std::vector<std::thread> clients;
  clients.reserve(requests);
  for (std::size_t index = 0; index < requests; ++index) {
    clients.emplace_back([&, index] {
      if (DispatchOne(*stub, index, 1, 1, false)) {
        completed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& client : clients) {
    client.join();
  }

  const bool stale_attempt_rejected = DispatchOne(*stub, 0, 1, 1, true);
  const bool stale_epoch_rejected = DispatchOne(*stub, requests + 1, 1, 0, true);
  infersched::v1::DrainRequest drain_request;
  drain_request.set_engine_id(engine->engine_id);
  infersched::v1::DrainReply drain_reply;
  grpc::ClientContext drain_context;
  const grpc::Status drain_status =
      stub->Drain(&drain_context, drain_request, &drain_reply);
  server->Shutdown();

  const bool success = completed.load(std::memory_order_relaxed) == requests &&
                       stale_attempt_rejected && stale_epoch_rejected &&
                       drain_status.ok() &&
                       drain_reply.in_flight() == 0;
  std::cout << "{\"requests\":" << requests << ",\"completed\":"
            << completed.load(std::memory_order_relaxed)
            << ",\"stale_attempt_rejected\":" << std::boolalpha
            << stale_attempt_rejected
            << ",\"stale_epoch_rejected\":" << stale_epoch_rejected
            << ",\"drained\":" << (drain_status.ok() &&
                                          drain_reply.in_flight() == 0)
            << "}\n";
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
