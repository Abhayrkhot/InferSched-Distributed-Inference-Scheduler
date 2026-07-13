#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>

#include "infersched/distributed/engine_service.hpp"
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

}  // namespace

int main(const int argc, char** argv) {
  const std::string listen = Argument(argc, argv, "--listen", "127.0.0.1:50052");
  const std::string router = Argument(argc, argv, "--router", "127.0.0.1:50051");
  const std::string engine_id = Argument(argc, argv, "--engine-id", "engine-1");
  const std::string incarnation =
      Argument(argc, argv, "--incarnation", "incarnation-1");
  const auto delay = std::chrono::milliseconds(std::stoll(
      Argument(argc, argv, "--batch-delay-ms", "0")));

  infersched::engine::EngineConfig config;
  config.kv_blocks = 4096;
  config.max_batch_sequences = 64;
  config.max_prefill_sequences_per_iteration = 64;
  infersched::distributed::EngineService service(config, 1024, delay);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server) {
    std::cerr << "failed to start EngineControl at " << listen << '\n';
    return EXIT_FAILURE;
  }

  auto channel = grpc::CreateChannel(router, grpc::InsecureChannelCredentials());
  auto stub = infersched::v1::RouterControl::NewStub(channel);
  bool registered = false;
  for (int attempt = 0; attempt < 100 && !registered; ++attempt) {
    infersched::v1::RegisterRequest request;
    request.set_engine_id(engine_id);
    request.set_incarnation_id(incarnation);
    request.set_endpoint(listen);
    request.add_models("benchmark-model");
    *request.mutable_capacity() = service.Capacity();
    infersched::v1::RegisterReply reply;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(200));
    const grpc::Status status = stub->Register(&context, request, &reply);
    registered = status.ok() && reply.accepted();
    if (!registered) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  if (!registered) {
    std::cerr << "failed to register with RouterControl at " << router << '\n';
    server->Shutdown();
    return EXIT_FAILURE;
  }

  std::atomic<bool> stop{false};
  std::thread heartbeat([&] {
    while (!stop.load(std::memory_order_acquire)) {
      infersched::v1::HeartbeatRequest request;
      request.set_engine_id(engine_id);
      request.set_incarnation_id(incarnation);
      *request.mutable_capacity() = service.Capacity();
      infersched::v1::HeartbeatReply reply;
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() +
                           std::chrono::milliseconds(200));
      static_cast<void>(stub->Heartbeat(&context, request, &reply));
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  std::cout << "EngineControl listening at " << listen << '\n';
  server->Wait();
  stop.store(true, std::memory_order_release);
  heartbeat.join();
  return EXIT_SUCCESS;
}
