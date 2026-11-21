#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "infersched/distributed/engine_service.hpp"
#include "infersched.grpc.pb.h"
#ifdef INFERSCHED_HAVE_LLAMA
#include "infersched/engine/gpu_monitor.hpp"
#include "infersched/engine/llama_executor.hpp"
#endif

namespace {

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop.store(true, std::memory_order_release); }

std::string Argument(const int argc, char** argv, const std::string& name,
                     const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      return argv[index + 1];
    }
  }
  return fallback;
}

std::vector<std::string> SplitCommas(const std::string& value) {
  std::vector<std::string> parts;
  std::stringstream stream(value);
  std::string part;
  while (std::getline(stream, part, ',')) {
    if (!part.empty()) {
      parts.push_back(part);
    }
  }
  return parts;
}

// One RouterControl session: register, then heartbeat; re-register whenever
// the Router reports the lease lost or becomes unreachable (Router restart).
class RouterSession {
 public:
  RouterSession(std::string router, std::string engine_id, std::string incarnation,
                std::string endpoint, infersched::distributed::EngineService& service)
      : router_(std::move(router)), engine_id_(std::move(engine_id)),
        incarnation_(std::move(incarnation)), endpoint_(std::move(endpoint)),
        service_(service),
        stub_(infersched::v1::RouterControl::NewStub(
            grpc::CreateChannel(router_, grpc::InsecureChannelCredentials()))) {}

  bool Register() {
    infersched::v1::RegisterRequest request;
    request.set_engine_id(engine_id_);
    request.set_incarnation_id(incarnation_);
    request.set_endpoint(endpoint_);
    request.add_models("benchmark-model");
    *request.mutable_capacity() = service_.Capacity();
    infersched::v1::RegisterReply reply;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(200));
    registered_ = stub_->Register(&context, request, &reply).ok() && reply.accepted();
    return registered_;
  }

  void Tick() {
    if (!registered_) {
      static_cast<void>(Register());
      return;
    }
    infersched::v1::HeartbeatRequest request;
    request.set_engine_id(engine_id_);
    request.set_incarnation_id(incarnation_);
    *request.mutable_capacity() = service_.Capacity();
    infersched::v1::HeartbeatReply reply;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(200));
    const grpc::Status status = stub_->Heartbeat(&context, request, &reply);
    if (!status.ok() || !reply.lease_ok()) {
      static_cast<void>(Register());
    }
  }

  [[nodiscard]] bool registered() const { return registered_; }
  [[nodiscard]] const std::string& router() const { return router_; }

 private:
  std::string router_;
  std::string engine_id_;
  std::string incarnation_;
  std::string endpoint_;
  infersched::distributed::EngineService& service_;
  std::unique_ptr<infersched::v1::RouterControl::Stub> stub_;
  bool registered_{};
};

}  // namespace

int main(const int argc, char** argv) {
  const std::string listen = Argument(argc, argv, "--listen", "127.0.0.1:50052");
  const auto routers = SplitCommas(Argument(argc, argv, "--router", "127.0.0.1:50051"));
  const std::string engine_id = Argument(argc, argv, "--engine-id", "engine-1");
  const std::string incarnation =
      Argument(argc, argv, "--incarnation", "incarnation-1");
  const auto heartbeat_interval = std::chrono::milliseconds(std::stoll(
      Argument(argc, argv, "--heartbeat-ms", "200")));

  infersched::engine::EngineConfig config;
  config.kv_blocks = std::stoull(Argument(argc, argv, "--kv-blocks", "4096"));
  config.max_batch_sequences =
      std::stoull(Argument(argc, argv, "--max-batch-sequences", "64"));
  config.max_prefill_sequences_per_iteration =
      std::stoull(Argument(argc, argv, "--max-prefill-sequences", "64"));
  config.prefix_cache_entries =
      std::stoull(Argument(argc, argv, "--prefix-cache-entries", "1024"));
  config.result_cache_entries =
      std::stoull(Argument(argc, argv, "--result-cache-entries", "1024"));
  config.max_batch_tokens =
      std::stoull(Argument(argc, argv, "--max-batch-tokens", "4096"));
  config.prefix_caching = Argument(argc, argv, "--prefix-cache", "on") != "off";
  const std::string policy = Argument(argc, argv, "--policy", "continuous");
  if (policy != "continuous" && policy != "static") {
    std::cerr << "--policy must be continuous or static\n";
    return EXIT_FAILURE;
  }
  config.admission = policy == "static" ? infersched::engine::AdmissionPolicy::kStatic
                                        : infersched::engine::AdmissionPolicy::kContinuous;
  // Phenomenological cost model (microseconds). Defaults keep the historical
  // engine model; experiments pass a heavier one with --time-scale 1.
  const auto micros = [&](const std::string& name, const std::string& fallback) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double, std::micro>(std::stod(Argument(argc, argv, name, fallback))));
  };
  config.cost_model.prefill_fixed = micros("--prefill-fixed-us", "10");
  config.cost_model.prefill_per_token = micros("--prefill-per-token-us", "1");
  config.cost_model.decode_fixed = micros("--decode-fixed-us", "5");
  config.cost_model.decode_per_sequence = micros("--decode-per-seq-us", "2");
  config.cost_model.jitter_basis_points =
      static_cast<std::uint32_t>(std::stoul(Argument(argc, argv, "--jitter-bp", "0")));

  infersched::distributed::EngineServiceOptions options;
  options.queue_capacity = std::stoull(Argument(argc, argv, "--queue-capacity", "1024"));
  // Fixed wall-clock delay per continuous-batching iteration (test hook).
  options.step_delay = std::chrono::milliseconds(
      std::stoll(Argument(argc, argv, "--batch-delay-ms", "0")));
  // Wall-clock pacing of the cost model: 1.0 = simulated time is real time.
  options.time_scale = std::stod(Argument(argc, argv, "--time-scale", "0"));
  options.cancellation = Argument(argc, argv, "--cancellation", "on") != "off";

  // Executor: the cost model (default) or a real model through llama.cpp.
  const std::string backend = Argument(argc, argv, "--backend", "cost-model");
  std::unique_ptr<infersched::engine::StepExecutor> executor;
#ifdef INFERSCHED_HAVE_LLAMA
  std::unique_ptr<infersched::engine::GpuMonitor> gpu;
#endif
  if (backend == "llama") {
#ifdef INFERSCHED_HAVE_LLAMA
    infersched::engine::LlamaExecutorConfig llama;
    llama.model_path = Argument(argc, argv, "--model", "");
    llama.gpu_layers = std::stoi(Argument(argc, argv, "--gpu-layers", "999"));
    llama.kv_tokens = config.kv_blocks * config.kv_block_size_tokens;
    llama.block_size_tokens = config.kv_block_size_tokens;
    llama.max_sequences = config.max_batch_sequences;
    llama.holder_sequences = std::stoull(Argument(argc, argv, "--holder-sequences", "32"));
    llama.max_batch_tokens = config.max_batch_tokens;
    llama.ubatch_tokens = std::stoull(Argument(argc, argv, "--ubatch-tokens", "512"));
    llama.flash_attention = Argument(argc, argv, "--flash-attn", "on") != "off";
    llama.threads = std::stoi(Argument(argc, argv, "--threads", "4"));
    llama.backend_sampling = Argument(argc, argv, "--backend-sampling", "off") == "on";
    try {
      executor = std::make_unique<infersched::engine::LlamaExecutor>(llama);
    } catch (const std::exception& error) {
      std::cerr << "llama backend: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
    options.time_scale = 0.0;  // Real compute takes real time; no pacing.
    gpu = std::make_unique<infersched::engine::GpuMonitor>();
    gpu->Start();
#else
    std::cerr << "--backend llama needs a build with -DINFERSCHED_WITH_LLAMA=ON\n";
    return EXIT_FAILURE;
#endif
  } else if (backend != "cost-model") {
    std::cerr << "--backend must be cost-model or llama\n";
    return EXIT_FAILURE;
  }

  infersched::distributed::EngineService service(config, options, std::move(executor));
  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server) {
    std::cerr << "failed to start EngineControl at " << listen << '\n';
    return EXIT_FAILURE;
  }

  std::vector<std::unique_ptr<RouterSession>> sessions;
  for (const auto& router : routers) {
    sessions.push_back(std::make_unique<RouterSession>(router, engine_id, incarnation,
                                                       listen, service));
  }
  // Wait (bounded) until at least one Router accepts the registration.
  bool any_registered = false;
  for (int attempt = 0; attempt < 100 && !any_registered; ++attempt) {
    for (auto& session : sessions) {
      any_registered = session->Register() || any_registered;
    }
    if (!any_registered) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  if (!any_registered) {
    std::cerr << "failed to register with any RouterControl\n";
    server->Shutdown();
    return EXIT_FAILURE;
  }

  std::signal(SIGTERM, HandleSignal);
  std::signal(SIGINT, HandleSignal);
  std::thread heartbeat([&] {
    while (!g_stop.load(std::memory_order_acquire)) {
      for (auto& session : sessions) {
        session->Tick();
      }
      std::this_thread::sleep_for(heartbeat_interval);
    }
  });

  std::cout << "EngineControl listening at " << listen << " for " << routers.size()
            << " router(s)" << std::endl;
  while (!g_stop.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  heartbeat.join();
  server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
  // Last stdout line: compute-efficiency counters for experiments.
  std::string utilization = service.UtilizationJson(engine_id);
#ifdef INFERSCHED_HAVE_LLAMA
  if (gpu != nullptr && gpu->available()) {
    const auto window = service.ActiveWindow();
    const auto summary = window.has_value() ? gpu->Summarize(window->first, window->second)
                                            : gpu->Stop();
    utilization.pop_back();  // Extend the JSON object with GPU telemetry.
    utilization += ",\"gpu_name\":\"" + summary.name + "\",\"gpu_samples\":" +
                   std::to_string(summary.samples) + ",\"gpu_util_pct\":" +
                   std::to_string(summary.mean_utilization_pct) +
                   ",\"gpu_mem_util_pct\":" +
                   std::to_string(summary.mean_memory_utilization_pct) +
                   ",\"gpu_power_w\":" + std::to_string(summary.mean_power_w) +
                   ",\"gpu_peak_used_mib\":" +
                   std::to_string(summary.peak_used_bytes / (1024 * 1024)) + "}";
  }
#endif
  std::cout << utilization << std::endl;
  return EXIT_SUCCESS;
}
