#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <grpcpp/grpcpp.h>

#include "infersched.grpc.pb.h"

namespace infersched::distributed {

struct EngineRecord {
  std::string engine_id;
  std::string incarnation_id;
  std::string endpoint;
  v1::EngineCapacity capacity;
  std::chrono::steady_clock::time_point last_heartbeat;
};

class RouterService final : public v1::RouterControl::Service {
 public:
  explicit RouterService(
      std::chrono::milliseconds lease_ttl = std::chrono::milliseconds(1000))
      : lease_ttl_(lease_ttl) {}
  grpc::Status Register(grpc::ServerContext* context,
                        const v1::RegisterRequest* request,
                        v1::RegisterReply* reply) override;
  grpc::Status Heartbeat(grpc::ServerContext* context,
                         const v1::HeartbeatRequest* request,
                         v1::HeartbeatReply* reply) override;

  [[nodiscard]] std::optional<EngineRecord> WaitForEngine(
      std::chrono::milliseconds timeout);
  [[nodiscard]] std::optional<EngineRecord> WaitForNewIncarnation(
      std::string_view previous_incarnation,
      std::chrono::milliseconds timeout);

 private:
  void PruneExpiredLocked();

  std::chrono::milliseconds lease_ttl_;
  std::mutex mutex_;
  std::condition_variable registered_;
  std::unordered_map<std::string, EngineRecord> engines_;
};

}  // namespace infersched::distributed
