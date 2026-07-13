#include "infersched/distributed/router_service.hpp"

namespace infersched::distributed {

grpc::Status RouterService::Register(grpc::ServerContext* context,
                                     const v1::RegisterRequest* request,
                                     v1::RegisterReply* reply) {
  static_cast<void>(context);
  if (request->engine_id().empty() || request->incarnation_id().empty() ||
      request->endpoint().empty()) {
    reply->set_accepted(false);
    return grpc::Status::OK;
  }
  {
    std::lock_guard lock(mutex_);
    engines_.insert_or_assign(
        request->engine_id(),
        EngineRecord{.engine_id = request->engine_id(),
                     .incarnation_id = request->incarnation_id(),
                     .endpoint = request->endpoint(),
                     .capacity = request->capacity(),
                     .last_heartbeat = std::chrono::steady_clock::now()});
  }
  reply->set_accepted(true);
  reply->set_assigned_lease_ttl_ms(
      static_cast<std::uint64_t>(lease_ttl_.count()));
  registered_.notify_all();
  return grpc::Status::OK;
}

grpc::Status RouterService::Heartbeat(grpc::ServerContext* context,
                                      const v1::HeartbeatRequest* request,
                                      v1::HeartbeatReply* reply) {
  static_cast<void>(context);
  std::lock_guard lock(mutex_);
  PruneExpiredLocked();
  const auto iterator = engines_.find(request->engine_id());
  if (iterator == engines_.end() ||
      iterator->second.incarnation_id != request->incarnation_id()) {
    reply->set_lease_ok(false);
    return grpc::Status::OK;
  }
  iterator->second.capacity = request->capacity();
  iterator->second.last_heartbeat = std::chrono::steady_clock::now();
  reply->set_lease_ok(true);
  return grpc::Status::OK;
}

std::optional<EngineRecord> RouterService::WaitForEngine(
    const std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  const auto has_live_engine = [&] {
    PruneExpiredLocked();
    return !engines_.empty();
  };
  if (!registered_.wait_for(lock, timeout, has_live_engine)) {
    return std::nullopt;
  }
  return engines_.begin()->second;
}

std::optional<EngineRecord> RouterService::WaitForNewIncarnation(
    const std::string_view previous_incarnation,
    const std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  const auto has_new = [&] {
    PruneExpiredLocked();
    for (const auto& [engine_id, engine] : engines_) {
      static_cast<void>(engine_id);
      if (engine.incarnation_id != previous_incarnation) {
        return true;
      }
    }
    return false;
  };
  if (!registered_.wait_for(lock, timeout, has_new)) {
    return std::nullopt;
  }
  for (const auto& [engine_id, engine] : engines_) {
    static_cast<void>(engine_id);
    if (engine.incarnation_id != previous_incarnation) {
      return engine;
    }
  }
  return std::nullopt;
}

void RouterService::PruneExpiredLocked() {
  const auto now = std::chrono::steady_clock::now();
  std::erase_if(engines_, [&](const auto& entry) {
    return now - entry.second.last_heartbeat >= lease_ttl_;
  });
}

}  // namespace infersched::distributed
