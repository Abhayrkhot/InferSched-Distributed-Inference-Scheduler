#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "infersched.grpc.pb.h"
#include "infersched/engine/continuous_engine.hpp"

namespace infersched::distributed {

struct EngineServiceOptions {
  // Maximum requests accepted but not yet terminal (queued + running).
  std::size_t queue_capacity{1024};
  // Fixed wall-clock delay added to every continuous-batching iteration.
  // Used by tests to hold work in flight long enough to inject failures.
  std::chrono::milliseconds step_delay{0};
  // Wall-clock nanoseconds slept per simulated nanosecond of cost-model time.
  // 0 runs the simulation as fast as possible; 1.0 paces it in real time.
  double time_scale{0.0};
  // Bound on remembered (request_id -> attempt) fences used to reject stale
  // or duplicate dispatches. Postgres remains the authoritative fence.
  std::size_t fence_memory{65536};
};

// Network wrapper around one ContinuousEngine. A single worker thread owns the
// engine and runs iterations back to back; Dispatch RPCs hand requests to it
// and new requests join the running batch at the next iteration boundary
// (iteration-level continuous batching across RPCs, not per-RPC batches).
class EngineService final : public v1::EngineControl::Service {
 public:
  explicit EngineService(engine::EngineConfig config,
                         EngineServiceOptions options = {});
  ~EngineService() override;
  EngineService(const EngineService&) = delete;
  EngineService& operator=(const EngineService&) = delete;

  grpc::Status Dispatch(grpc::ServerContext* context,
                        const v1::DispatchRequest* request,
                        grpc::ServerWriter<v1::DispatchEvent>* writer) override;
  grpc::Status Cancel(grpc::ServerContext* context,
                      const v1::CancelRequest* request,
                      v1::CancelReply* reply) override;
  grpc::Status Drain(grpc::ServerContext* context,
                     const v1::DrainRequest* request,
                     v1::DrainReply* reply) override;

  [[nodiscard]] v1::EngineCapacity Capacity() const;

  struct Stats {
    std::uint64_t accepted{};
    std::uint64_t rejected{};
    std::uint64_t completed{};
    std::uint64_t failed{};
    std::uint64_t cancelled{};
    std::size_t peak_running{};
  };
  [[nodiscard]] Stats stats() const;

 private:
  struct Job {
    v1::DispatchRequest dispatch;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<v1::DispatchEvent> events;
    bool done{};
    std::atomic<bool> superseded{};
  };

  [[nodiscard]] std::string ValidateAndReserve(
      const v1::DispatchRequest& request, const std::shared_ptr<Job>& job);
  void RememberAttempt(const std::string& request_id, std::uint64_t attempt);
  void RequestCancel(const std::string& request_id, std::string reason);
  void WorkerLoop();
  void Publish(Job& job, v1::DispatchEvent event, bool terminal);
  void FinishJob(const std::shared_ptr<Job>& job, const engine::EngineResult& result);
  [[nodiscard]] engine::EngineRequest ToEngineRequest(
      const v1::InferenceRequest& request) const;
  [[nodiscard]] static v1::ProgressEvent BaseEvent(const Job& job);

  const engine::EngineConfig config_;
  const EngineServiceOptions options_;

  // Guarded by mutex_: hand-off state between RPC threads and the worker.
  mutable std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable drained_;
  std::deque<std::shared_ptr<Job>> incoming_;
  std::deque<std::pair<std::string, std::string>> cancels_;
  std::unordered_map<std::int32_t, std::uint64_t> partition_epochs_;
  std::unordered_map<std::string, std::uint64_t> request_attempts_;
  std::deque<std::string> attempt_order_;
  std::unordered_map<std::string, std::shared_ptr<Job>> live_jobs_;
  std::size_t outstanding_{};
  bool draining_{};
  bool stop_{};
  Stats stats_;

  // Published by the worker for Capacity()/heartbeats.
  std::atomic<std::size_t> free_kv_blocks_;
  std::atomic<std::size_t> running_seqs_{};
  std::atomic<std::size_t> waiting_seqs_{};

  // Owned by the worker thread only.
  engine::ContinuousEngine engine_;
  std::thread worker_;
};

}  // namespace infersched::distributed
