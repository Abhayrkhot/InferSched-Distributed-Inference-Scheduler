#pragma once

#include <atomic>
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

class EngineService final : public v1::EngineControl::Service {
 public:
  explicit EngineService(engine::EngineConfig config,
                         std::size_t queue_capacity = 1024,
                         std::chrono::milliseconds processing_delay =
                             std::chrono::milliseconds::zero());
  ~EngineService() override;

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

 private:
  struct Job {
    v1::DispatchRequest dispatch;
    std::mutex mutex;
    std::condition_variable ready;
    bool done{};
    std::atomic<bool> superseded{};
    engine::EngineResult result;
  };

  [[nodiscard]] std::string ValidateAndReserve(
      const v1::DispatchRequest& request,
      const std::shared_ptr<Job>& job);
  void WorkerLoop();
  static engine::EngineRequest ToEngineRequest(const v1::InferenceRequest& request);
  static void WriteProgress(const Job& job,
                            grpc::ServerWriter<v1::DispatchEvent>& writer);

  const std::size_t queue_capacity_;
  const std::chrono::milliseconds processing_delay_;
  mutable std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable drained_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::unordered_map<std::int32_t, std::uint64_t> partition_epochs_;
  std::unordered_map<std::string, std::uint64_t> request_attempts_;
  std::unordered_map<std::string, std::weak_ptr<Job>> live_jobs_;
  bool draining_{};
  bool stop_{};
  std::size_t running_jobs_{};
  engine::ContinuousEngine engine_;
  const std::size_t total_kv_blocks_;
  std::atomic<std::size_t> free_kv_blocks_;
  std::thread worker_;
};

}  // namespace infersched::distributed
