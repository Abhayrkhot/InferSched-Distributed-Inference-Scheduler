#include "infersched/distributed/engine_service.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace infersched::distributed {

EngineService::EngineService(engine::EngineConfig config,
                             const std::size_t queue_capacity,
                             const std::chrono::milliseconds processing_delay)
    : queue_capacity_(queue_capacity), processing_delay_(processing_delay),
      engine_(config),
      total_kv_blocks_(config.kv_blocks), free_kv_blocks_(config.kv_blocks),
      worker_(&EngineService::WorkerLoop, this) {}

EngineService::~EngineService() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  work_ready_.notify_all();
  worker_.join();
}

grpc::Status EngineService::Dispatch(
    grpc::ServerContext* context, const v1::DispatchRequest* request,
    grpc::ServerWriter<v1::DispatchEvent>* writer) {
  static_cast<void>(context);
  auto job = std::make_shared<Job>();
  job->dispatch = *request;
  const std::string rejection = ValidateAndReserve(*request, job);

  v1::DispatchEvent acknowledgement;
  auto* ack = acknowledgement.mutable_ack();
  if (!rejection.empty()) {
    ack->set_status(v1::DispatchAck::REJECTED);
    ack->set_reason(rejection);
    static_cast<void>(writer->Write(acknowledgement));
    return grpc::Status::OK;
  }
  ack->set_status(v1::DispatchAck::ACCEPTED);
  if (!writer->Write(acknowledgement)) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "ack stream closed");
  }

  std::unique_lock lock(job->mutex);
  job->ready.wait(lock, [&] { return job->done; });
  lock.unlock();
  WriteProgress(*job, *writer);
  return grpc::Status::OK;
}

grpc::Status EngineService::Cancel(grpc::ServerContext* context,
                                   const v1::CancelRequest* request,
                                   v1::CancelReply* reply) {
  static_cast<void>(context);
  std::lock_guard lock(mutex_);
  const auto iterator = live_jobs_.find(request->request_id());
  const auto job = iterator == live_jobs_.end() ? nullptr : iterator->second.lock();
  if (job && job->dispatch.fence().ownership_epoch() ==
                 request->fence().ownership_epoch() &&
      job->dispatch.fence().attempt_id() == request->fence().attempt_id()) {
    job->superseded.store(true, std::memory_order_release);
    reply->set_accepted(true);
  }
  return grpc::Status::OK;
}

grpc::Status EngineService::Drain(grpc::ServerContext* context,
                                  const v1::DrainRequest* request,
                                  v1::DrainReply* reply) {
  static_cast<void>(context);
  static_cast<void>(request);
  std::unique_lock lock(mutex_);
  draining_ = true;
  drained_.wait(lock, [&] { return queue_.empty() && running_jobs_ == 0; });
  reply->set_in_flight(0);
  return grpc::Status::OK;
}

v1::EngineCapacity EngineService::Capacity() const {
  std::lock_guard lock(mutex_);
  v1::EngineCapacity capacity;
  capacity.set_total_kv_blocks(
      static_cast<std::uint32_t>(total_kv_blocks_));
  capacity.set_free_kv_blocks(
      static_cast<std::uint32_t>(free_kv_blocks_.load(std::memory_order_relaxed)));
  capacity.set_running_seqs(static_cast<std::uint32_t>(running_jobs_));
  capacity.set_waiting_seqs(static_cast<std::uint32_t>(queue_.size()));
  capacity.set_batch_headroom(static_cast<std::uint32_t>(
      queue_capacity_ > queue_.size() ? queue_capacity_ - queue_.size() : 0));
  return capacity;
}

std::string EngineService::ValidateAndReserve(
    const v1::DispatchRequest& request, const std::shared_ptr<Job>& job) {
  std::lock_guard lock(mutex_);
  if (draining_) {
    return "draining";
  }
  if (queue_.size() >= queue_capacity_) {
    return "queue_full";
  }
  const auto partition = request.fence().partition_id();
  const auto epoch = request.fence().ownership_epoch();
  const auto current_epoch = partition_epochs_.find(partition);
  if (current_epoch != partition_epochs_.end() && epoch < current_epoch->second) {
    return "stale_ownership_epoch";
  }
  partition_epochs_[partition] = epoch;

  const auto attempt = request.fence().attempt_id();
  const auto current_attempt = request_attempts_.find(request.request().request_id());
  if (current_attempt != request_attempts_.end() &&
      attempt <= current_attempt->second) {
    return "stale_or_duplicate_attempt";
  }
  const auto live = live_jobs_.find(request.request().request_id());
  if (live != live_jobs_.end() && !live->second.expired()) {
    // ContinuousEngine requires one live sequence per request id. P4 rejects an
    // overlapping retry without advancing its fence; P5 may reroute it to a
    // different Engine or retry after the current attempt becomes terminal.
    return "request_already_in_flight";
  }
  request_attempts_[request.request().request_id()] = attempt;
  live_jobs_[request.request().request_id()] = job;
  queue_.push_back(job);
  work_ready_.notify_one();
  return {};
}

void EngineService::WorkerLoop() {
  while (true) {
    std::vector<std::shared_ptr<Job>> batch;
    {
      std::unique_lock lock(mutex_);
      work_ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty()) {
        return;
      }
      // Briefly coalesce concurrent RPCs into one continuous-batching run.
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      lock.lock();
      while (!queue_.empty()) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
      running_jobs_ += batch.size();
    }

    const std::size_t result_offset = engine_.results().size();
    for (const auto& job : batch) {
      engine_.Submit(ToEngineRequest(job->dispatch.request()));
    }
    if (processing_delay_ > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(processing_delay_);
    }
    engine_.RunUntilIdle();
    free_kv_blocks_.store(engine_.kv_cache().free_blocks(),
                          std::memory_order_relaxed);

    std::unordered_map<std::string, engine::EngineResult> new_results;
    for (std::size_t index = result_offset; index < engine_.results().size(); ++index) {
      new_results.insert_or_assign(engine_.results()[index].request_id,
                                   engine_.results()[index]);
    }
    for (const auto& job : batch) {
      std::lock_guard job_lock(job->mutex);
      const auto result = new_results.find(job->dispatch.request().request_id());
      if (result == new_results.end()) {
        job->result = engine::EngineResult{
            .request_id = job->dispatch.request().request_id(),
            .succeeded = false,
            .failure_reason = "missing_engine_result"};
      } else {
        job->result = result->second;
      }
      if (job->superseded.load(std::memory_order_acquire)) {
        job->result.succeeded = false;
        job->result.failure_reason = "superseded";
      }
      job->done = true;
      job->ready.notify_all();
    }

    {
      std::lock_guard lock(mutex_);
      running_jobs_ -= batch.size();
      for (const auto& job : batch) {
        const auto iterator = live_jobs_.find(job->dispatch.request().request_id());
        if (iterator != live_jobs_.end() && iterator->second.lock() == job) {
          live_jobs_.erase(iterator);
        }
      }
      if (queue_.empty() && running_jobs_ == 0) {
        drained_.notify_all();
      }
    }
  }
}

engine::EngineRequest EngineService::ToEngineRequest(
    const v1::InferenceRequest& request) {
  return engine::EngineRequest{
      .request_id = request.request_id(),
      .model_revision = request.model_revision(),
      .tokenizer_revision = request.tokenizer_revision(),
      .prompt_hash = request.prompt_hash(),
      .prompt_tokens = request.prompt_tokens(),
      .max_output_tokens = request.sampling().max_output_tokens(),
      .priority = request.priority(),
      .sampling_seed = request.sampling().seed(),
      .temperature_micros = static_cast<std::uint64_t>(
          std::max(0.0, request.sampling().temperature()) * 1'000'000.0),
      .top_p_micros = static_cast<std::uint64_t>(
          std::max(0.0, request.sampling().top_p()) * 1'000'000.0)};
}

void EngineService::WriteProgress(
    const Job& job, grpc::ServerWriter<v1::DispatchEvent>& writer) {
  v1::DispatchEvent prefill;
  auto* prefill_event = prefill.mutable_progress();
  prefill_event->set_kind(v1::ProgressEvent::PREFILL_DONE);
  prefill_event->set_request_id(job.dispatch.request().request_id());
  prefill_event->set_trace_id(job.dispatch.request().trace_id());
  prefill_event->set_span_id(job.dispatch.request().span_id());
  *prefill_event->mutable_fence() = job.dispatch.fence();
  static_cast<void>(writer.Write(prefill));

  v1::DispatchEvent terminal;
  auto* event = terminal.mutable_progress();
  event->set_request_id(job.dispatch.request().request_id());
  event->set_trace_id(job.dispatch.request().trace_id());
  event->set_span_id(job.dispatch.request().span_id());
  *event->mutable_fence() = job.dispatch.fence();
  if (job.result.succeeded) {
    event->set_kind(v1::ProgressEvent::COMPLETED);
    event->set_tokens_generated(job.dispatch.request().sampling().max_output_tokens());
    event->set_result_ref(job.result.output);
    event->set_prefix_cache_hit(job.result.prefix_cache_hit);
    event->set_result_cache_hit(job.result.result_cache_hit);
  } else {
    event->set_kind(v1::ProgressEvent::FAILED);
    event->set_failure_reason(job.result.failure_reason);
    event->set_retryable(job.result.failure_reason == "superseded");
  }
  static_cast<void>(writer.Write(terminal));
}

}  // namespace infersched::distributed
