#include "infersched/distributed/engine_service.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>

#include "infersched/engine/prompt_blocks.hpp"
#include "infersched/util/json.hpp"

namespace infersched::distributed {
namespace {

constexpr auto kStreamPollInterval = std::chrono::milliseconds(50);
constexpr auto kStallWait = std::chrono::milliseconds(1);

bool RetryableFailure(const std::string& reason) {
  return reason == "superseded" || reason == "client_cancelled" ||
         reason == "engine_stopping";
}

}  // namespace

EngineService::EngineService(engine::EngineConfig config,
                             EngineServiceOptions options,
                             std::unique_ptr<engine::StepExecutor> executor)
    : config_(config), options_(options), free_kv_blocks_(config.kv_blocks),
      engine_(config, std::move(executor)) {
  backend_ = std::string(engine_.executor().name());
  measured_ = engine_.executor().measured();
  worker_ = std::thread(&EngineService::WorkerLoop, this);
}

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
  const std::string& request_id = request->request().request_id();
  if (!writer->Write(acknowledgement)) {
    RequestCancel(request_id, "client_cancelled");
    return grpc::Status(grpc::StatusCode::CANCELLED, "ack stream closed");
  }

  // Stream progress at the Engine's own cadence until the attempt is terminal.
  while (true) {
    std::deque<v1::DispatchEvent> batch;
    bool done = false;
    {
      std::unique_lock lock(job->mutex);
      job->changed.wait_for(lock, kStreamPollInterval,
                            [&] { return job->done || !job->events.empty(); });
      batch.swap(job->events);
      done = job->done;
    }
    for (const auto& event : batch) {
      if (!writer->Write(event)) {
        RequestCancel(request_id, "client_cancelled");
        return grpc::Status(grpc::StatusCode::CANCELLED, "progress stream closed");
      }
    }
    if (done) {
      return grpc::Status::OK;
    }
    if (context != nullptr && context->IsCancelled()) {
      // Router deadline or explicit TryCancel: free the sequence's KV now
      // instead of finishing work nobody will accept.
      RequestCancel(request_id, "client_cancelled");
      return grpc::Status(grpc::StatusCode::CANCELLED, "dispatch cancelled");
    }
  }
}

grpc::Status EngineService::Cancel(grpc::ServerContext* context,
                                   const v1::CancelRequest* request,
                                   v1::CancelReply* reply) {
  static_cast<void>(context);
  std::lock_guard lock(mutex_);
  const auto iterator = live_jobs_.find(request->request_id());
  if (iterator != live_jobs_.end()) {
    const auto& fence = iterator->second->dispatch.fence();
    if (fence.ownership_epoch() == request->fence().ownership_epoch() &&
        fence.attempt_id() == request->fence().attempt_id()) {
      iterator->second->superseded.store(true, std::memory_order_release);
      if (options_.cancellation) {
        cancels_.emplace_back(request->request_id(), "superseded");
      }
      reply->set_accepted(true);
      work_ready_.notify_one();
    }
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
  drained_.wait(lock, [&] { return outstanding_ == 0; });
  reply->set_in_flight(0);
  return grpc::Status::OK;
}

v1::EngineCapacity EngineService::Capacity() const {
  std::size_t outstanding = 0;
  std::size_t incoming = 0;
  {
    std::lock_guard lock(mutex_);
    outstanding = outstanding_;
    incoming = incoming_.size();
  }
  v1::EngineCapacity capacity;
  capacity.set_total_kv_blocks(static_cast<std::uint32_t>(config_.kv_blocks));
  capacity.set_free_kv_blocks(static_cast<std::uint32_t>(
      free_kv_blocks_.load(std::memory_order_relaxed)));
  capacity.set_running_seqs(static_cast<std::uint32_t>(
      running_seqs_.load(std::memory_order_relaxed)));
  capacity.set_waiting_seqs(static_cast<std::uint32_t>(
      waiting_seqs_.load(std::memory_order_relaxed) + incoming));
  capacity.set_batch_headroom(static_cast<std::uint32_t>(
      options_.queue_capacity > outstanding ? options_.queue_capacity - outstanding
                                            : 0));
  return capacity;
}

EngineService::Stats EngineService::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

std::string EngineService::UtilizationJson(const std::string& engine_id) const {
  std::lock_guard lock(mutex_);
  const auto& engine = engine_stats_;
  const auto ms = [](const std::chrono::nanoseconds value) {
    return std::to_string(std::chrono::duration<double, std::milli>(value).count());
  };
  double window_ms = 0.0;
  if (first_accept_.has_value() && last_finish_.has_value()) {
    window_ms = std::chrono::duration<double, std::milli>(*last_finish_ - *first_accept_)
                    .count();
  }
  // Wall-clock busy time is modeled time paced by time_scale (plus any fixed
  // per-iteration delay, which is not modeled compute and is excluded). A
  // real-model executor measures it directly.
  const double scale = measured_ ? 1.0 : options_.time_scale;
  const double busy_wall_ms =
      std::chrono::duration<double, std::milli>(engine.simulated_busy_time).count() * scale;
  // Output that reached a live client. With cancellation off, abandoned
  // attempts still complete inside the engine; they are not useful work.
  const std::uint64_t useful =
      engine.useful_output_tokens > stats_.abandoned_output_tokens
          ? engine.useful_output_tokens - stats_.abandoned_output_tokens
          : 0;
  const double busy_s = std::chrono::duration<double>(engine.simulated_busy_time).count();
  util::JsonObject json;
  json.Add("engine_id", engine_id)
      .Add("backend", backend_)
      .Add("measured", measured_)
      .Add("policy", config_.admission == engine::AdmissionPolicy::kStatic ? "static"
                                                                           : "continuous")
      .Add("cancellation", options_.cancellation)
      .Add("completed", stats_.completed)
      .Add("failed", stats_.failed)
      .Add("cancelled", stats_.cancelled)
      .Add("decode_iterations", engine.iterations)
      .Add("decode_token_slots", engine.decode_token_slots)
      .Add("completed_output_tokens", engine.useful_output_tokens)
      .Add("useful_output_tokens", useful)
      .Add("wasted_decode_tokens", engine.wasted_decode_tokens)
      .Add("abandoned_output_tokens", stats_.abandoned_output_tokens)
      .Add("prefill_tokens_computed", engine.prefill_tokens_computed)
      .Add("prefill_tokens_reused", engine.prefill_tokens_reused)
      .Add("wasted_prefill_tokens", engine.wasted_prefill_tokens)
      .Add("preemptions", engine.preemptions)
      .AddRaw("mean_batch_occupancy", std::to_string(engine.mean_batch_occupancy()))
      .Add("peak_running", stats_.peak_running)
      // busy_ms is measured with a real-model backend ("measured": true) and
      // modeled otherwise; the modeled_* names are kept for the P9 scripts.
      .AddRaw("busy_ms", ms(engine.simulated_busy_time))
      .AddRaw("modeled_busy_ms", ms(engine.simulated_busy_time))
      .AddRaw("modeled_prefill_ms", ms(engine.simulated_prefill_time))
      .AddRaw("modeled_decode_ms", ms(engine.simulated_decode_time))
      .AddRaw("useful_tokens_per_busy_s",
              std::to_string(busy_s > 0.0 ? static_cast<double>(useful) / busy_s : 0.0))
      .AddRaw("time_scale", std::to_string(options_.time_scale))
      .AddRaw("window_ms", std::to_string(window_ms))
      .AddRaw("busy_fraction",
              std::to_string(window_ms > 0.0 && scale > 0.0
                                 ? busy_wall_ms / window_ms
                                 : 0.0));
  return json.str();
}

std::optional<std::pair<std::chrono::steady_clock::time_point,
                        std::chrono::steady_clock::time_point>>
EngineService::ActiveWindow() const {
  std::lock_guard lock(mutex_);
  if (!first_accept_.has_value() || !last_finish_.has_value()) {
    return std::nullopt;
  }
  return std::make_pair(*first_accept_, *last_finish_);
}

std::string EngineService::ValidateAndReserve(
    const v1::DispatchRequest& request, const std::shared_ptr<Job>& job) {
  std::lock_guard lock(mutex_);
  const auto reject = [&](std::string reason) {
    ++stats_.rejected;
    return reason;
  };
  if (draining_ || stop_) {
    return reject("draining");
  }
  if (outstanding_ >= options_.queue_capacity) {
    return reject("queue_full");
  }
  const auto partition = request.fence().partition_id();
  const auto epoch = request.fence().ownership_epoch();
  const auto current_epoch = partition_epochs_.find(partition);
  if (current_epoch != partition_epochs_.end() && epoch < current_epoch->second) {
    return reject("stale_ownership_epoch");
  }
  partition_epochs_[partition] = epoch;

  const std::string& request_id = request.request().request_id();
  const auto attempt = request.fence().attempt_id();
  const auto current_attempt = request_attempts_.find(request_id);
  if (current_attempt != request_attempts_.end() &&
      attempt <= current_attempt->second) {
    return reject("stale_or_duplicate_attempt");
  }
  if (live_jobs_.contains(request_id)) {
    // ContinuousEngine holds one live sequence per request id. An overlapping
    // retry is rejected without advancing the fence; the Router reroutes it
    // or retries after the current attempt becomes terminal.
    return reject("request_already_in_flight");
  }
  RememberAttempt(request_id, attempt);
  live_jobs_.emplace(request_id, job);
  incoming_.push_back(job);
  if (!first_accept_.has_value()) {
    first_accept_ = std::chrono::steady_clock::now();
  }
  ++outstanding_;
  ++stats_.accepted;
  work_ready_.notify_one();
  return {};
}

void EngineService::RememberAttempt(const std::string& request_id,
                                    const std::uint64_t attempt) {
  const auto [iterator, inserted] = request_attempts_.insert_or_assign(request_id, attempt);
  static_cast<void>(iterator);
  if (!inserted) {
    return;
  }
  attempt_order_.push_back(request_id);
  // Forgetting a live request's fence is safe: live_jobs_ still rejects an
  // overlapping dispatch, and Postgres rejects any stale completion.
  while (attempt_order_.size() > options_.fence_memory) {
    request_attempts_.erase(attempt_order_.front());
    attempt_order_.pop_front();
  }
}

void EngineService::RequestCancel(const std::string& request_id,
                                  std::string reason) {
  {
    std::lock_guard lock(mutex_);
    const auto live = live_jobs_.find(request_id);
    if (live != live_jobs_.end()) {
      live->second->abandoned.store(true, std::memory_order_release);
    }
    if (!options_.cancellation) {
      return;  // Measurement mode: keep computing the abandoned sequence.
    }
    cancels_.emplace_back(request_id, std::move(reason));
  }
  work_ready_.notify_one();
}

void EngineService::WorkerLoop() {
  // Worker-owned map from engine request id to its RPC-side job.
  std::unordered_map<std::string, std::shared_ptr<Job>> running;
  std::deque<std::shared_ptr<Job>> unstarted;
  while (true) {
    std::deque<std::shared_ptr<Job>> incoming;
    std::deque<std::pair<std::string, std::string>> cancels;
    {
      std::unique_lock lock(mutex_);
      work_ready_.wait(lock, [&] {
        return stop_ || !incoming_.empty() || !cancels_.empty() || !engine_.idle();
      });
      if (stop_) {
        unstarted.swap(incoming_);
        break;
      }
      incoming.swap(incoming_);
      cancels.swap(cancels_);
    }

    // New requests join the live batch at this iteration boundary.
    for (auto& job : incoming) {
      engine::EngineRequest request = ToEngineRequest(job->dispatch.request());
      request.arrival_time = engine_.clock().Now();
      running.insert_or_assign(request.request_id, job);
      engine_.Submit(std::move(request));
    }
    incoming.clear();
    for (const auto& [request_id, reason] : cancels) {
      static_cast<void>(engine_.Cancel(request_id, reason));
    }

    const auto simulated_before = engine_.clock().Now();
    const bool progressed = engine_.RunOneIteration();
    const auto simulated = engine_.clock().Now() - simulated_before;

    // Publish capacity before results so anyone woken by a terminal event
    // already sees the freed sequence and KV blocks.
    free_kv_blocks_.store(engine_.kv_cache().free_blocks(), std::memory_order_relaxed);
    running_seqs_.store(engine_.active_count(), std::memory_order_relaxed);
    waiting_seqs_.store(engine_.pending_count(), std::memory_order_relaxed);

    for (const auto& request_id : engine_.TakeFirstTokens()) {
      const auto job = running.find(request_id);
      if (job == running.end()) {
        continue;
      }
      v1::DispatchEvent event;
      *event.mutable_progress() = BaseEvent(*job->second);
      event.mutable_progress()->set_kind(v1::ProgressEvent::PREFILL_DONE);
      event.mutable_progress()->set_tokens_generated(1);
      Publish(*job->second, std::move(event), false);
    }
    for (const auto& result : engine_.TakeResults()) {
      auto node = running.extract(result.request_id);
      if (!node.empty()) {
        FinishJob(node.mapped(), result);
      }
    }

    // Optional wall-clock pacing so simulated cost (and test-only fixed
    // delays) turn into real time spent per iteration. Interruptible by stop.
    std::chrono::microseconds pace = options_.step_delay;
    if (options_.time_scale > 0.0 && !measured_) {
      pace += std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::duration<double, std::nano>(
              static_cast<double>(simulated.count()) * options_.time_scale));
    }
    std::unique_lock lock(mutex_);
    stats_.peak_running = std::max(stats_.peak_running, engine_.active_count());
    engine_stats_ = engine_.stats();
    if (progressed && pace > std::chrono::microseconds::zero()) {
      work_ready_.wait_for(lock, pace, [&] { return stop_; });
    } else if (!progressed && !engine_.idle()) {
      // Nothing admissible right now; wait briefly for new work or cancels.
      work_ready_.wait_for(lock, kStallWait, [&] {
        return stop_ || !incoming_.empty() || !cancels_.empty();
      });
    }
  }

  // Shutdown: fail every live attempt as retryable so a Router can redispatch.
  for (auto& job : unstarted) {
    running.insert_or_assign(job->dispatch.request().request_id(), job);
  }
  for (auto& [request_id, job] : running) {
    FinishJob(job, engine::EngineResult{.request_id = request_id,
                                        .succeeded = false,
                                        .failure_reason = "engine_stopping"});
  }
}

void EngineService::Publish(Job& job, v1::DispatchEvent event, const bool terminal) {
  {
    std::lock_guard lock(job.mutex);
    job.events.push_back(std::move(event));
    job.done = job.done || terminal;
  }
  job.changed.notify_all();
}

void EngineService::FinishJob(const std::shared_ptr<Job>& job,
                              const engine::EngineResult& result) {
  v1::DispatchEvent terminal;
  auto* event = terminal.mutable_progress();
  *event = BaseEvent(*job);
  const bool superseded = job->superseded.load(std::memory_order_acquire);
  const bool abandoned = job->abandoned.load(std::memory_order_acquire);
  const bool succeeded = result.succeeded && !superseded;
  if (succeeded) {
    event->set_kind(v1::ProgressEvent::COMPLETED);
    event->set_tokens_generated(job->dispatch.request().sampling().max_output_tokens());
    event->set_result_ref(result.output);
    event->set_prefix_cache_hit(result.prefix_cache_hit);
    event->set_result_cache_hit(result.result_cache_hit);
  } else {
    const std::string reason = superseded ? "superseded" : result.failure_reason;
    event->set_kind(v1::ProgressEvent::FAILED);
    event->set_failure_reason(reason);
    event->set_retryable(RetryableFailure(reason));
  }
  Publish(*job, std::move(terminal), true);

  std::lock_guard lock(mutex_);
  const auto live = live_jobs_.find(job->dispatch.request().request_id());
  if (live != live_jobs_.end() && live->second == job) {
    live_jobs_.erase(live);
  }
  --outstanding_;
  last_finish_ = std::chrono::steady_clock::now();
  if (result.succeeded && (superseded || abandoned)) {
    stats_.abandoned_output_tokens +=
        job->dispatch.request().sampling().max_output_tokens();
  }
  if (succeeded) {
    ++stats_.completed;
  } else if (result.failure_reason == "superseded" ||
             result.failure_reason == "client_cancelled" || superseded) {
    ++stats_.cancelled;
  } else {
    ++stats_.failed;
  }
  if (outstanding_ == 0) {
    drained_.notify_all();
  }
}

v1::ProgressEvent EngineService::BaseEvent(const Job& job) {
  v1::ProgressEvent event;
  event.set_request_id(job.dispatch.request().request_id());
  event.set_trace_id(job.dispatch.request().trace_id());
  event.set_span_id(job.dispatch.request().span_id());
  *event.mutable_fence() = job.dispatch.fence();
  return event;
}

engine::EngineRequest EngineService::ToEngineRequest(
    const v1::InferenceRequest& request) const {
  engine::EngineRequest converted{
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

  converted.prompt_block_hashes = engine::PromptBlockHashes(
      request.prompt_hash(), request.prefix_id(), request.prefix_tokens(),
      converted.prompt_tokens, config_.kv_block_size_tokens);
  return converted;
}

}  // namespace infersched::distributed
