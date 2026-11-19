// Durable Router: Kafka partition owner that dispatches requests to a fleet of
// Engines and finalizes them through PostgreSQL fencing + transactional outbox.
//
// Threads:
//   main      Kafka ownership events, ingest (Postgres), completion handling,
//             outbox relay, offset commits. Owns all per-partition state.
//   workers   N dispatch workers, each with its own Postgres connection. A
//             worker selects an Engine (load + Redis prefix affinity), assigns
//             a fenced attempt, streams the Dispatch RPC, and finalizes or
//             retries on another Engine.
//   gRPC      RouterControl (Engine registration and heartbeats).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "infersched/coordination/redis_hints.hpp"
#include "infersched/distributed/engine_selector.hpp"
#include "infersched/distributed/router_service.hpp"
#include "infersched/durable/durable_store.hpp"
#include "infersched/durable/kafka_poller.hpp"
#include "infersched/durable/outbox_relay.hpp"
#include "infersched/durable/request_ingestor.hpp"
#include "infersched/durable/trace_emitter.hpp"
#include "infersched/util/json.hpp"
#include "infersched.grpc.pb.h"

namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
namespace v1 = infersched::v1;
namespace coordination = infersched::coordination;
namespace distributed = infersched::distributed;
namespace durable = infersched::durable;
using infersched::util::JsonObject;

std::atomic<bool> g_signal_stop{false};
void HandleSignal(int) { g_signal_stop.store(true, std::memory_order_release); }

std::string Argument(const int argc, char** argv, const std::string& name,
                     const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      return argv[index + 1];
    }
  }
  return fallback;
}

struct Options {
  std::string listen;
  std::string brokers;
  std::string router_id;
  std::string consumer_group;
  std::string offset_reset;
  std::string ready_file;
  std::string database;
  std::string redis;
  bool traces{true};
  std::size_t expected{};                 // 0: run until signalled.
  std::chrono::seconds run_seconds{};     // 0: no time limit.
  std::size_t workers{128};               // Concurrent dispatch attempts.
  std::size_t db_connections{16};         // Postgres connections shared by workers.
  std::size_t max_attempts{4};            // Failed attempts before FAILED_TERMINAL.
  std::size_t max_rejections{32};         // Engine REJECTED acks before giving up.
  std::chrono::milliseconds dispatch_timeout{30'000};
  std::chrono::milliseconds engine_wait{1'000};
  std::chrono::milliseconds startup_engine_wait{10'000};
  std::chrono::milliseconds drain_timeout{5'000};
  std::size_t affinity_slack{8};
};

Options ParseOptions(const int argc, char** argv) {
  Options options;
  options.listen = Argument(argc, argv, "--listen", "127.0.0.1:50051");
  options.brokers = Argument(argc, argv, "--brokers", "localhost:9092");
  options.router_id = Argument(argc, argv, "--router-id", "router-p5");
  options.consumer_group = Argument(argc, argv, "--consumer-group", "infersched-router-p5");
  options.offset_reset = Argument(argc, argv, "--offset-reset", "earliest");
  options.ready_file = Argument(argc, argv, "--ready-file", "");
  options.database = Argument(
      argc, argv, "--database",
      "host=localhost port=55432 dbname=infersched user=infersched password=infersched");
  options.redis = Argument(argc, argv, "--redis", "127.0.0.1:6379");
  options.traces = Argument(argc, argv, "--traces", "true") == "true";
  options.expected = std::stoull(Argument(argc, argv, "--requests", "0"));
  options.run_seconds = std::chrono::seconds(std::stoll(Argument(argc, argv, "--run-seconds", "0")));
  options.workers = std::max<std::size_t>(1, std::stoull(Argument(argc, argv, "--workers", "128")));
  options.db_connections =
      std::max<std::size_t>(1, std::stoull(Argument(argc, argv, "--db-connections", "16")));
  options.max_attempts = std::max<std::size_t>(1, std::stoull(Argument(argc, argv, "--max-attempts", "4")));
  options.max_rejections = std::stoull(Argument(argc, argv, "--max-rejections", "32"));
  options.dispatch_timeout =
      std::chrono::milliseconds(std::stoll(Argument(argc, argv, "--dispatch-timeout-ms", "30000")));
  options.drain_timeout =
      std::chrono::milliseconds(std::stoll(Argument(argc, argv, "--drain-timeout-ms", "5000")));
  options.affinity_slack = std::stoull(Argument(argc, argv, "--affinity-slack", "8"));
  return options;
}

template <typename T>
class BlockingQueue {
 public:
  void Push(T value) {
    {
      std::lock_guard lock(mutex_);
      items_.push_back(std::move(value));
    }
    ready_.notify_one();
  }
  std::optional<T> Pop(const std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    ready_.wait_for(lock, timeout, [&] { return closed_ || !items_.empty(); });
    if (items_.empty()) {
      return std::nullopt;
    }
    T value = std::move(items_.front());
    items_.pop_front();
    return value;
  }
  void Close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    ready_.notify_all();
  }
  [[nodiscard]] bool closed() const {
    std::lock_guard lock(mutex_);
    return closed_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<T> items_;
  bool closed_{};
};

struct WorkItem {
  v1::InferenceRequest request;
  std::int32_t partition{};
  std::int64_t offset{};
  std::uint64_t epoch{};
  bool from_kafka{};
  std::chrono::system_clock::time_point received;
};

enum class Outcome {
  kCompleted,
  kFailedTerminal,
  kAlreadyTerminal,
  kNotOwner,
  kRetryLater,  // Infrastructure error (e.g. lost DB connection); requeue.
  kStopped,
};

struct Completion {
  WorkItem item;
  Outcome outcome{};
};

// In-flight attempts: exact per-Engine counts for load estimation, and client
// contexts so a partition revocation can tear down its streams immediately.
class InflightRegistry {
 public:
  // Selection and reservation are one atomic step, so concurrent workers see
  // each other's choices instead of all piling onto the same idle Engine.
  std::optional<distributed::Selection> SelectAndReserve(
      const std::vector<distributed::EngineRecord>& engines,
      const std::unordered_set<std::string>& excluded,
      std::optional<coordination::PrefixHint> hint, const std::size_t affinity_slack,
      const std::string_view tie_breaker) {
    std::lock_guard lock(mutex_);
    auto selection = distributed::SelectEngine({.engines = engines,
                                                .local_inflight = counts_,
                                                .excluded = excluded,
                                                .hint = std::move(hint),
                                                .affinity_slack = affinity_slack,
                                                .tie_breaker = tie_breaker});
    if (selection.has_value()) {
      ++counts_[selection->engine.engine_id];
    }
    return selection;
  }
  void Release(const std::string& engine_id) {
    std::lock_guard lock(mutex_);
    auto count = counts_.find(engine_id);
    if (count != counts_.end() && --count->second == 0) {
      counts_.erase(count);
    }
  }
  void Begin(const std::string& request_id, const std::string& engine_id,
             const std::int32_t partition, grpc::ClientContext* context) {
    std::lock_guard lock(mutex_);
    entries_.insert_or_assign(request_id, Entry{.engine_id = engine_id,
                                                .partition = partition,
                                                .context = context});
  }
  // Returns true when the attempt was revoked (partition lost or shutdown).
  bool End(const std::string& request_id) {
    std::lock_guard lock(mutex_);
    const auto entry = entries_.find(request_id);
    if (entry == entries_.end()) {
      return false;
    }
    const bool revoked = entry->second.revoked;
    entries_.erase(entry);
    return revoked;
  }
  void RevokePartitions(const std::vector<std::int32_t>& partitions) {
    std::lock_guard lock(mutex_);
    for (auto& [request_id, entry] : entries_) {
      static_cast<void>(request_id);
      if (std::ranges::find(partitions, entry.partition) != partitions.end()) {
        entry.revoked = true;
        entry.context->TryCancel();
      }
    }
  }
  void RevokeAll() {
    std::lock_guard lock(mutex_);
    for (auto& [request_id, entry] : entries_) {
      static_cast<void>(request_id);
      entry.revoked = true;
      entry.context->TryCancel();
    }
  }
  [[nodiscard]] std::unordered_map<std::string, std::size_t> Counts() const {
    std::lock_guard lock(mutex_);
    return counts_;
  }

 private:
  struct Entry {
    std::string engine_id;
    std::int32_t partition{};
    grpc::ClientContext* context{};
    bool revoked{};
  };
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_map<std::string, std::size_t> counts_;
};

class StubCache {
 public:
  std::shared_ptr<v1::EngineControl::Stub> Get(const std::string& endpoint) {
    std::lock_guard lock(mutex_);
    auto& stub = stubs_[endpoint];
    if (!stub) {
      stub = v1::EngineControl::NewStub(
          grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
    }
    return stub;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<v1::EngineControl::Stub>> stubs_;
};

// Dispatch workers far outnumber Postgres connections: a worker leases a
// connection only for its short fenced writes, never while it streams a
// Dispatch RPC, so in-flight concurrency is not capped by max_connections.
class StorePool {
 public:
  StorePool(std::string database, const std::size_t size)
      : database_(std::move(database)), slots_(size) {
    for (std::size_t slot = 0; slot < size; ++slot) {
      free_.push_back(slot);
    }
  }

  template <typename Fn>
  auto Run(Fn&& fn) -> decltype(fn(std::declval<durable::DurableStore&>())) {
    std::size_t slot = 0;
    {
      std::unique_lock lock(mutex_);
      available_.wait(lock, [&] { return !free_.empty(); });
      slot = free_.back();
      free_.pop_back();
    }
    struct Return {
      StorePool& pool;
      std::size_t slot;
      ~Return() { pool.Release(slot); }
    } lease{*this, slot};
    try {
      if (!slots_[slot]) {
        slots_[slot] = std::make_unique<durable::DurableStore>(database_);
      }
      return fn(*slots_[slot]);
    } catch (...) {
      slots_[slot].reset();  // Broken connection: reconnect on next lease.
      throw;
    }
  }

 private:
  void Release(const std::size_t slot) {
    {
      std::lock_guard lock(mutex_);
      free_.push_back(slot);
    }
    available_.notify_one();
  }

  std::string database_;
  std::mutex mutex_;
  std::condition_variable available_;
  // Each slot is touched only by its current leaseholder.
  std::vector<std::unique_ptr<durable::DurableStore>> slots_;
  std::vector<std::size_t> free_;
};

struct RouterStats {
  std::atomic<std::uint64_t> dispatched{};
  std::atomic<std::uint64_t> completed{};
  std::atomic<std::uint64_t> failed_terminal{};
  std::atomic<std::uint64_t> retries{};
  std::atomic<std::uint64_t> rejections{};
  std::atomic<std::uint64_t> affinity_dispatches{};
  std::atomic<std::uint64_t> revoked_attempts{};
  std::atomic<std::uint64_t> prefix_cache_hits{};
  std::mutex mutex;
  std::map<std::string, std::uint64_t> by_engine;

  void Dispatched(const std::string& engine_id, const bool affinity) {
    dispatched.fetch_add(1, std::memory_order_relaxed);
    if (affinity) {
      affinity_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard lock(mutex);
    ++by_engine[engine_id];
  }
};

struct DispatchResult {
  enum class Kind { kCompleted, kRejected, kFailed, kTransport };
  Kind kind{Kind::kTransport};
  std::string output;
  std::string reason;
  bool retryable{};
  bool revoked{};
  bool prefix_cache_hit{};
  bool result_cache_hit{};
  std::optional<std::chrono::microseconds> ttft;
};

const char* KindName(const DispatchResult::Kind kind) {
  switch (kind) {
    case DispatchResult::Kind::kCompleted: return "completed";
    case DispatchResult::Kind::kRejected: return "rejected";
    case DispatchResult::Kind::kFailed: return "failed";
    case DispatchResult::Kind::kTransport: return "transport_error";
  }
  return "unknown";
}

struct Runtime {
  Options options;
  distributed::RouterService& service;
  durable::KafkaPoller& poller;
  coordination::RedisHints* redis{};
  durable::TraceEmitter* traces{};
  InflightRegistry inflight;
  StubCache stubs;
  RouterStats stats;
  std::atomic<bool> stopping{false};
  std::mutex sleep_mutex;
  std::condition_variable sleep_cv;

  void SleepFor(const std::chrono::milliseconds duration) {
    std::unique_lock lock(sleep_mutex);
    sleep_cv.wait_for(lock, duration, [&] { return stopping.load(); });
  }
  void Stop() {
    stopping.store(true);
    sleep_cv.notify_all();
  }
};

std::string ResultEnvelope(const Runtime& runtime, const WorkItem& item,
                           const durable::Fence& fence, const DispatchResult* result,
                           const std::string_view failure) {
  const v1::InferenceRequest& request = item.request;
  JsonObject envelope;
  envelope.Add("request_id", request.request_id())
      .Add("trace_id", request.trace_id())
      .Add("span_id", request.span_id())
      .Add("status", result != nullptr ? "COMPLETED" : "FAILED")
      .Add("router_id", runtime.options.router_id)
      .Add("engine_id", fence.engine_id)
      .Add("partition_id", fence.partition_id)
      .Add("ownership_epoch", fence.ownership_epoch)
      .Add("attempt_id", fence.attempt_id);
  if (result != nullptr) {
    envelope.Add("output", result->output)
        .Add("prefix_cache_hit", result->prefix_cache_hit)
        .Add("result_cache_hit", result->result_cache_hit);
  } else {
    envelope.Add("error", failure);
  }
  return envelope.str();
}

void SendCancel(v1::EngineControl::Stub& stub, const std::string& request_id,
                const durable::Fence& fence) {
  v1::CancelRequest cancel;
  cancel.set_request_id(request_id);
  cancel.mutable_fence()->set_partition_id(fence.partition_id);
  cancel.mutable_fence()->set_ownership_epoch(fence.ownership_epoch);
  cancel.mutable_fence()->set_attempt_id(fence.attempt_id);
  v1::CancelReply reply;
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 200ms);
  static_cast<void>(stub.Cancel(&context, cancel, &reply));
}

DispatchResult Dispatch(Runtime& runtime, const distributed::EngineRecord& engine,
                        const v1::InferenceRequest& request, const durable::Fence& fence) {
  v1::DispatchRequest dispatch;
  *dispatch.mutable_request() = request;
  dispatch.mutable_fence()->set_partition_id(fence.partition_id);
  dispatch.mutable_fence()->set_ownership_epoch(fence.ownership_epoch);
  dispatch.mutable_fence()->set_attempt_id(fence.attempt_id);

  const auto stub = runtime.stubs.Get(engine.endpoint);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + runtime.options.dispatch_timeout);
  runtime.inflight.Begin(request.request_id(), engine.engine_id, fence.partition_id, &context);
  const auto started = Clock::now();

  DispatchResult result;
  bool terminal = false;
  auto reader = stub->Dispatch(&context, dispatch);
  v1::DispatchEvent event;
  while (reader->Read(&event)) {
    if (event.has_ack()) {
      if (event.ack().status() == v1::DispatchAck::REJECTED) {
        result.kind = DispatchResult::Kind::kRejected;
        result.reason = event.ack().reason();
        terminal = true;
      }
      continue;
    }
    if (!event.has_progress()) {
      continue;
    }
    const v1::ProgressEvent& progress = event.progress();
    if (progress.fence().ownership_epoch() != fence.ownership_epoch ||
        progress.fence().attempt_id() != fence.attempt_id) {
      continue;  // Not this attempt's event.
    }
    switch (progress.kind()) {
      case v1::ProgressEvent::PREFILL_DONE:
        result.ttft = std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - started);
        break;
      case v1::ProgressEvent::COMPLETED:
        result.kind = DispatchResult::Kind::kCompleted;
        result.output = progress.result_ref();
        result.prefix_cache_hit = progress.prefix_cache_hit();
        result.result_cache_hit = progress.result_cache_hit();
        terminal = true;
        break;
      case v1::ProgressEvent::FAILED:
        result.kind = DispatchResult::Kind::kFailed;
        result.reason = progress.failure_reason();
        result.retryable = progress.retryable();
        terminal = true;
        break;
      default:
        break;
    }
  }
  const grpc::Status status = reader->Finish();
  result.revoked = runtime.inflight.End(request.request_id());
  if (!terminal) {
    result.kind = DispatchResult::Kind::kTransport;
    result.retryable = true;
    result.reason = "transport:" + std::to_string(static_cast<int>(status.error_code())) +
                    ':' + status.error_message();
    // The stream is gone but the Engine may still hold the sequence (deadline,
    // revocation, network partition): cancel the exact fenced attempt.
    if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED || result.revoked) {
      SendCancel(*stub, request.request_id(), fence);
    }
  }
  return result;
}

void EmitSpan(Runtime& runtime, const WorkItem& item, std::string name,
              const std::chrono::system_clock::time_point start,
              std::vector<std::pair<std::string, std::string>> attributes) {
  if (runtime.traces == nullptr) {
    return;
  }
  attributes.emplace_back("router_id", runtime.options.router_id);
  attributes.emplace_back("request_id", item.request.request_id());
  runtime.traces->Emit(durable::Span{.trace_id = item.request.trace_id(),
                                     .span_id = durable::TraceEmitter::NewSpanId(),
                                     .parent_span_id = item.request.span_id(),
                                     .name = std::move(name),
                                     .start = start,
                                     .end = std::chrono::system_clock::now(),
                                     .attributes = std::move(attributes)});
}

Outcome Execute(Runtime& runtime, StorePool& db, const WorkItem& item) {
  const v1::InferenceRequest& request = item.request;
  const std::string& request_id = request.request_id();
  std::unordered_set<std::string> excluded;
  std::size_t failures = 0;
  std::size_t rejections = 0;
  std::size_t attempts = 0;
  std::string last_failure = "no_attempt";
  auto backoff = 20ms;
  const auto retry_pause = [&] {
    runtime.SleepFor(backoff);
    backoff = std::min<std::chrono::milliseconds>(backoff * 2, 1000ms);
  };
  const auto resolve_conflict = [&] {
    return db.Run([&](durable::DurableStore& store) { return store.IsTerminal(request_id); })
               ? Outcome::kAlreadyTerminal
               : Outcome::kNotOwner;
  };
  const auto finish = [&](const Outcome outcome) {
    if (outcome == Outcome::kCompleted || outcome == Outcome::kFailedTerminal) {
      EmitSpan(runtime, item, "router.request", item.received,
               {{"outcome", outcome == Outcome::kCompleted ? "completed" : "failed_terminal"},
                {"attempts", std::to_string(attempts)},
                {"partition_id", std::to_string(item.partition)}});
    }
    return outcome;
  };

  while (true) {
    if (runtime.stopping.load()) {
      return Outcome::kStopped;
    }
    if (!runtime.poller.Owns(item.partition)) {
      return Outcome::kNotOwner;
    }
    const auto engines = runtime.service.WaitForLiveEngines(runtime.options.engine_wait);
    if (engines.empty()) {
      continue;  // Engine outage does not consume the retry budget.
    }
    std::optional<coordination::PrefixHint> hint;
    if (runtime.redis != nullptr && !request.prefix_id().empty()) {
      hint = runtime.redis->GetPrefixOwner(request.model_revision(), request.prefix_id());
    }
    auto selection = runtime.inflight.SelectAndReserve(
        engines, excluded, hint, runtime.options.affinity_slack, request_id);
    if (!selection.has_value()) {
      // Every live Engine already failed this request: retry them all.
      excluded.clear();
      continue;
    }
    const distributed::EngineRecord& engine = selection->engine;
    // The reservation counts toward this Engine's load until the attempt ends.
    struct ReleaseOnExit {
      InflightRegistry& registry;
      const std::string& engine_id;
      ~ReleaseOnExit() { registry.Release(engine_id); }
    } reservation{runtime.inflight, engine.engine_id};
    const auto assignment = db.Run([&](durable::DurableStore& store) {
      return store.AssignAttempt(request_id, item.partition, item.epoch, engine.engine_id);
    });
    if (!assignment.has_value()) {
      return resolve_conflict();  // Lost ownership, or already terminal.
    }
    ++attempts;
    runtime.stats.Dispatched(engine.engine_id, selection->affinity);
    const auto dispatch_start = std::chrono::system_clock::now();
    const DispatchResult result = Dispatch(runtime, engine, request, assignment->fence);
    std::vector<std::pair<std::string, std::string>> attributes{
        {"engine_id", engine.engine_id},
        {"attempt_id", std::to_string(assignment->fence.attempt_id)},
        {"ownership_epoch", std::to_string(assignment->fence.ownership_epoch)},
        {"result", KindName(result.kind)},
        {"affinity", selection->affinity ? "true" : "false"}};
    if (!result.reason.empty()) {
      attributes.emplace_back("reason", result.reason);
    }
    if (result.ttft.has_value()) {
      attributes.emplace_back("ttft_us", std::to_string(result.ttft->count()));
    }
    EmitSpan(runtime, item, "router.dispatch", dispatch_start, std::move(attributes));

    switch (result.kind) {
      case DispatchResult::Kind::kCompleted: {
        if (result.prefix_cache_hit) {
          runtime.stats.prefix_cache_hits.fetch_add(1, std::memory_order_relaxed);
        }
        const std::string envelope =
            ResultEnvelope(runtime, item, assignment->fence, &result, {});
        if (!db.Run([&](durable::DurableStore& store) {
              return store.Finalize(request_id, assignment->fence, envelope);
            })) {
          return resolve_conflict();  // A newer fence owns this request.
        }
        if (runtime.redis != nullptr && !request.prefix_id().empty()) {
          runtime.redis->PutPrefixOwner(
              request.model_revision(), request.prefix_id(),
              {.engine_id = engine.engine_id, .incarnation_id = engine.incarnation_id});
        }
        runtime.stats.completed.fetch_add(1, std::memory_order_relaxed);
        return finish(Outcome::kCompleted);
      }
      case DispatchResult::Kind::kRejected:
        runtime.stats.rejections.fetch_add(1, std::memory_order_relaxed);
        excluded.insert(engine.engine_id);
        last_failure = "rejected:" + result.reason;
        if (++rejections < runtime.options.max_rejections) {
          retry_pause();
          continue;
        }
        break;
      case DispatchResult::Kind::kFailed:
      case DispatchResult::Kind::kTransport:
        if (result.revoked) {
          runtime.stats.revoked_attempts.fetch_add(1, std::memory_order_relaxed);
          return runtime.stopping.load() ? Outcome::kStopped : Outcome::kNotOwner;
        }
        last_failure = result.reason;
        if (result.retryable && ++failures < runtime.options.max_attempts) {
          runtime.stats.retries.fetch_add(1, std::memory_order_relaxed);
          excluded.insert(engine.engine_id);
          retry_pause();
          continue;
        }
        break;
    }

    // Non-retryable failure or retry budget exhausted: fenced terminal failure.
    const std::string envelope =
        ResultEnvelope(runtime, item, assignment->fence, nullptr, last_failure);
    if (!db.Run([&](durable::DurableStore& store) {
          return store.FailTerminal(request_id, item.partition, item.epoch, envelope);
        })) {
      return resolve_conflict();
    }
    runtime.stats.failed_terminal.fetch_add(1, std::memory_order_relaxed);
    return finish(Outcome::kFailedTerminal);
  }
}

std::string StatsJson(Runtime& runtime, const std::size_t resolved) {
  JsonObject engines;
  {
    std::lock_guard lock(runtime.stats.mutex);
    for (const auto& [engine_id, count] : runtime.stats.by_engine) {
      engines.Add(engine_id, count);
    }
  }
  JsonObject output;
  output.Add("router_id", runtime.options.router_id)
      .Add("completed", resolved)
      .Add("succeeded", runtime.stats.completed.load())
      .Add("failed_terminal", runtime.stats.failed_terminal.load())
      .Add("dispatched", runtime.stats.dispatched.load())
      .Add("retries", runtime.stats.retries.load())
      .Add("rejections", runtime.stats.rejections.load())
      .Add("revoked_attempts", runtime.stats.revoked_attempts.load())
      .Add("affinity_dispatches", runtime.stats.affinity_dispatches.load())
      .Add("prefix_cache_hits", runtime.stats.prefix_cache_hits.load())
      .AddRaw("dispatch_by_engine", engines.str());
  if (runtime.redis != nullptr) {
    const auto redis = runtime.redis->stats();
    output.AddRaw("redis", JsonObject{}
                               .Add("lookups", redis.lookups)
                               .Add("hits", redis.hits)
                               .Add("writes", redis.writes)
                               .Add("errors", redis.errors)
                               .str());
  }
  if (runtime.traces != nullptr) {
    output.Add("spans_emitted", runtime.traces->emitted());
  }
  return output.str();
}

}  // namespace

int main(const int argc, char** argv) {
  const Options options = ParseOptions(argc, argv);
  std::signal(SIGTERM, HandleSignal);
  std::signal(SIGINT, HandleSignal);

  durable::DurableStore store(options.database);
  store.Migrate();
  durable::RequestIngestor ingestor(store);
  durable::KafkaOutboxPublisher publisher(options.brokers);
  durable::OutboxRelay relay(store, publisher);
  durable::KafkaPoller poller(options.brokers, options.consumer_group,
                              "inference.requests", options.offset_reset);

  distributed::RouterService service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(options.listen, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server) {
    std::cerr << "failed to start RouterControl at " << options.listen << '\n';
    return EXIT_FAILURE;
  }
  if (service.WaitForLiveEngines(options.startup_engine_wait).empty()) {
    std::cerr << "no Engine registered within startup wait\n";
    server->Shutdown();
    return EXIT_FAILURE;
  }

  std::unique_ptr<coordination::RedisHints> redis;
  if (const auto address = coordination::RedisHints::ParseAddress(options.redis)) {
    redis = std::make_unique<coordination::RedisHints>(*address);
  }
  std::unique_ptr<durable::TraceEmitter> traces;
  if (options.traces) {
    traces = std::make_unique<durable::TraceEmitter>(options.brokers, "router");
  }
  Runtime runtime{.options = options, .service = service, .poller = poller,
                  .redis = redis.get(), .traces = traces.get()};

  BlockingQueue<WorkItem> work;
  BlockingQueue<Completion> done;
  StorePool db_pool(options.database, options.db_connections);
  std::vector<std::thread> workers;
  workers.reserve(options.workers);
  for (std::size_t index = 0; index < options.workers; ++index) {
    workers.emplace_back([&] {
      while (true) {
        auto item = work.Pop(250ms);
        if (!item.has_value()) {
          if (work.closed()) {
            return;
          }
          continue;
        }
        Outcome outcome = Outcome::kStopped;
        try {
          outcome = Execute(runtime, db_pool, *item);
        } catch (const std::exception& error) {
          // Lost DB connection or similar: reconnect on the next item and
          // requeue this one; Postgres fencing makes the retry safe.
          std::cerr << "worker error for " << item->request.request_id() << ": "
                    << error.what() << '\n';
          runtime.SleepFor(100ms);
          outcome = Outcome::kRetryLater;
        }
        done.Push(Completion{.item = std::move(*item), .outcome = outcome});
      }
    });
  }

  // Main-thread state (single owner, no locking).
  std::unordered_map<std::int32_t, std::uint64_t> epochs;
  std::unordered_set<std::string> active;
  std::unordered_map<std::string, std::vector<durable::KafkaMessage>> deferred;
  std::size_t resolved = 0;
  const std::size_t window = options.workers * 2;

  const auto enqueue = [&](WorkItem item) {
    active.insert(item.request.request_id());
    work.Push(std::move(item));
  };
  // Commits Kafka offsets whose outbox rows (result or DLQ) are all broker-
  // acknowledged. One query for the whole batch.
  const auto ack_published = [&](const std::vector<std::pair<std::int32_t, std::int64_t>>& sources) {
    if (sources.empty()) {
      return;
    }
    const std::vector<bool> published = store.SourcesPublished(sources);
    for (std::size_t index = 0; index < sources.size(); ++index) {
      if (published[index]) {
        poller.MarkPublishComplete(sources[index].first, sources[index].second);
      }
    }
  };
  const auto handle_messages = [&](std::vector<durable::KafkaMessage> messages) {
    struct Parsed {
      durable::KafkaMessage message;
      v1::InferenceRequest request;
    };
    std::vector<Parsed> parsed;
    std::vector<std::pair<std::int32_t, std::int64_t>> to_ack;
    bool publish = false;
    parsed.reserve(messages.size());
    for (auto& message : messages) {
      if (!poller.Owns(message.partition)) {
        continue;
      }
      v1::InferenceRequest request;
      if (request.ParseFromString(message.payload) && !request.request_id().empty()) {
        parsed.push_back(Parsed{.message = std::move(message), .request = std::move(request)});
        continue;
      }
      // Poison: parsing is deterministic, so the bounded parse retries run
      // back to back instead of stalling the partition's commit on redelivery.
      durable::IngestOutcome outcome;
      do {
        outcome = ingestor.Handle(message.partition, message.offset, message.payload);
      } while (outcome == durable::IngestOutcome::kRetryParse);
      if (outcome == durable::IngestOutcome::kPoisonRecorded) {
        publish = true;
        to_ack.emplace_back(message.partition, message.offset);
      }
    }
    std::vector<durable::IngestRow> rows;
    rows.reserve(parsed.size());
    for (const auto& entry : parsed) {
      rows.push_back(durable::IngestRow{.request_id = entry.request.request_id(),
                                        .source_partition = entry.message.partition,
                                        .source_offset = entry.message.offset,
                                        .raw_payload = entry.message.payload});
    }
    const std::vector<durable::IngestStatus> statuses = store.IngestBatch(rows);
    for (std::size_t index = 0; index < parsed.size(); ++index) {
      auto& [message, request] = parsed[index];
      if (active.contains(request.request_id())) {
        deferred[request.request_id()].push_back(std::move(message));  // Redelivery in flight.
        continue;
      }
      if (statuses[index].terminal) {
        publish = true;  // Outcome exists; make sure it is published, then ack.
        to_ack.emplace_back(message.partition, message.offset);
        continue;
      }
      const auto epoch = epochs.find(message.partition);
      if (epoch == epochs.end()) {
        continue;
      }
      WorkItem item{.request = std::move(request), .partition = message.partition,
                    .offset = message.offset, .epoch = epoch->second, .from_kafka = true,
                    .received = std::chrono::system_clock::now()};
      if (message.timestamp_ms >= 0) {
        // Producer CreateTime -> enqueued: Kafka transit plus ingest backlog.
        EmitSpan(runtime, item, "router.ingest",
                 std::chrono::system_clock::time_point(
                     std::chrono::milliseconds(message.timestamp_ms)),
                 {{"partition_id", std::to_string(message.partition)},
                  {"offset", std::to_string(message.offset)},
                  {"batch_size", std::to_string(messages.size())}});
      }
      enqueue(std::move(item));
    }
    if (publish) {
      static_cast<void>(relay.PublishBatch(1000));
    }
    ack_published(to_ack);
  };
  const auto handle_completions = [&](std::vector<Completion> completions) {
    std::vector<Completion> to_ack;
    std::vector<durable::KafkaMessage> redeliveries;
    for (auto& completion : completions) {
      const std::string request_id = completion.item.request.request_id();  // Copy: item may move.
      active.erase(request_id);
      switch (completion.outcome) {
        case Outcome::kCompleted:
        case Outcome::kFailedTerminal:
          ++resolved;
          [[fallthrough]];
        case Outcome::kAlreadyTerminal:
          to_ack.push_back(std::move(completion));
          break;
        case Outcome::kNotOwner:
        case Outcome::kRetryLater: {
          // Requeue while this Router still owns the partition: after an
          // infrastructure error, or when the partition was revoked and then
          // re-acquired under a new epoch while the old attempt drained.
          const auto epoch = epochs.find(completion.item.partition);
          const bool new_epoch =
              epoch != epochs.end() && epoch->second != completion.item.epoch;
          if (epoch != epochs.end() && poller.Owns(completion.item.partition) &&
              !runtime.stopping.load() &&
              (new_epoch || completion.outcome == Outcome::kRetryLater)) {
            completion.item.epoch = epoch->second;
            enqueue(std::move(completion.item));
          }
          break;
        }
        case Outcome::kStopped:
          break;
      }
      if (auto node = deferred.extract(request_id); !node.empty()) {
        for (auto& message : node.mapped()) {
          redeliveries.push_back(std::move(message));
        }
      }
    }
    if (!to_ack.empty()) {
      while (relay.PublishBatch(1000) == 1000) {
      }
      std::vector<std::pair<std::int32_t, std::int64_t>> sources;
      for (const auto& completion : to_ack) {
        if (completion.item.from_kafka) {
          sources.emplace_back(completion.item.partition, completion.item.offset);
        }
      }
      ack_published(sources);
    }
    if (!redeliveries.empty()) {
      handle_messages(std::move(redeliveries));
    }
  };
  const auto drain_completions = [&](const std::chrono::milliseconds wait) {
    std::vector<Completion> completions;
    if (auto first = done.Pop(wait)) {
      completions.push_back(std::move(*first));
      while (auto next = done.Pop(0ms)) {
        completions.push_back(std::move(*next));
      }
    }
    if (!completions.empty()) {
      handle_completions(std::move(completions));
    }
  };

  poller.Start();
  const auto deadline = Clock::now() + options.run_seconds;
  auto last_relay = Clock::now();
  const auto should_run = [&] {
    if (g_signal_stop.load(std::memory_order_acquire)) {
      return false;
    }
    if (options.expected > 0 && resolved >= options.expected) {
      return false;
    }
    return options.run_seconds.count() == 0 || Clock::now() < deadline;
  };
  while (should_run()) {
    drain_completions(0ms);
    if (const auto ownership = poller.PopOwnershipEvent(0ms)) {
      if (ownership->kind == durable::OwnershipEventKind::kRevoked) {
        for (const std::int32_t partition : ownership->partitions) {
          epochs.erase(partition);
        }
        runtime.inflight.RevokePartitions(ownership->partitions);
        continue;
      }
      std::unordered_set<std::int32_t> assigned;
      for (const std::int32_t partition : ownership->partitions) {
        epochs.insert_or_assign(partition, store.AcquirePartition(partition, options.router_id));
        assigned.insert(partition);
      }
      if (!options.ready_file.empty()) {
        std::ofstream marker(options.ready_file, std::ios::trunc);
        if (!marker) {
          std::cerr << "failed to write readiness marker: " << options.ready_file << '\n';
          break;
        }
        marker << "assigned\n";
      }
      // Recover unresolved work for the newly acquired partitions under the
      // new epoch (a previous owner may have crashed mid-attempt).
      for (const auto& recovered : store.RecoverUnresolved()) {
        if (!assigned.contains(recovered.source_partition) ||
            active.contains(recovered.request_id)) {
          continue;
        }
        v1::InferenceRequest request;
        if (!request.ParseFromString(recovered.raw_payload)) {
          continue;
        }
        enqueue(WorkItem{.request = std::move(request),
                         .partition = recovered.source_partition,
                         .offset = recovered.source_offset,
                         .epoch = epochs.at(recovered.source_partition),
                         .from_kafka = false,
                         .received = std::chrono::system_clock::now()});
      }
      continue;
    }
    if (Clock::now() - last_relay > 500ms) {
      // Publish outbox rows left behind by crashed Routers or failed batches.
      static_cast<void>(relay.PublishBatch(1000));
      last_relay = Clock::now();
    }
    if (active.size() >= window) {
      drain_completions(20ms);  // Backpressure: stop pulling from Kafka.
      continue;
    }
    // Drain whatever Kafka has buffered (bounded by the in-flight window) and
    // ingest it as one batch: one Postgres commit instead of one per message.
    if (auto first = poller.Pop(active.empty() ? 100ms : 2ms)) {
      std::vector<durable::KafkaMessage> batch;
      batch.push_back(std::move(*first));
      while (batch.size() < window - std::min(window, active.size()) + 1) {
        auto next = poller.Pop(0ms);
        if (!next.has_value()) {
          break;
        }
        batch.push_back(std::move(*next));
      }
      handle_messages(std::move(batch));
    }
  }

  // Graceful shutdown: no new work; let in-flight attempts finish within the
  // drain timeout, then revoke the rest (they stay unresolved for recovery).
  work.Close();
  const auto drain_deadline = Clock::now() + options.drain_timeout;
  while (!active.empty() && Clock::now() < drain_deadline) {
    drain_completions(50ms);
  }
  runtime.Stop();
  runtime.inflight.RevokeAll();
  for (auto& worker : workers) {
    worker.join();
  }
  drain_completions(0ms);
  static_cast<void>(relay.PublishBatch(1000));
  poller.Stop();
  server->Shutdown(std::chrono::system_clock::now() + 1s);
  std::cout << StatsJson(runtime, resolved) << std::endl;
  const bool reached = options.expected == 0 || resolved >= options.expected;
  return reached ? EXIT_SUCCESS : EXIT_FAILURE;
}
