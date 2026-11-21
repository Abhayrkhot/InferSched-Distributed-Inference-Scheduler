// Hermetic tests for the network Engine (in-process gRPC), Engine selection,
// Redis hint degradation, and JSON escaping.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "infersched.grpc.pb.h"
#include "infersched/coordination/redis_hints.hpp"
#include "infersched/distributed/engine_selector.hpp"
#include "infersched/distributed/engine_service.hpp"
#include "infersched/engine/continuous_engine.hpp"
#include "infersched/engine/request_queue.hpp"
#include "infersched/util/json.hpp"

namespace {

using namespace std::chrono_literals;
using infersched::coordination::PrefixHint;
using infersched::coordination::RedisHints;
using infersched::distributed::EngineRecord;
using infersched::distributed::EngineService;
using infersched::distributed::EngineServiceOptions;
using infersched::distributed::SelectEngine;
using infersched::engine::ContinuousEngine;
using infersched::engine::EngineConfig;
using infersched::engine::EngineRequest;
namespace v1 = infersched::v1;
using Clock = std::chrono::steady_clock;

EngineRequest Request(std::string id, std::size_t prompt, std::size_t output) {
  return EngineRequest{.request_id = std::move(id),
                       .model_revision = "m",
                       .tokenizer_revision = "t",
                       .prompt_hash = "h",
                       .prompt_tokens = prompt,
                       .max_output_tokens = output};
}

TEST(RequestQueue, RemoveKeepsRemainingOrder) {
  infersched::engine::RequestQueue queue;
  for (const char* id : {"c", "a", "b"}) {
    queue.Push(Request(id, 1, 1));
  }
  ASSERT_TRUE(queue.Remove("a").has_value());
  EXPECT_FALSE(queue.Remove("a").has_value());
  EXPECT_EQ(queue.PopReady({})->request_id, "b");
  EXPECT_EQ(queue.PopReady({})->request_id, "c");
  EXPECT_TRUE(queue.empty());
}

TEST(ContinuousEngine, CancelFreesRunningAndQueuedRequests) {
  EngineConfig config;
  config.kv_blocks = 64;
  config.max_batch_sequences = 1;
  ContinuousEngine engine(config);
  engine.Submit(Request("running", 32, 100));
  engine.Submit(Request("queued", 32, 100));
  ASSERT_TRUE(engine.RunOneIteration());
  ASSERT_EQ(engine.active_count(), 1u);
  ASSERT_GT(engine.kv_cache().allocated_blocks(), 0u);

  EXPECT_TRUE(engine.Cancel("running", "superseded"));
  EXPECT_TRUE(engine.Cancel("queued", "superseded"));
  EXPECT_FALSE(engine.Cancel("missing", "superseded"));
  EXPECT_TRUE(engine.idle());
  EXPECT_EQ(engine.kv_cache().allocated_blocks(), 0u);
  const auto results = engine.TakeResults();
  ASSERT_EQ(results.size(), 2u);
  for (const auto& result : results) {
    EXPECT_FALSE(result.succeeded);
    EXPECT_EQ(result.failure_reason, "superseded");
  }
  EXPECT_TRUE(engine.TakeResults().empty());
  EXPECT_TRUE(engine.kv_cache().CheckInvariants());
}

TEST(ContinuousEngine, StaticPolicyAdmitsOnlyAfterTheBatchDrains) {
  for (const auto policy : {infersched::engine::AdmissionPolicy::kStatic,
                            infersched::engine::AdmissionPolicy::kContinuous}) {
    EngineConfig config;
    config.admission = policy;
    config.max_batch_sequences = 4;
    ContinuousEngine engine(config);
    engine.Submit(Request("long", 16, 10));
    ASSERT_TRUE(engine.RunOneIteration());
    EngineRequest late = Request("short", 16, 2);
    late.arrival_time = engine.clock().Now();
    engine.Submit(std::move(late));
    ASSERT_TRUE(engine.RunOneIteration());
    const bool static_policy = policy == infersched::engine::AdmissionPolicy::kStatic;
    // Static leaves the free slot idle while the long sequence decodes.
    EXPECT_EQ(engine.active_count(), static_policy ? 1u : 2u);
    engine.RunUntilIdle();
    const auto results = engine.TakeResults();
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results.front().request_id, static_policy ? "long" : "short");
  }
}

TEST(ContinuousEngine, TokenAccountingBalancesUnderPreemption) {
  EngineConfig config;
  config.kv_blocks = 12;
  config.kv_block_size_tokens = 4;
  config.max_batch_sequences = 4;
  config.max_prefill_sequences_per_iteration = 4;
  ContinuousEngine engine(config);
  std::size_t requested = 0;
  for (int index = 0; index < 6; ++index) {
    EngineRequest request = Request("r" + std::to_string(index), 8, 12);
    request.prompt_hash = "unique-" + std::to_string(index);  // No result-cache hits.
    engine.Submit(std::move(request));
    requested += 12;
  }
  engine.RunUntilIdle();
  const auto& stats = engine.stats();
  ASSERT_EQ(engine.TakeResults().size(), 6u);
  EXPECT_GT(stats.preemptions, 0u);
  // Every decoded token is either delivered or thrown away by preemption.
  EXPECT_EQ(stats.decode_token_slots, stats.useful_output_tokens + stats.wasted_decode_tokens);
  EXPECT_EQ(stats.useful_output_tokens, requested);
  EXPECT_GT(stats.wasted_decode_tokens, 0u);
  EXPECT_GT(stats.wasted_prefill_tokens, 0u);
  EXPECT_EQ(stats.completed_requests, 6u);
  EXPECT_GT(stats.mean_batch_occupancy(), 1.0);
}

TEST(ContinuousEngine, CancellationCountsTheComputeItDiscards) {
  ContinuousEngine engine(EngineConfig{});
  engine.Submit(Request("abandoned", 16, 100));
  for (int step = 0; step < 3; ++step) {
    ASSERT_TRUE(engine.RunOneIteration());
  }
  ASSERT_TRUE(engine.Cancel("abandoned", "client_cancelled"));
  EXPECT_EQ(engine.stats().wasted_decode_tokens, 3u);
  EXPECT_EQ(engine.stats().wasted_prefill_tokens, 16u);
  EXPECT_EQ(engine.stats().useful_output_tokens, 0u);
  EXPECT_EQ(engine.stats().cancellations, 1u);
}

TEST(ContinuousEngine, ReportsFirstTokenBeforeCompletion) {
  ContinuousEngine engine(EngineConfig{});
  engine.Submit(Request("r", 16, 3));
  ASSERT_TRUE(engine.RunOneIteration());
  EXPECT_EQ(engine.TakeFirstTokens(), std::vector<std::string>{"r"});
  EXPECT_TRUE(engine.TakeResults().empty());
  engine.RunUntilIdle();
  EXPECT_TRUE(engine.TakeFirstTokens().empty());
  ASSERT_EQ(engine.TakeResults().size(), 1u);
}

// ---------------------------------------------------------------------------
// EngineService over real in-process gRPC
// ---------------------------------------------------------------------------

class EngineServiceTest : public ::testing::Test {
 protected:
  void Start(EngineServiceOptions options) {
    EngineConfig config;
    config.kv_blocks = 1024;
    config.max_batch_sequences = 32;
    config.max_prefill_sequences_per_iteration = 32;
    service_ = std::make_unique<EngineService>(config, options);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    stub_ = v1::EngineControl::NewStub(grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
  }
  void TearDown() override {
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + 1s);
    }
    server_.reset();
    service_.reset();
  }

  static v1::DispatchRequest Dispatch(const std::string& id, std::uint32_t output,
                                      std::uint64_t attempt = 1) {
    v1::DispatchRequest request;
    request.mutable_request()->set_request_id(id);
    request.mutable_request()->set_model_revision("m");
    request.mutable_request()->set_tokenizer_revision("t");
    request.mutable_request()->set_prompt_hash("prompt-" + id);
    request.mutable_request()->set_prompt_tokens(32);
    request.mutable_request()->mutable_sampling()->set_max_output_tokens(output);
    request.mutable_fence()->set_partition_id(0);
    request.mutable_fence()->set_ownership_epoch(1);
    request.mutable_fence()->set_attempt_id(attempt);
    return request;
  }

  struct Observed {
    std::vector<v1::ProgressEvent::Kind> kinds;
    std::optional<Clock::time_point> first_token_at;
    std::optional<Clock::time_point> terminal_at;
    std::string failure_reason;
    bool retryable{};
    grpc::Status status;
  };

  Observed Run(const v1::DispatchRequest& request, grpc::ClientContext* context = nullptr) {
    grpc::ClientContext local;
    grpc::ClientContext* ctx = context != nullptr ? context : &local;
    Observed observed;
    auto reader = stub_->Dispatch(ctx, request);
    v1::DispatchEvent event;
    while (reader->Read(&event)) {
      if (!event.has_progress()) {
        continue;
      }
      const auto kind = event.progress().kind();
      observed.kinds.push_back(kind);
      if (kind == v1::ProgressEvent::PREFILL_DONE) {
        observed.first_token_at = Clock::now();
      } else {
        observed.terminal_at = Clock::now();
        observed.failure_reason = event.progress().failure_reason();
        observed.retryable = event.progress().retryable();
      }
    }
    observed.status = reader->Finish();
    return observed;
  }

  std::unique_ptr<EngineService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<v1::EngineControl::Stub> stub_;
};

TEST_F(EngineServiceTest, LateArrivalJoinsRunningBatchInsteadOfWaiting) {
  // 20 ms per iteration: the long request needs ~2 s; the short one ~60 ms.
  Start({.step_delay = 20ms});
  Observed long_run;
  std::thread long_thread([&] { long_run = Run(Dispatch("long", 100)); });
  std::this_thread::sleep_for(200ms);
  const auto short_started = Clock::now();
  const Observed short_run = Run(Dispatch("short", 3));
  long_thread.join();

  ASSERT_TRUE(short_run.terminal_at.has_value());
  ASSERT_TRUE(long_run.terminal_at.has_value());
  EXPECT_LT(*short_run.terminal_at, *long_run.terminal_at);
  EXPECT_LT(*short_run.terminal_at - short_started, 1s);
  EXPECT_EQ(short_run.kinds.back(), v1::ProgressEvent::COMPLETED);
  EXPECT_GE(service_->stats().peak_running, 2u);
}

TEST_F(EngineServiceTest, StreamsFirstTokenBeforeCompletion) {
  Start({.step_delay = 10ms});
  const Observed run = Run(Dispatch("ttft", 30));
  ASSERT_EQ(run.kinds.size(), 2u);
  EXPECT_EQ(run.kinds[0], v1::ProgressEvent::PREFILL_DONE);
  EXPECT_EQ(run.kinds[1], v1::ProgressEvent::COMPLETED);
  EXPECT_GE(*run.terminal_at - *run.first_token_at, 150ms);
}

TEST_F(EngineServiceTest, ClientCancellationFreesTheSequence) {
  Start({.step_delay = 20ms});
  grpc::ClientContext context;
  std::thread canceller([&] {
    std::this_thread::sleep_for(100ms);
    context.TryCancel();
  });
  const Observed run = Run(Dispatch("abandoned", 1000), &context);
  canceller.join();
  EXPECT_EQ(run.status.error_code(), grpc::StatusCode::CANCELLED);
  const auto deadline = Clock::now() + 2s;
  while (service_->stats().cancelled == 0 && Clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_EQ(service_->stats().cancelled, 1u);
  EXPECT_EQ(service_->Capacity().running_seqs(), 0u);
  EXPECT_EQ(service_->Capacity().batch_headroom(), 1024u);
}

TEST_F(EngineServiceTest, FencedCancelRpcSupersedesOnlyTheMatchingAttempt) {
  Start({.step_delay = 20ms});
  Observed run;
  std::thread dispatcher([&] { run = Run(Dispatch("victim", 1000, 7)); });
  std::this_thread::sleep_for(100ms);

  v1::CancelRequest stale;
  stale.set_request_id("victim");
  stale.mutable_fence()->set_ownership_epoch(1);
  stale.mutable_fence()->set_attempt_id(6);
  v1::CancelReply stale_reply;
  grpc::ClientContext stale_context;
  ASSERT_TRUE(stub_->Cancel(&stale_context, stale, &stale_reply).ok());
  EXPECT_FALSE(stale_reply.accepted());

  v1::CancelRequest current = stale;
  current.mutable_fence()->set_attempt_id(7);
  v1::CancelReply current_reply;
  grpc::ClientContext current_context;
  ASSERT_TRUE(stub_->Cancel(&current_context, current, &current_reply).ok());
  EXPECT_TRUE(current_reply.accepted());
  dispatcher.join();
  ASSERT_FALSE(run.kinds.empty());
  EXPECT_EQ(run.kinds.back(), v1::ProgressEvent::FAILED);
  EXPECT_EQ(run.failure_reason, "superseded");
  EXPECT_TRUE(run.retryable);
}

TEST_F(EngineServiceTest, CancellationOffKeepsComputingAbandonedWork) {
  Start({.step_delay = 5ms, .cancellation = false});
  grpc::ClientContext context;
  std::thread canceller([&] {
    std::this_thread::sleep_for(50ms);
    context.TryCancel();
  });
  const Observed run = Run(Dispatch("orphan", 40), &context);
  canceller.join();
  EXPECT_EQ(run.status.error_code(), grpc::StatusCode::CANCELLED);
  const auto deadline = Clock::now() + 3s;
  while (service_->stats().abandoned_output_tokens == 0 && Clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  // The sequence ran to completion although nobody could accept it.
  EXPECT_EQ(service_->stats().abandoned_output_tokens, 40u);
  EXPECT_EQ(service_->stats().cancelled, 0u);
  const std::string json = service_->UtilizationJson("engine-x");
  EXPECT_NE(json.find("\"engine_id\":\"engine-x\""), std::string::npos);
  EXPECT_NE(json.find("\"abandoned_output_tokens\":40"), std::string::npos);
  EXPECT_NE(json.find("\"cancellation\":false"), std::string::npos);
}

TEST_F(EngineServiceTest, ConcurrentDispatchesAllCompleteAndDrain) {
  Start({});
  std::atomic<int> completed{0};
  std::vector<std::thread> clients;
  for (int index = 0; index < 64; ++index) {
    clients.emplace_back([&, index] {
      const Observed run = Run(Dispatch("c-" + std::to_string(index), 8));
      if (!run.kinds.empty() && run.kinds.back() == v1::ProgressEvent::COMPLETED) {
        completed.fetch_add(1);
      }
    });
  }
  for (auto& client : clients) {
    client.join();
  }
  EXPECT_EQ(completed.load(), 64);
  v1::DrainRequest drain;
  v1::DrainReply reply;
  grpc::ClientContext context;
  ASSERT_TRUE(stub_->Drain(&context, drain, &reply).ok());
  EXPECT_EQ(reply.in_flight(), 0u);
  EXPECT_EQ(service_->Capacity().running_seqs(), 0u);
  EXPECT_EQ(service_->stats().completed, 64u);
}

// ---------------------------------------------------------------------------
// Engine selection
// ---------------------------------------------------------------------------

EngineRecord Engine(std::string id, std::uint32_t running, std::uint32_t waiting = 0,
                    std::uint32_t free_kv = 1000, std::uint32_t headroom = 100) {
  EngineRecord record;
  record.engine_id = id;
  record.incarnation_id = id + "-1";
  record.endpoint = "unused";
  record.capacity.set_total_kv_blocks(1000);
  record.capacity.set_free_kv_blocks(free_kv);
  record.capacity.set_running_seqs(running);
  record.capacity.set_waiting_seqs(waiting);
  record.capacity.set_batch_headroom(headroom);
  return record;
}

struct SelectFixture {
  std::vector<EngineRecord> engines;
  std::unordered_map<std::string, std::size_t> inflight;
  std::unordered_set<std::string> excluded;
  infersched::distributed::RoutingPolicy policy{
      infersched::distributed::RoutingPolicy::kLeastLoaded};
  std::size_t ticket{};
  std::optional<std::string> Pick(std::optional<PrefixHint> hint = std::nullopt,
                                  bool* affinity = nullptr) {
    const auto selection = SelectEngine({.engines = engines, .local_inflight = inflight,
                                         .excluded = excluded, .hint = hint,
                                         .affinity_slack = 8, .policy = policy,
                                         .round_robin_ticket = ticket});
    if (!selection) {
      return std::nullopt;
    }
    if (affinity != nullptr) {
      *affinity = selection->affinity;
    }
    return selection->engine.engine_id;
  }
};

TEST(EngineSelector, PicksLeastLoadedUsingLocalInflightAndHeartbeat) {
  SelectFixture fixture{.engines = {Engine("a", 10), Engine("b", 2), Engine("c", 5)}};
  EXPECT_EQ(fixture.Pick(), "b");
  fixture.inflight["b"] = 7;  // This Router's own in-flight work dominates.
  EXPECT_EQ(fixture.Pick(), "c");
}

TEST(EngineSelector, SkipsExcludedSaturatedAndLowKvEngines) {
  SelectFixture fixture{.engines = {Engine("a", 0, 0, 1000, 0), Engine("b", 3),
                                    Engine("c", 0, 0, 10)}};
  EXPECT_EQ(fixture.Pick(), "b");  // a has no headroom, c has <5% KV free.
  fixture.excluded = {"b"};
  EXPECT_EQ(fixture.Pick(), "c");
  fixture.excluded = {"a", "b", "c"};
  EXPECT_FALSE(fixture.Pick().has_value());
}

TEST(EngineSelector, RoundRobinAndFirstIgnoreLoadButHonourExclusion) {
  using infersched::distributed::ParseRoutingPolicy;
  using infersched::distributed::RoutingPolicy;
  EXPECT_EQ(ParseRoutingPolicy("round-robin"), RoutingPolicy::kRoundRobin);
  EXPECT_EQ(ParseRoutingPolicy("first"), RoutingPolicy::kFirst);
  EXPECT_EQ(ParseRoutingPolicy("least-loaded"), RoutingPolicy::kLeastLoaded);
  EXPECT_FALSE(ParseRoutingPolicy("random").has_value());

  SelectFixture fixture{.engines = {Engine("c", 0), Engine("a", 50), Engine("b", 0)}};
  fixture.policy = RoutingPolicy::kRoundRobin;
  std::vector<std::string> picks;
  for (fixture.ticket = 0; fixture.ticket < 4; ++fixture.ticket) {
    picks.push_back(*fixture.Pick(PrefixHint{"b", "b-1"}));
  }
  EXPECT_EQ(picks, (std::vector<std::string>{"a", "b", "c", "a"}));
  fixture.policy = RoutingPolicy::kFirst;
  EXPECT_EQ(fixture.Pick(), "a");  // Even though "a" is the most loaded.
  fixture.excluded = {"a"};
  EXPECT_EQ(fixture.Pick(), "b");
}

TEST(EngineSelector, PrefixAffinityWithinSlackOnlyForTheSameIncarnation) {
  SelectFixture fixture{.engines = {Engine("a", 1), Engine("b", 6)}};
  bool affinity = false;
  EXPECT_EQ(fixture.Pick(PrefixHint{"b", "b-1"}, &affinity), "b");
  EXPECT_TRUE(affinity);
  // A restarted Engine has an empty prefix cache: the old hint is ignored.
  EXPECT_EQ(fixture.Pick(PrefixHint{"b", "b-0"}, &affinity), "a");
  EXPECT_FALSE(affinity);
  // Beyond the slack, load balance wins over cache reuse.
  fixture.engines[1] = Engine("b", 20);
  EXPECT_EQ(fixture.Pick(PrefixHint{"b", "b-1"}, &affinity), "a");
  EXPECT_FALSE(affinity);
  // Hints for unknown Engines are ignored.
  EXPECT_EQ(fixture.Pick(PrefixHint{"z", "z-1"}), "a");
}

// ---------------------------------------------------------------------------
// Redis hints and JSON
// ---------------------------------------------------------------------------

TEST(RedisHints, UnreachableRedisDegradesToNoHintsQuickly) {
  RedisHints hints({.host = "127.0.0.1", .port = 1, .timeout = 20ms,
                    .reconnect_backoff = 10s});
  const auto started = Clock::now();
  for (int index = 0; index < 100; ++index) {
    EXPECT_FALSE(hints.GetPrefixOwner("m", "p").has_value());
    EXPECT_FALSE(hints.PutPrefixOwner("m", "p", {"e", "i"}));
  }
  // One failed connect, then the backoff short-circuits every later call.
  EXPECT_LT(Clock::now() - started, 500ms);
  EXPECT_EQ(hints.stats().errors, 1u);
}

TEST(RedisHints, ParsesAddressesAndDisablesOnEmpty) {
  EXPECT_FALSE(RedisHints::ParseAddress("").has_value());
  EXPECT_FALSE(RedisHints::ParseAddress("none").has_value());
  EXPECT_FALSE(RedisHints::ParseAddress("off").has_value());
  const auto parsed = RedisHints::ParseAddress("redis.internal:6380");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->host, "redis.internal");
  EXPECT_EQ(parsed->port, 6380);
  EXPECT_FALSE(RedisHints::ParseAddress("host:notaport").has_value());
}

TEST(Json, EscapesControlAndQuoteCharacters) {
  const std::string json = infersched::util::JsonObject{}
                               .Add("text", std::string_view("a\"b\\c\n\x01"))
                               .Add("n", std::uint64_t{7})
                               .Add("ok", true)
                               .str();
  EXPECT_EQ(json, "{\"text\":\"a\\\"b\\\\c\\n\\u0001\",\"n\":7,\"ok\":true}");
}

}  // namespace
