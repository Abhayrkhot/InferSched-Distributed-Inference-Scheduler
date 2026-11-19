#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "infersched/durable/durable_store.hpp"
#include "infersched/durable/kafka_poller.hpp"
#include "infersched/durable/outbox_relay.hpp"
#include "infersched.pb.h"

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

struct Workload {
  std::size_t prefixes{};          // 0: legacy identical prompt, no shared prefix.
  std::uint32_t max_output_tokens{4};
  std::string prefix_namespace{"p8-prefix"};
};

infersched::v1::InferenceRequest BuildRequest(const std::string& request_id,
                                              const Workload& workload = {},
                                              const std::size_t index = 0) {
  infersched::v1::InferenceRequest request;
  request.set_request_id(request_id);
  request.set_model_id("benchmark-model");
  request.set_model_revision("benchmark-model-v1");
  request.set_tokenizer_revision("tokenizer-v1");
  request.set_trace_id("trace-" + request_id);
  request.set_span_id("span-" + std::to_string(index));
  request.mutable_sampling()->set_max_output_tokens(workload.max_output_tokens);
  if (workload.prefixes == 0) {
    request.set_prompt_hash("p5-prompt");
    request.set_prompt_tokens(16);
    return request;
  }
  // Distinct prompts that share one of K 64-token prefixes.
  request.set_prompt_hash("prompt-" + request_id);
  request.set_prompt_tokens(96);
  request.set_prefix_id(workload.prefix_namespace + '-' +
                        std::to_string(index % workload.prefixes));
  request.set_prefix_tokens(64);
  return request;
}

bool AwaitResult(const std::string& request_id) {
  infersched::durable::KafkaPoller results(
      "localhost:9092", request_id, "inference.results");
  results.Start();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto message = results.Pop(std::chrono::milliseconds(250));
    if (message.has_value() && message->key == request_id) {
      results.MarkPublishComplete(message->partition, message->offset);
      results.Stop();
      return true;
    }
  }
  results.Stop();
  return false;
}

}  // namespace

int main(const int argc, char** argv) {
  const std::string mode = Argument(argc, argv, "--mode", "normal");
  const std::string request_file =
      Argument(argc, argv, "--request-file", "/tmp/infersched-p5-recovery-id");
  const std::size_t count = static_cast<std::size_t>(
      std::stoull(Argument(argc, argv, "--count", "1")));
  const std::string published_marker =
      Argument(argc, argv, "--published-marker", "");
  const Workload workload{
      .prefixes = static_cast<std::size_t>(std::stoull(Argument(argc, argv, "--prefixes", "0"))),
      .max_output_tokens = static_cast<std::uint32_t>(
          std::stoul(Argument(argc, argv, "--max-output-tokens", "4"))),
      .prefix_namespace = Argument(argc, argv, "--prefix-namespace", "p8-prefix")};
  const std::string prefix_override = Argument(argc, argv, "--batch-prefix", "");
  const auto wait_seconds = std::chrono::seconds(
      std::stoll(Argument(argc, argv, "--wait-seconds", "35")));
  if (mode == "seed") {
    const std::string request_id = "p5-recovery-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto request = BuildRequest(request_id);
    infersched::durable::DurableStore store(
        "host=localhost port=55432 dbname=infersched user=infersched "
        "password=infersched");
    store.Migrate();
    const std::int64_t offset =
        std::chrono::steady_clock::now().time_since_epoch().count();
    if (!store.Ingest(request_id, 3, offset, request.SerializeAsString())) {
      return EXIT_FAILURE;
    }
    std::ofstream output(request_file);
    output << request_id;
    return output.good() ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  std::string request_id;
  if (mode == "await") {
    std::ifstream input(request_file);
    input >> request_id;
    if (request_id.empty()) {
      return EXIT_FAILURE;
    }
  } else if (mode == "batch") {
    const std::string prefix =
        prefix_override.empty()
            ? "p6-batch-" + std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count())
            : prefix_override;
    infersched::durable::KafkaPoller results(
        "localhost:9092", prefix, "inference.results");
    results.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    infersched::durable::KafkaOutboxPublisher publisher("localhost:9092",
                                                       std::chrono::seconds(30));
    std::unordered_set<std::string> pending;
    std::vector<infersched::durable::OutboxRecord> records;
    records.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const std::string id = prefix + "-" + std::to_string(index);
      pending.insert(id);
      records.push_back(infersched::durable::OutboxRecord{
          .topic = "inference.requests", .message_key = id,
          .payload = BuildRequest(id, workload, index).SerializeAsString()});
    }
    // One batched produce: the burst reaches Kafka at broker speed instead
    // of one acknowledged round trip per request.
    const auto delivered = publisher.PublishMany(records);
    if (std::find(delivered.begin(), delivered.end(), false) != delivered.end()) {
      results.Stop();
      return EXIT_FAILURE;
    }
    if (!published_marker.empty()) {
      std::ofstream marker(published_marker);
      marker << prefix;
    }
    const auto deadline = std::chrono::steady_clock::now() + wait_seconds;
    while (!pending.empty() && std::chrono::steady_clock::now() < deadline) {
      const auto message = results.Pop(std::chrono::milliseconds(250));
      if (message.has_value() && pending.erase(message->key) == 1) {
        results.MarkPublishComplete(message->partition, message->offset);
      }
    }
    results.Stop();
    std::cout << "{\"submitted\":" << count << ",\"completed\":"
              << count - pending.size() << "}\n";
    return pending.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
  } else {
    request_id = "p5-e2e-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    infersched::durable::KafkaOutboxPublisher publisher("localhost:9092");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (!publisher.Publish(infersched::durable::OutboxRecord{
            .topic = "inference.requests", .message_key = request_id,
            .payload = BuildRequest(request_id).SerializeAsString()})) {
      return EXIT_FAILURE;
    }
  }
  const bool completed = AwaitResult(request_id);
  std::cout << "{\"request_id\":\"" << request_id
            << "\",\"completed\":" << std::boolalpha << completed << "}\n";
  return completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
