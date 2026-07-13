#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>

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

infersched::v1::InferenceRequest BuildRequest(const std::string& request_id) {
  infersched::v1::InferenceRequest request;
  request.set_request_id(request_id);
  request.set_model_id("benchmark-model");
  request.set_model_revision("benchmark-model-v1");
  request.set_tokenizer_revision("tokenizer-v1");
  request.set_prompt_hash("p5-prompt");
  request.set_prompt_tokens(16);
  request.mutable_sampling()->set_max_output_tokens(4);
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
    const std::string prefix = "p6-batch-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    infersched::durable::KafkaPoller results(
        "localhost:9092", prefix, "inference.results");
    results.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    infersched::durable::KafkaOutboxPublisher publisher("localhost:9092");
    std::unordered_set<std::string> pending;
    for (std::size_t index = 0; index < count; ++index) {
      const std::string id = prefix + "-" + std::to_string(index);
      pending.insert(id);
      if (!publisher.Publish(infersched::durable::OutboxRecord{
              .topic = "inference.requests", .message_key = id,
              .payload = BuildRequest(id).SerializeAsString()})) {
        results.Stop();
        return EXIT_FAILURE;
      }
    }
    if (!published_marker.empty()) {
      std::ofstream marker(published_marker);
      marker << prefix;
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(35);
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
