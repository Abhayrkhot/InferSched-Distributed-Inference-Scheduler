#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace RdKafka {
class Producer;
}

namespace infersched::durable {

struct Span {
  std::string trace_id;
  std::string span_id;
  std::string parent_span_id;
  std::string name;
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  // Serialized as a flat JSON object of string values.
  std::vector<std::pair<std::string, std::string>> attributes;
};

// Fire-and-forget span export to Kafka (`inference.traces`), keyed by
// trace_id so one trace's spans land on one partition. Dropping a span never
// blocks or fails request processing. Thread-safe.
class TraceEmitter {
 public:
  TraceEmitter(std::string brokers, std::string service,
               std::string topic = "inference.traces");
  ~TraceEmitter();
  TraceEmitter(const TraceEmitter&) = delete;
  TraceEmitter& operator=(const TraceEmitter&) = delete;

  void Emit(const Span& span);
  [[nodiscard]] static std::string NewSpanId();
  [[nodiscard]] static std::string Serialize(const Span& span, std::string_view service);

  [[nodiscard]] std::uint64_t emitted() const { return emitted_.load(); }
  [[nodiscard]] std::uint64_t dropped() const { return dropped_.load(); }

 private:
  std::string service_;
  std::string topic_;
  std::unique_ptr<RdKafka::Producer> producer_;
  std::atomic<std::uint64_t> emitted_{};
  std::atomic<std::uint64_t> dropped_{};
};

}  // namespace infersched::durable
