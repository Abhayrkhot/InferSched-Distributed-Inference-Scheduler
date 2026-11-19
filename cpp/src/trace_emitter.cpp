#include "infersched/durable/trace_emitter.hpp"

#include <cstdio>
#include <random>

#include <librdkafka/rdkafkacpp.h>

#include "infersched/util/json.hpp"

namespace infersched::durable {
namespace {

std::int64_t UnixMicros(const std::chrono::system_clock::time_point point) {
  return std::chrono::duration_cast<std::chrono::microseconds>(point.time_since_epoch())
      .count();
}

}  // namespace

TraceEmitter::TraceEmitter(std::string brokers, std::string service, std::string topic)
    : service_(std::move(service)), topic_(std::move(topic)) {
  std::string error;
  std::unique_ptr<RdKafka::Conf> config(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  if (config->set("bootstrap.servers", brokers, error) != RdKafka::Conf::CONF_OK ||
      config->set("linger.ms", "20", error) != RdKafka::Conf::CONF_OK ||
      config->set("queue.buffering.max.messages", "100000", error) !=
          RdKafka::Conf::CONF_OK) {
    return;  // Tracing disabled; requests are unaffected.
  }
  producer_.reset(RdKafka::Producer::create(config.get(), error));
}

TraceEmitter::~TraceEmitter() {
  if (producer_) {
    static_cast<void>(producer_->flush(1000));
  }
}

std::string TraceEmitter::NewSpanId() {
  thread_local std::mt19937_64 generator{std::random_device{}()};
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(generator()));
  return buffer;
}

std::string TraceEmitter::Serialize(const Span& span, const std::string_view service) {
  util::JsonObject attributes;
  for (const auto& [key, value] : span.attributes) {
    attributes.Add(key, value);
  }
  const std::int64_t start = UnixMicros(span.start);
  return util::JsonObject{}
      .Add("trace_id", span.trace_id)
      .Add("span_id", span.span_id)
      .Add("parent_span_id", span.parent_span_id)
      .Add("name", span.name)
      .Add("service", service)
      .Add("start_unix_us", start)
      .Add("duration_us", UnixMicros(span.end) - start)
      .AddRaw("attributes", attributes.str())
      .str();
}

void TraceEmitter::Emit(const Span& span) {
  if (!producer_ || span.trace_id.empty()) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const std::string payload = Serialize(span, service_);
  const RdKafka::ErrorCode error = producer_->produce(
      topic_, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
      const_cast<char*>(payload.data()), payload.size(), span.trace_id.data(),
      span.trace_id.size(), 0, nullptr);
  producer_->poll(0);
  if (error == RdKafka::ERR_NO_ERROR) {
    emitted_.fetch_add(1, std::memory_order_relaxed);
  } else {
    dropped_.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace infersched::durable
