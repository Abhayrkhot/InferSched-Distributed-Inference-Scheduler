#include "infersched/coordination/redis_hints.hpp"

#include <charconv>
#include <sys/time.h>

#include <hiredis/hiredis.h>

namespace infersched::coordination {
namespace {

timeval ToTimeval(const std::chrono::milliseconds value) {
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(value).count();
  return timeval{.tv_sec = static_cast<decltype(timeval::tv_sec)>(micros / 1'000'000),
                 .tv_usec = static_cast<decltype(timeval::tv_usec)>(micros % 1'000'000)};
}

struct ReplyDeleter {
  void operator()(redisReply* reply) const { freeReplyObject(reply); }
};

}  // namespace

RedisHints::RedisHints(RedisHintsOptions options) : options_(std::move(options)) {}

RedisHints::~RedisHints() {
  if (context_ != nullptr) {
    redisFree(context_);
  }
}

std::optional<RedisHintsOptions> RedisHints::ParseAddress(const std::string_view address) {
  if (address.empty() || address == "none" || address == "off") {
    return std::nullopt;  // Hints disabled.
  }
  RedisHintsOptions options;
  const auto colon = address.rfind(':');
  options.host = std::string(address.substr(0, colon));
  if (colon != std::string_view::npos) {
    const auto port = address.substr(colon + 1);
    int parsed = 0;
    const auto [end, error] = std::from_chars(port.data(), port.data() + port.size(), parsed);
    if (error != std::errc{} || end != port.data() + port.size()) {
      return std::nullopt;
    }
    options.port = parsed;
  }
  return options;
}

std::string RedisHints::PrefixKey(const std::string_view model_revision,
                                  const std::string_view prefix_id) {
  std::string key = "infersched:prefix:v1:";
  key.append(model_revision);
  key.push_back(':');
  key.append(prefix_id);
  return key;
}

bool RedisHints::EnsureConnectedLocked() {
  if (context_ != nullptr) {
    return true;
  }
  if (std::chrono::steady_clock::now() < retry_after_) {
    return false;
  }
  context_ = redisConnectWithTimeout(options_.host.c_str(), options_.port,
                                     ToTimeval(options_.timeout));
  if (context_ == nullptr || context_->err != 0 ||
      redisSetTimeout(context_, ToTimeval(options_.timeout)) != REDIS_OK) {
    FailLocked();
    return false;
  }
  return true;
}

void RedisHints::FailLocked() {
  ++stats_.errors;
  if (context_ != nullptr) {
    redisFree(context_);
    context_ = nullptr;
  }
  retry_after_ = std::chrono::steady_clock::now() + options_.reconnect_backoff;
}

std::optional<PrefixHint> RedisHints::GetPrefixOwner(const std::string_view model_revision,
                                                     const std::string_view prefix_id) {
  if (prefix_id.empty()) {
    return std::nullopt;
  }
  std::lock_guard lock(mutex_);
  ++stats_.lookups;
  if (!EnsureConnectedLocked()) {
    return std::nullopt;
  }
  const std::string key = PrefixKey(model_revision, prefix_id);
  std::unique_ptr<redisReply, ReplyDeleter> reply(static_cast<redisReply*>(
      redisCommand(context_, "GET %b", key.data(), key.size())));
  if (!reply || reply->type == REDIS_REPLY_ERROR) {
    FailLocked();
    return std::nullopt;
  }
  if (reply->type != REDIS_REPLY_STRING) {
    return std::nullopt;  // Miss (nil) or a key someone else overwrote oddly.
  }
  const std::string_view value(reply->str, reply->len);
  const auto separator = value.find('\n');
  if (separator == std::string_view::npos) {
    return std::nullopt;
  }
  ++stats_.hits;
  return PrefixHint{.engine_id = std::string(value.substr(0, separator)),
                    .incarnation_id = std::string(value.substr(separator + 1))};
}

bool RedisHints::PutPrefixOwner(const std::string_view model_revision,
                                const std::string_view prefix_id,
                                const PrefixHint& hint) {
  if (prefix_id.empty()) {
    return false;
  }
  std::lock_guard lock(mutex_);
  if (!EnsureConnectedLocked()) {
    return false;
  }
  const std::string key = PrefixKey(model_revision, prefix_id);
  const std::string value = hint.engine_id + '\n' + hint.incarnation_id;
  const std::string ttl = std::to_string(options_.ttl.count());
  std::unique_ptr<redisReply, ReplyDeleter> reply(static_cast<redisReply*>(
      redisCommand(context_, "SET %b %b EX %s", key.data(), key.size(), value.data(),
                   value.size(), ttl.c_str())));
  if (!reply || reply->type == REDIS_REPLY_ERROR) {
    FailLocked();
    return false;
  }
  ++stats_.writes;
  return true;
}

RedisHintsStats RedisHints::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

}  // namespace infersched::coordination
