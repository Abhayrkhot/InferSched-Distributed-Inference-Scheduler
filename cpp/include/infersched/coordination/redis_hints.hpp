#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "infersched/coordination/prefix_hint.hpp"

struct redisContext;

namespace infersched::coordination {

struct RedisHintsOptions {
  std::string host{"127.0.0.1"};
  int port{6379};
  std::chrono::seconds ttl{120};
  std::chrono::milliseconds timeout{25};
  // After a connection or command error, skip Redis for this long instead of
  // paying the timeout on every request.
  std::chrono::milliseconds reconnect_backoff{1000};
};

struct RedisHintsStats {
  std::uint64_t lookups{};
  std::uint64_t hits{};
  std::uint64_t writes{};
  std::uint64_t errors{};
};

// Shared prefix-affinity hints in Redis. Every operation is best effort:
// Redis being down, slow, or flushed only costs cache locality, never
// correctness (Postgres fences every completion). Thread-safe.
class RedisHints {
 public:
  explicit RedisHints(RedisHintsOptions options);
  ~RedisHints();
  RedisHints(const RedisHints&) = delete;
  RedisHints& operator=(const RedisHints&) = delete;

  // Parses "host:port". "", "none", or "off" yield nullopt (hints disabled).
  [[nodiscard]] static std::optional<RedisHintsOptions> ParseAddress(
      std::string_view address);

  [[nodiscard]] std::optional<PrefixHint> GetPrefixOwner(
      std::string_view model_revision, std::string_view prefix_id);
  bool PutPrefixOwner(std::string_view model_revision, std::string_view prefix_id,
                      const PrefixHint& hint);
  [[nodiscard]] RedisHintsStats stats() const;

  [[nodiscard]] static std::string PrefixKey(std::string_view model_revision,
                                             std::string_view prefix_id);

 private:
  bool EnsureConnectedLocked();
  void FailLocked();

  const RedisHintsOptions options_;
  mutable std::mutex mutex_;
  redisContext* context_{};
  std::chrono::steady_clock::time_point retry_after_{};
  RedisHintsStats stats_;
};

}  // namespace infersched::coordination
