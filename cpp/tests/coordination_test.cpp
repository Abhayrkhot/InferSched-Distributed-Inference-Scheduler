// Live Redis tests for prefix-affinity hints (requires `docker compose up`).

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

#include "infersched/coordination/redis_hints.hpp"

namespace {

using namespace std::chrono_literals;
using infersched::coordination::PrefixHint;
using infersched::coordination::RedisHints;

std::string UniquePrefix() {
  return "test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}

TEST(RedisHintsLive, RoundTripsAndOverwritesPrefixOwner) {
  RedisHints hints({.host = "127.0.0.1", .port = 6379});
  const std::string prefix = UniquePrefix();
  EXPECT_FALSE(hints.GetPrefixOwner("model-v1", prefix).has_value());
  ASSERT_TRUE(hints.PutPrefixOwner("model-v1", prefix, {"engine-a", "inc-1"}));
  EXPECT_EQ(hints.GetPrefixOwner("model-v1", prefix), (PrefixHint{"engine-a", "inc-1"}));
  // Namespaced by model revision.
  EXPECT_FALSE(hints.GetPrefixOwner("model-v2", prefix).has_value());
  ASSERT_TRUE(hints.PutPrefixOwner("model-v1", prefix, {"engine-b", "inc-7"}));
  EXPECT_EQ(hints.GetPrefixOwner("model-v1", prefix), (PrefixHint{"engine-b", "inc-7"}));
  EXPECT_EQ(hints.stats().errors, 0u);
}

TEST(RedisHintsLive, HintsExpire) {
  RedisHints hints({.host = "127.0.0.1", .port = 6379, .ttl = 1s});
  const std::string prefix = UniquePrefix();
  ASSERT_TRUE(hints.PutPrefixOwner("model-v1", prefix, {"engine-a", "inc-1"}));
  ASSERT_TRUE(hints.GetPrefixOwner("model-v1", prefix).has_value());
  std::this_thread::sleep_for(2100ms);
  EXPECT_FALSE(hints.GetPrefixOwner("model-v1", prefix).has_value());
}

}  // namespace
