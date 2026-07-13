#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infersched::concurrency {

template <typename Value>
class ShardedRegistry {
 public:
  explicit ShardedRegistry(const std::size_t shard_count) {
    if (shard_count == 0) {
      throw std::invalid_argument("registry shard count must be positive");
    }
    shards_.reserve(shard_count);
    for (std::size_t index = 0; index < shard_count; ++index) {
      shards_.push_back(std::make_unique<Shard>());
    }
  }

  void Upsert(std::string key, Value value) {
    Shard& shard = ShardFor(key);
    std::unique_lock lock(shard.mutex);
    shard.values.insert_or_assign(std::move(key), std::move(value));
  }

  [[nodiscard]] std::optional<Value> Get(const std::string_view key) const {
    const Shard& shard = ShardFor(key);
    std::shared_lock lock(shard.mutex);
    const auto iterator = shard.values.find(std::string(key));
    return iterator == shard.values.end() ? std::nullopt
                                          : std::optional<Value>{iterator->second};
  }

  [[nodiscard]] bool Erase(const std::string_view key) {
    Shard& shard = ShardFor(key);
    std::unique_lock lock(shard.mutex);
    return shard.values.erase(std::string(key)) != 0;
  }

  [[nodiscard]] std::size_t size() const {
    std::size_t total = 0;
    for (const auto& shard : shards_) {
      std::shared_lock lock(shard->mutex);
      total += shard->values.size();
    }
    return total;
  }

 private:
  struct Shard {
    mutable std::shared_mutex mutex;
    std::unordered_map<std::string, Value> values;
  };

  [[nodiscard]] Shard& ShardFor(const std::string_view key) {
    return *shards_[std::hash<std::string_view>{}(key) % shards_.size()];
  }
  [[nodiscard]] const Shard& ShardFor(const std::string_view key) const {
    return *shards_[std::hash<std::string_view>{}(key) % shards_.size()];
  }

  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace infersched::concurrency
