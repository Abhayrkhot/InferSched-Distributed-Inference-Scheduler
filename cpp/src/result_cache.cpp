#include "infersched/engine/result_cache.hpp"

#include <stdexcept>
#include <utility>

namespace infersched::engine {

ResultCache::ResultCache(const std::size_t capacity) : capacity_(capacity) {
  if (capacity == 0) {
    throw std::invalid_argument("result cache capacity must be positive");
  }
}

void ResultCache::Put(std::string key, std::string result) {
  const auto existing = entries_.find(key);
  if (existing != entries_.end()) {
    existing->second.result = std::move(result);
    recency_.splice(recency_.begin(), recency_, existing->second.recency);
    return;
  }

  if (entries_.size() == capacity_) {
    entries_.erase(recency_.back());
    recency_.pop_back();
  }
  recency_.push_front(std::move(key));
  entries_.emplace(recency_.front(), Entry{std::move(result), recency_.begin()});
}

std::optional<std::string> ResultCache::Get(const std::string_view key) {
  const auto iterator = entries_.find(std::string(key));
  if (iterator == entries_.end()) {
    ++misses_;
    return std::nullopt;
  }
  ++hits_;
  recency_.splice(recency_.begin(), recency_, iterator->second.recency);
  return iterator->second.result;
}

}  // namespace infersched::engine
