#pragma once

#include <cstddef>
#include <list>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace infersched::engine {

class ResultCache {
 public:
  explicit ResultCache(std::size_t capacity);

  void Put(std::string key, std::string result);
  [[nodiscard]] std::optional<std::string> Get(std::string_view key);
  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
  [[nodiscard]] std::size_t misses() const noexcept { return misses_; }

 private:
  struct Entry {
    std::string result;
    std::list<std::string>::iterator recency;
  };

  std::size_t capacity_;
  std::list<std::string> recency_;  // most-recent at front
  std::unordered_map<std::string, Entry> entries_;
  std::size_t hits_{};
  std::size_t misses_{};
};

}  // namespace infersched::engine
