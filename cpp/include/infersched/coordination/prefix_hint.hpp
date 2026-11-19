#pragma once

#include <string>

namespace infersched::coordination {

// Disposable "this Engine incarnation probably holds this prefix's KV blocks"
// hint. A hint for a different incarnation is stale: a restarted Engine starts
// with an empty prefix cache.
struct PrefixHint {
  std::string engine_id;
  std::string incarnation_id;
  bool operator==(const PrefixHint&) const = default;
};

}  // namespace infersched::coordination
