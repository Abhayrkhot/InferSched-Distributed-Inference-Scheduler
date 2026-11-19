#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>

namespace infersched::util {

// Appends `value` as a JSON string literal (RFC 8259 escaping).
inline void AppendJsonString(std::string& out, const std::string_view value) {
  out.push_back('"');
  for (const char raw : value) {
    const auto c = static_cast<unsigned char>(raw);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char escaped[7];
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
          out += escaped;
        } else {
          out.push_back(raw);
        }
    }
  }
  out.push_back('"');
}

// Minimal flat JSON object builder; values are escaped, keys are trusted.
class JsonObject {
 public:
  JsonObject& Add(const std::string_view key, const std::string_view value) {
    Key(key);
    AppendJsonString(body_, value);
    return *this;
  }
  JsonObject& Add(const std::string_view key, const char* value) {
    return Add(key, std::string_view(value));
  }
  template <typename Integer>
    requires(std::is_integral_v<Integer> && !std::is_same_v<Integer, bool>)
  JsonObject& Add(const std::string_view key, const Integer value) {
    Key(key);
    body_ += std::to_string(value);
    return *this;
  }
  JsonObject& Add(const std::string_view key, const bool value) {
    Key(key);
    body_ += value ? "true" : "false";
    return *this;
  }
  // Embeds an already-serialized JSON value (object, array, number).
  JsonObject& AddRaw(const std::string_view key, const std::string_view json) {
    Key(key);
    body_.append(json);
    return *this;
  }
  [[nodiscard]] std::string str() const { return body_.empty() ? "{}" : body_ + "}"; }

 private:
  void Key(const std::string_view key) {
    body_.push_back(body_.empty() ? '{' : ',');
    AppendJsonString(body_, key);
    body_.push_back(':');
  }
  std::string body_;
};

}  // namespace infersched::util
