// Real-model serving benchmark: ContinuousEngine scheduling a GGUF model run
// by llama.cpp, driven in wall-clock time by an open-loop arrival schedule.
//
// Every number this prints is measured: tokens come out of the model, latency
// is wall time from a request's scheduled arrival, and GPU telemetry comes
// from NVML. The same workload can be dumped as token ids (--dump-workload) and
// replayed against llama-server or vLLM by scripts/bench_openai_like.py.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "infersched/engine/continuous_engine.hpp"
#include "infersched/engine/gpu_monitor.hpp"
#include "infersched/engine/llama_executor.hpp"
#include "infersched/engine/prompt_blocks.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using infersched::engine::ContinuousEngine;
using infersched::engine::EngineConfig;
using infersched::engine::EngineRequest;
using infersched::engine::GpuMonitor;
using infersched::engine::LlamaExecutor;
using infersched::engine::LlamaExecutorConfig;

// The last occurrence wins, so a driver can append overrides to defaults.
std::string Arg(const int argc, char** argv, const std::string& name,
                const std::string& fallback) {
  std::string value = fallback;
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      value = argv[index + 1];
    }
  }
  return value;
}
bool Flag(const int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) {
      return true;
    }
  }
  return false;
}

struct WorkItem {
  double arrival_s{};
  std::size_t prompt_tokens{};
  std::size_t output_tokens{};
  std::string prefix_id;
};

struct Workload {
  std::vector<WorkItem> items;
  std::size_t prefix_tokens{};
};

Workload MakeWorkload(const int argc, char** argv, const std::size_t count,
                      const std::uint64_t seed) {
  const double qps = std::stod(Arg(argc, argv, "--qps", "0"));
  const auto prompt_min = std::stoull(Arg(argc, argv, "--prompt-min", "128"));
  const auto prompt_max = std::stoull(Arg(argc, argv, "--prompt-max", "512"));
  const auto short_min = std::stoull(Arg(argc, argv, "--short-min", "16"));
  const auto short_max = std::stoull(Arg(argc, argv, "--short-max", "64"));
  const auto long_min = std::stoull(Arg(argc, argv, "--long-min", "256"));
  const auto long_max = std::stoull(Arg(argc, argv, "--long-max", "512"));
  const double long_fraction = std::stod(Arg(argc, argv, "--long-fraction", "0.2"));
  const auto prefixes = std::stoull(Arg(argc, argv, "--prefixes", "0"));
  Workload workload;
  workload.prefix_tokens = std::stoull(Arg(argc, argv, "--prefix-tokens", "0"));

  std::mt19937_64 random(seed);
  std::exponential_distribution<double> gap(qps > 0.0 ? qps : 1.0);
  std::uniform_int_distribution<std::size_t> prompt(prompt_min, prompt_max);
  std::uniform_int_distribution<std::size_t> short_out(short_min, short_max);
  std::uniform_int_distribution<std::size_t> long_out(long_min, long_max);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_int_distribution<std::size_t> prefix(0, prefixes > 0 ? prefixes - 1 : 0);
  double now = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    WorkItem item;
    if (qps > 0.0 && index > 0) {
      now += gap(random);
    }
    item.arrival_s = qps > 0.0 ? now : 0.0;
    item.prompt_tokens = prompt(random) + (prefixes > 0 ? workload.prefix_tokens : 0);
    item.output_tokens = unit(random) < long_fraction ? long_out(random) : short_out(random);
    if (prefixes > 0) {
      item.prefix_id = "prefix-" + std::to_string(prefix(random));
    }
    workload.items.push_back(std::move(item));
  }
  return workload;
}

EngineRequest ToRequest(const std::string& id, const std::string& prompt_hash,
                        const WorkItem& item, const std::size_t prefix_tokens,
                        const std::size_t block_size) {
  EngineRequest request;
  request.request_id = id;
  request.model_revision = "bench-model";
  request.tokenizer_revision = "bench-tokenizer";
  request.prompt_hash = prompt_hash;
  request.prompt_tokens = item.prompt_tokens;
  request.max_output_tokens = item.output_tokens;
  request.prompt_block_hashes = infersched::engine::PromptBlockHashes(
      prompt_hash, item.prefix_id, prefix_tokens, item.prompt_tokens, block_size);
  return request;
}

double Percentile(std::vector<double> values, const double quantile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = quantile * static_cast<double>(values.size() - 1);
  const auto low = static_cast<std::size_t>(std::floor(rank));
  const auto high = static_cast<std::size_t>(std::ceil(rank));
  const double weight = rank - static_cast<double>(low);
  return values[low] * (1.0 - weight) + values[high] * weight;
}

double Seconds(const Clock::duration duration) {
  return std::chrono::duration<double>(duration).count();
}

std::string JsonString(const std::string& value) {
  std::string escaped = "\"";
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      escaped.push_back('\\');
    }
    if (static_cast<unsigned char>(character) >= 0x20) {
      escaped.push_back(character);
    }
  }
  return escaped + '"';
}

}  // namespace

int main(const int argc, char** argv) {
  const std::string model = Arg(argc, argv, "--model", "");
  if (model.empty()) {
    std::cerr << "--model PATH.gguf is required\n";
    return EXIT_FAILURE;
  }
  const std::string policy = Arg(argc, argv, "--policy", "continuous");
  if (policy != "continuous" && policy != "static" && policy != "serial") {
    std::cerr << "--policy must be continuous, static or serial\n";
    return EXIT_FAILURE;
  }
  const std::size_t requests = std::stoull(Arg(argc, argv, "--requests", "200"));
  const std::size_t warmup = std::stoull(Arg(argc, argv, "--warmup", "16"));
  const auto seed = static_cast<std::uint64_t>(std::stoull(Arg(argc, argv, "--seed", "1")));

  EngineConfig config;
  config.kv_block_size_tokens = 16;
  config.kv_blocks = std::stoull(Arg(argc, argv, "--kv-blocks", "2048"));
  config.max_batch_sequences =
      policy == "serial" ? 1 : std::stoull(Arg(argc, argv, "--max-batch", "32"));
  config.max_batch_tokens = std::stoull(Arg(argc, argv, "--max-batch-tokens", "2048"));
  if (policy == "static") {
    // Request-level batching admits a batch only when the previous one has
    // drained, so a per-step prompt budget would cap the batch at a few
    // requests. Let it prefill a whole batch in one step instead (llama.cpp
    // still splits that step into --ubatch-tokens micro-batches).
    config.max_batch_tokens = std::max<std::size_t>(config.max_batch_tokens,
                                                    config.max_batch_sequences * 1024);
  }
  // Request-level batching fills the whole batch at once, so it may admit a
  // full batch per iteration; continuous batching caps prompts per step.
  config.max_prefill_sequences_per_iteration =
      policy == "static" ? config.max_batch_sequences
                         : std::stoull(Arg(argc, argv, "--max-prefill-seqs", "8"));
  config.admission = policy == "static" ? infersched::engine::AdmissionPolicy::kStatic
                                        : infersched::engine::AdmissionPolicy::kContinuous;
  config.prefix_cache_entries = std::stoull(Arg(argc, argv, "--prefix-cache-entries", "1024"));
  config.result_cache_entries = 1;  // Every prompt is unique; no result reuse.
  config.prefix_caching = Arg(argc, argv, "--prefix-cache", "on") != "off";

  LlamaExecutorConfig llama;
  llama.model_path = model;
  llama.gpu_layers = std::stoi(Arg(argc, argv, "--gpu-layers", "999"));
  llama.kv_tokens = config.kv_blocks * config.kv_block_size_tokens;
  llama.block_size_tokens = config.kv_block_size_tokens;
  llama.max_sequences = config.max_batch_sequences;
  llama.holder_sequences = std::stoull(Arg(argc, argv, "--holders", "32"));
  llama.max_batch_tokens = config.max_batch_tokens;
  llama.ubatch_tokens = std::stoull(Arg(argc, argv, "--ubatch-tokens", "512"));
  llama.flash_attention = Arg(argc, argv, "--flash-attn", "on") != "off";
  llama.threads = std::stoi(Arg(argc, argv, "--threads", "4"));
  llama.backend_sampling = Arg(argc, argv, "--backend-sampling", "off") == "on";

  auto executor_owner = std::make_unique<LlamaExecutor>(llama);
  LlamaExecutor& executor = *executor_owner;
  ContinuousEngine engine(config, std::move(executor_owner));

  const Workload workload = MakeWorkload(argc, argv, requests, seed);
  if (const std::string path = Arg(argc, argv, "--dump-workload", ""); !path.empty()) {
    std::ofstream out(path);
    for (std::size_t index = 0; index < workload.items.size(); ++index) {
      const auto& item = workload.items[index];
      const auto request = ToRequest("r" + std::to_string(index),
                                     "w" + std::to_string(seed) + "-" + std::to_string(index),
                                     item, workload.prefix_tokens, config.kv_block_size_tokens);
      out << "{\"id\":\"r" << index << "\",\"arrival_s\":" << item.arrival_s
          << ",\"max_tokens\":" << item.output_tokens << ",\"prefix_id\":"
          << JsonString(item.prefix_id) << ",\"prompt\":[";
      const auto tokens = executor.PromptTokens(request);
      for (std::size_t position = 0; position < tokens.size(); ++position) {
        out << (position == 0 ? "" : ",") << tokens[position];
      }
      out << "]}\n";
    }
    if (Flag(argc, argv, "--dump-only")) {
      return EXIT_SUCCESS;
    }
  }
  std::ofstream outputs_file;
  if (const std::string path = Arg(argc, argv, "--dump-outputs", ""); !path.empty()) {
    outputs_file.open(path);
    executor.SetCompletionObserver(
        [&outputs_file](const std::string& id, const std::vector<std::int32_t>& tokens) {
          if (id.starts_with("warm-")) {
            return;
          }
          outputs_file << "{\"id\":\"" << id << "\",\"tokens\":[";
          for (std::size_t index = 0; index < tokens.size(); ++index) {
            outputs_file << (index == 0 ? "" : ",") << tokens[index];
          }
          outputs_file << "]}\n";
        });
  }

  // Warm-up: load kernels and allocate buffers before the clock starts.
  for (std::size_t index = 0; index < warmup; ++index) {
    WorkItem item{.arrival_s = 0, .prompt_tokens = 256, .output_tokens = 32, .prefix_id = {}};
    auto request = ToRequest("warm-" + std::to_string(index), "warm-" + std::to_string(index),
                             item, 0, config.kv_block_size_tokens);
    request.arrival_time = engine.clock().Now();
    engine.Submit(std::move(request));
  }
  engine.RunUntilIdle();
  static_cast<void>(engine.TakeResults());
  static_cast<void>(engine.TakeFirstTokens());
  engine.ClearCaches();
  const auto before = engine.stats();
  const auto executor_before = executor.stats();

  const std::size_t count = workload.items.size();
  std::unordered_map<std::string, std::size_t> index_of;
  index_of.reserve(count);
  std::vector<Clock::time_point> first_token(count);
  std::vector<Clock::time_point> finished(count);
  std::vector<bool> succeeded(count, false);
  std::ofstream trace;
  if (const std::string path = Arg(argc, argv, "--step-trace", ""); !path.empty()) {
    trace.open(path);
  }
  GpuMonitor gpu;
  gpu.Start();
  const Clock::time_point start = Clock::now();
  const auto arrival = [&](const std::size_t index) {
    return start + std::chrono::duration_cast<Clock::duration>(
                       std::chrono::duration<double>(workload.items[index].arrival_s));
  };
  std::size_t next = 0;
  std::size_t done = 0;
  while (done < count) {
    const auto now = Clock::now();
    while (next < count && arrival(next) <= now) {
      const std::string id = "r" + std::to_string(next);
      index_of.emplace(id, next);
      auto request = ToRequest(id, "w" + std::to_string(seed) + "-" + std::to_string(next),
                               workload.items[next], workload.prefix_tokens,
                               config.kv_block_size_tokens);
      request.arrival_time = engine.clock().Now();
      engine.Submit(std::move(request));
      ++next;
    }
    if (engine.idle()) {
      if (next < count) {
        std::this_thread::sleep_until(arrival(next));
      }
      continue;
    }
    const std::size_t calls_before = executor.stats().decode_calls;
    const bool progressed = engine.RunOneIteration();
    const auto after = Clock::now();
    if (trace.is_open() && executor.stats().decode_calls != calls_before) {
      const auto& step = executor.last_step();
      trace << "{\"prompt\":" << step.prompt_tokens << ",\"decode\":" << step.decode_tokens
            << ",\"context\":" << step.context_tokens << ",\"us\":"
            << std::chrono::duration<double, std::micro>(step.duration).count() << "}\n";
    }
    for (const auto& id : engine.TakeFirstTokens()) {
      first_token[index_of.at(id)] = after;
    }
    for (const auto& result : engine.TakeResults()) {
      const std::size_t index = index_of.at(result.request_id);
      finished[index] = after;
      succeeded[index] = result.succeeded;
      if (first_token[index] == Clock::time_point{}) {
        first_token[index] = after;
      }
      ++done;
    }
    if (!progressed) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  const Clock::time_point end = *std::max_element(finished.begin(), finished.end());
  const auto gpu_summary = gpu.Stop();
  const auto& stats = engine.stats();
  const auto& executor_stats = executor.stats();

  std::vector<double> ttft;
  std::vector<double> tpot;
  std::vector<double> e2e;
  std::size_t output_tokens = 0;
  std::size_t prompt_tokens = 0;
  std::size_t failed = 0;
  for (std::size_t index = 0; index < count; ++index) {
    if (!succeeded[index]) {
      ++failed;
      continue;
    }
    const auto& item = workload.items[index];
    output_tokens += item.output_tokens;
    prompt_tokens += item.prompt_tokens;
    ttft.push_back(Seconds(first_token[index] - arrival(index)) * 1e3);
    e2e.push_back(Seconds(finished[index] - arrival(index)) * 1e3);
    if (item.output_tokens > 1) {
      tpot.push_back(Seconds(finished[index] - first_token[index]) * 1e3 /
                     static_cast<double>(item.output_tokens - 1));
    }
  }
  const double duration = Seconds(end - start);
  const double busy = std::chrono::duration<double>(stats.simulated_busy_time -
                                                    before.simulated_busy_time)
                          .count();
  const std::size_t iterations = stats.iterations - before.iterations;
  const std::size_t slots = stats.decode_token_slots - before.decode_token_slots;
  const auto fixed = [](const double value, const int digits = 3) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(digits);
    out << value;
    return out.str();
  };

  std::ostringstream json;
  json << "{\"label\":" << JsonString(Arg(argc, argv, "--label", policy))
       << ",\"model\":" << JsonString(executor.model_description())
       << ",\"model_params\":" << executor.model_parameters()
       << ",\"model_bytes\":" << executor.model_bytes()
       << ",\"policy\":\"" << policy << "\""
       << ",\"max_batch\":" << config.max_batch_sequences
       << ",\"max_prefill_seqs\":" << config.max_prefill_sequences_per_iteration
       << ",\"max_batch_tokens\":" << config.max_batch_tokens
       << ",\"kv_blocks\":" << config.kv_blocks
       << ",\"prefix_cache\":" << (config.prefix_caching ? "true" : "false")
       << ",\"backend_sampling\":" << (llama.backend_sampling ? "true" : "false")
       << ",\"qps_offered\":" << Arg(argc, argv, "--qps", "0")
       << ",\"requests\":" << count << ",\"failed\":" << failed
       << ",\"duration_s\":" << fixed(duration)
       << ",\"achieved_rps\":" << fixed(static_cast<double>(count - failed) / duration)
       << ",\"output_tokens\":" << output_tokens << ",\"prompt_tokens\":" << prompt_tokens
       << ",\"output_tok_s\":" << fixed(static_cast<double>(output_tokens) / duration, 1)
       << ",\"total_tok_s\":"
       << fixed(static_cast<double>(output_tokens + prompt_tokens) / duration, 1)
       << ",\"ttft_p50_ms\":" << fixed(Percentile(ttft, 0.5), 1)
       << ",\"ttft_p90_ms\":" << fixed(Percentile(ttft, 0.9), 1)
       << ",\"ttft_p99_ms\":" << fixed(Percentile(ttft, 0.99), 1)
       << ",\"tpot_p50_ms\":" << fixed(Percentile(tpot, 0.5), 2)
       << ",\"tpot_p99_ms\":" << fixed(Percentile(tpot, 0.99), 2)
       << ",\"e2e_p50_ms\":" << fixed(Percentile(e2e, 0.5), 1)
       << ",\"e2e_p99_ms\":" << fixed(Percentile(e2e, 0.99), 1)
       << ",\"iterations\":" << iterations
       << ",\"mean_batch_occupancy\":"
       << fixed(iterations == 0 ? 0.0 : static_cast<double>(slots) / static_cast<double>(iterations))
       << ",\"preemptions\":" << stats.preemptions - before.preemptions
       << ",\"prefill_tokens_computed\":"
       << stats.prefill_tokens_computed - before.prefill_tokens_computed
       << ",\"prefill_tokens_reused\":"
       << stats.prefill_tokens_reused - before.prefill_tokens_reused
       << ",\"wasted_decode_tokens\":"
       << stats.wasted_decode_tokens - before.wasted_decode_tokens
       << ",\"busy_s\":" << fixed(busy)
       << ",\"busy_fraction\":" << fixed(duration > 0 ? busy / duration : 0.0)
       << ",\"model_time_s\":"
       << fixed(std::chrono::duration<double>(executor_stats.model_time -
                                              executor_before.model_time)
                    .count())
       << ",\"sample_time_s\":"
       << fixed(std::chrono::duration<double>(executor_stats.sample_time -
                                              executor_before.sample_time)
                    .count())
       << ",\"useful_tok_per_busy_s\":"
       << fixed(busy > 0 ? static_cast<double>(output_tokens) / busy : 0.0, 1)
       << ",\"blocks_parked\":" << executor_stats.blocks_parked - executor_before.blocks_parked
       << ",\"blocks_dropped\":"
       << executor_stats.blocks_dropped - executor_before.blocks_dropped
       << ",\"gpu\":" << JsonString(gpu_summary.name)
       << ",\"gpu_samples\":" << gpu_summary.samples
       << ",\"gpu_util_pct\":" << fixed(gpu_summary.mean_utilization_pct, 1)
       << ",\"gpu_mem_util_pct\":" << fixed(gpu_summary.mean_memory_utilization_pct, 1)
       << ",\"gpu_power_w\":" << fixed(gpu_summary.mean_power_w, 1)
       << ",\"gpu_peak_used_mib\":" << gpu_summary.peak_used_bytes / (1024 * 1024) << "}";
  std::cout << json.str() << std::endl;
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
