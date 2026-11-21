#include "infersched/engine/llama_executor.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <llama.h>

#include "infersched/engine/deterministic_engine.hpp"

namespace infersched::engine {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t Fnv1a(const std::string_view value) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const char character : value) {
    hash ^= static_cast<unsigned char>(character);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t Mix(std::uint64_t value) noexcept {  // splitmix64 finalizer
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

void QuietLog(const ggml_log_level level, const char* text, void* user_data) {
  static_cast<void>(user_data);
  if (level == GGML_LOG_LEVEL_ERROR) {
    std::fputs(text, stderr);
  }
}

void InitBackendOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (std::getenv("INFERSCHED_LLAMA_LOG") == nullptr) {
      llama_log_set(QuietLog, nullptr);
    }
    llama_backend_init();
  });
}

llama_batch& AsBatch(void* batch) { return *static_cast<llama_batch*>(batch); }

void AddToken(llama_batch& batch, const llama_token token, const llama_pos position,
              const llama_seq_id sequence, const bool output) {
  const auto index = static_cast<std::size_t>(batch.n_tokens);
  batch.token[index] = token;
  batch.pos[index] = position;
  batch.n_seq_id[index] = 1;
  batch.seq_id[index][0] = sequence;
  batch.logits[index] = static_cast<int8_t>(output);
  ++batch.n_tokens;
}

}  // namespace

LlamaExecutor::LlamaExecutor(LlamaExecutorConfig config) : config_(std::move(config)) {
  if (config_.block_size_tokens == 0 || config_.max_sequences == 0 ||
      config_.kv_tokens == 0 || config_.max_batch_tokens == 0) {
    throw std::invalid_argument("llama executor limits must be positive");
  }
  const std::size_t sequences = config_.max_sequences + config_.holder_sequences;
  if (sequences > 256) {  // LLAMA_MAX_SEQ in llama.cpp b12188
    throw std::invalid_argument("max_sequences + holder_sequences must be <= 256");
  }
  InitBackendOnce();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config_.gpu_layers;
  model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
  if (model_ == nullptr) {
    throw std::runtime_error("failed to load model " + config_.model_path);
  }

  batch_capacity_ = config_.max_batch_tokens + config_.max_sequences;
  llama_context_params context_params = llama_context_default_params();
  // Slack over the scheduler's logical capacity: a request whose whole
  // prompt matched recomputes its last token, and a shared partial tail block
  // is appended to by each sharer, so llama.cpp can need a few cells per
  // running sequence beyond the PagedKvCache count.
  context_params.n_ctx = static_cast<std::uint32_t>(
      config_.kv_tokens + 2 * config_.max_sequences * config_.block_size_tokens);
  context_params.n_batch = static_cast<std::uint32_t>(batch_capacity_);
  context_params.n_ubatch =
      static_cast<std::uint32_t>(std::min(config_.ubatch_tokens, batch_capacity_));
  context_params.n_seq_max = static_cast<std::uint32_t>(sequences);
  // A step needs logits only for each sequence's last token, so size the
  // output buffer by sequences, not batch tokens (151k-vocab logits for a
  // 2k-token batch would be ~1.2 GB). llama.cpp reserves one output per
  // sequence id at start-up, so the bound is n_seq_max.
  context_params.n_outputs_max = static_cast<std::uint32_t>(sequences);
  context_params.kv_unified = true;  // One cell pool, governed by PagedKvCache.
  context_params.flash_attn_type = config_.flash_attention
                                       ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                       : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  context_params.n_threads = config_.threads;
  context_params.n_threads_batch = config_.threads;
  context_params.no_perf = true;
  context_ = llama_init_from_model(model_, context_params);
  if (context_ == nullptr) {
    llama_model_free(model_);
    throw std::runtime_error("failed to create llama context");
  }

  if (config_.backend_sampling) {
    for (std::size_t id = 0; id < config_.max_sequences; ++id) {
      llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
      llama_sampler_chain_add(chain, llama_sampler_init_greedy());
      samplers_.push_back(chain);
      if (!llama_set_sampler(context_, static_cast<llama_seq_id>(id), chain)) {
        throw std::runtime_error("backend sampling is not supported by this model/backend");
      }
    }
  }

  vocab_ = llama_model_get_vocab(model_);
  vocab_size_ = llama_vocab_n_tokens(vocab_);
  // Synthetic prompts draw from the middle of the vocabulary, well away from
  // the special/control tokens that BPE vocabularies keep at the ends.
  token_low_ = vocab_size_ / 4;
  token_high_ = vocab_size_ * 3 / 4;

  batch_ = new llama_batch(llama_batch_init(static_cast<std::int32_t>(batch_capacity_), 0, 1));
  for (std::size_t id = config_.max_sequences; id > 0; --id) {
    free_running_.push_back(static_cast<std::int32_t>(id - 1));
  }
  running_by_id_.assign(config_.max_sequences, nullptr);
  const std::size_t max_blocks =
      (config_.kv_tokens + config_.block_size_tokens - 1) / config_.block_size_tokens;
  holder_slots_.assign(config_.holder_sequences,
                       std::vector<PagedKvCache::BlockId>(max_blocks, kNoBlock));
}

LlamaExecutor::~LlamaExecutor() {
  if (batch_ != nullptr) {
    llama_batch_free(AsBatch(batch_));
    delete static_cast<llama_batch*>(batch_);
  }
  llama_free(context_);
  for (llama_sampler* chain : samplers_) {
    llama_sampler_free(chain);
  }
  llama_model_free(model_);
}

std::vector<std::int32_t> LlamaExecutor::PromptTokens(const EngineRequest& request) const {
  const std::size_t length = std::max<std::size_t>(request.prompt_tokens, 1);
  const auto span = static_cast<std::uint64_t>(token_high_ - token_low_);
  std::vector<std::int32_t> tokens(length);
  const std::uint64_t tail_key = Fnv1a(request.prompt_hash);
  for (std::size_t position = 0; position < length; ++position) {
    const std::size_t block = position / config_.block_size_tokens;
    const std::uint64_t key = block < request.prompt_block_hashes.size()
                                  ? Fnv1a(request.prompt_block_hashes[block])
                                  : tail_key;
    tokens[position] = token_low_ + static_cast<std::int32_t>(
                                        Mix(key ^ Mix(position)) % span);
  }
  return tokens;
}

PrefillResult LlamaExecutor::Prefill(const PrefillWork& work) {
  const EngineRequest& request = *work.request;
  if (sequences_.contains(request.request_id)) {
    throw std::logic_error("sequence already running: " + request.request_id);
  }
  if (free_running_.empty()) {
    throw std::logic_error("no free llama sequence (max_sequences too small)");
  }
  Sequence sequence;
  sequence.id = free_running_.back();
  free_running_.pop_back();
  sequence.prompt = PromptTokens(request);

  llama_memory_t memory = llama_get_memory(context_);
  const std::size_t block_size = config_.block_size_tokens;
  const std::size_t prompt_length = sequence.prompt.size();
  // The last prompt token is always recomputed: its logits pick token 1.
  const std::size_t reusable = std::min(work.cached_tokens, prompt_length - 1);
  std::size_t reused = 0;
  for (std::size_t index = 0; index < work.matched_blocks && reused < reusable; ++index) {
    const auto found = blocks_.find(work.blocks[index]);
    if (found == blocks_.end() || found->second.source < 0 ||
        found->second.index != index) {
      break;  // Dropped (or never materialized): recompute from here.
    }
    BlockKv& kv = found->second;
    if (!Computed(kv.source)) {
      // Admitted earlier in this same iteration: its prompt KV does not exist
      // until the fused llama_decode runs, so it cannot be shared yet.
      ++stats_.reuse_not_ready;
      break;
    }
    const std::size_t begin = index * block_size;
    const std::size_t end = std::min(begin + kv.valid, reusable);
    if (end <= begin) {
      break;
    }
    llama_memory_seq_cp(memory, kv.source, sequence.id, static_cast<llama_pos>(begin),
                        static_cast<llama_pos>(end));
    reused = end;
    if (end < begin + kv.valid) {
      break;  // Partial copy: this sequence computes the rest of the block.
    }
    kv.copies.push_back(sequence.id);
    if (kv.valid < block_size) {
      break;
    }
  }
  stats_.prompt_tokens_copied += reused;
  sequence.compute_from = reused;
  sequence.prompt_pending = true;

  // This sequence materializes every prompt block nobody else holds.
  const std::size_t prompt_blocks = std::min(
      work.blocks.size(), (prompt_length + block_size - 1) / block_size);
  sequence.prompt_blocks.assign(work.blocks.begin(),
                                work.blocks.begin() + static_cast<std::ptrdiff_t>(prompt_blocks));
  for (std::size_t index = 0; index < prompt_blocks; ++index) {
    auto [entry, inserted] = blocks_.try_emplace(work.blocks[index]);
    if (inserted || entry->second.source < 0) {
      entry->second = BlockKv{
          .source = sequence.id,
          .index = static_cast<std::uint32_t>(index),
          .valid = static_cast<std::uint32_t>(
              std::min(block_size, prompt_length - index * block_size)),
          .copies = {}};
    }
  }
  const auto [inserted, ok] = sequences_.emplace(request.request_id, std::move(sequence));
  static_cast<void>(ok);
  running_by_id_[static_cast<std::size_t>(inserted->second.id)] = &inserted->second;
  return PrefillResult{.compute = std::chrono::nanoseconds::zero(), .reused_tokens = reused};
}

std::chrono::nanoseconds LlamaExecutor::Decode(
    const std::size_t batch_size, const std::span<const std::string_view> sequences) {
  static_cast<void>(batch_size);
  const auto start = Clock::now();
  llama_batch& batch = AsBatch(batch_);
  batch.n_tokens = 0;
  outputs_.clear();
  LlamaStepRecord step;
  for (const std::string_view id : sequences) {
    const auto found = sequences_.find(id);
    if (found == sequences_.end()) {
      throw std::logic_error("decode for unknown sequence " + std::string(id));
    }
    Sequence& sequence = found->second;
    const std::size_t prompt_length = sequence.prompt.size();
    const std::size_t needed =
        sequence.prompt_pending ? prompt_length - sequence.compute_from : 1;
    if (static_cast<std::size_t>(batch.n_tokens) + needed > batch_capacity_) {
      throw std::logic_error("scheduled batch exceeds llama batch capacity");
    }
    if (sequence.prompt_pending) {
      for (std::size_t position = sequence.compute_from; position < prompt_length;
           ++position) {
        AddToken(batch, sequence.prompt[position], static_cast<llama_pos>(position),
                 sequence.id, position + 1 == prompt_length);
      }
      stats_.prompt_tokens_computed += needed;
      step.prompt_tokens += needed;
      step.context_tokens += prompt_length;
      sequence.prompt_pending = false;
    } else {
      ++step.decode_tokens;
      step.context_tokens += prompt_length + sequence.generated.size();
      AddToken(batch, sequence.generated.back(),
               static_cast<llama_pos>(prompt_length + sequence.generated.size() - 1),
               sequence.id, true);
    }
    outputs_.emplace_back(&sequence, batch.n_tokens - 1);
  }
  if (batch.n_tokens == 0) {
    return std::chrono::nanoseconds::zero();
  }
  stats_.batch_tokens += static_cast<std::size_t>(batch.n_tokens);
  ++stats_.decode_calls;
  if (const int status = llama_decode(context_, batch); status != 0) {
    throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
  }
  llama_synchronize(context_);
  const auto model_done = Clock::now();
  for (auto& [sequence, index] : outputs_) {
    if (config_.backend_sampling) {
      const llama_token token = llama_get_sampled_token_ith(context_, index);
      if (token != LLAMA_TOKEN_NULL) {
        sequence->generated.push_back(token);
        continue;
      }
    }
    const float* logits = llama_get_logits_ith(context_, index);
    const float* best = std::max_element(logits, logits + vocab_size_);
    sequence->generated.push_back(static_cast<std::int32_t>(best - logits));
  }
  const auto done = Clock::now();
  stats_.model_time += model_done - start;
  stats_.sample_time += done - model_done;
  step.duration = done - start;
  last_step_ = step;
  return done - start;
}

bool LlamaExecutor::Computed(const std::int32_t sequence) const noexcept {
  if (sequence < 0) {
    return false;
  }
  if (IsHolder(sequence)) {
    return true;
  }
  const Sequence* running = running_by_id_[static_cast<std::size_t>(sequence)];
  return running != nullptr && !running->prompt_pending;
}

std::int32_t LlamaExecutor::FindHolder(const std::uint32_t index) const noexcept {
  for (std::size_t holder = 0; holder < holder_slots_.size(); ++holder) {
    if (holder_slots_[holder][index] == kNoBlock) {
      return static_cast<std::int32_t>(config_.max_sequences + holder);
    }
  }
  return -1;
}

void LlamaExecutor::Release(const std::string& request_id,
                            const std::span<const PagedKvCache::BlockId> blocks,
                            const PagedKvCache& kv) {
  static_cast<void>(blocks);
  const auto found = sequences_.find(request_id);
  if (found == sequences_.end()) {
    return;
  }
  Sequence& sequence = found->second;
  llama_memory_t memory = llama_get_memory(context_);
  const std::size_t block_size = config_.block_size_tokens;
  for (const PagedKvCache::BlockId block : sequence.prompt_blocks) {
    const auto entry = blocks_.find(block);
    if (entry == blocks_.end()) {
      continue;  // Freed already (OnBlockFreed) or never tracked.
    }
    BlockKv& state = entry->second;
    std::erase(state.copies, sequence.id);
    if (state.source != sequence.id) {
      continue;
    }
    if (kv.RefCount(block) == 0) {
      blocks_.erase(entry);
      continue;
    }
    if (!state.copies.empty()) {
      state.source = state.copies.back();  // Another running sequence shares it.
      state.copies.pop_back();
      continue;
    }
    if (sequence.prompt_pending) {
      state.source = -1;  // Released before its prompt was computed: no KV.
      continue;
    }
    // Only the prefix cache (or a partial sharer) still wants this block:
    // park its cells in a holder sequence so they survive this release.
    const std::int32_t holder = FindHolder(state.index);
    if (holder < 0) {
      state.source = -1;
      ++stats_.blocks_dropped;
      continue;
    }
    const std::size_t begin = state.index * block_size;
    llama_memory_seq_cp(memory, sequence.id, holder, static_cast<llama_pos>(begin),
                        static_cast<llama_pos>(begin + state.valid));
    holder_slots_[static_cast<std::size_t>(holder) - config_.max_sequences][state.index] = block;
    state.source = holder;
    ++stats_.blocks_parked;
  }
  llama_memory_seq_rm(memory, sequence.id, -1, -1);
  running_by_id_[static_cast<std::size_t>(sequence.id)] = nullptr;
  free_running_.push_back(sequence.id);
  sequences_.erase(found);
}

void LlamaExecutor::OnBlockFreed(const PagedKvCache::BlockId block) noexcept {
  const auto entry = blocks_.find(block);
  if (entry == blocks_.end()) {
    return;
  }
  const BlockKv state = std::move(entry->second);
  blocks_.erase(entry);
  if (state.source >= 0 && IsHolder(state.source)) {
    const std::size_t begin = state.index * config_.block_size_tokens;
    static_cast<void>(llama_memory_seq_rm(llama_get_memory(context_), state.source,
                                          static_cast<llama_pos>(begin),
                                          static_cast<llama_pos>(begin + state.valid)));
    holder_slots_[static_cast<std::size_t>(state.source) - config_.max_sequences]
                 [state.index] = kNoBlock;
  }
  // A running source drops its cells when it is released.
}

std::string LlamaExecutor::Output(const EngineRequest& request) {
  const auto found = sequences_.find(request.request_id);
  if (found == sequences_.end()) {
    return {};
  }
  const auto& generated = found->second.generated;
  if (completion_observer_) {
    completion_observer_(request.request_id, generated);
  }
  std::string text;
  std::vector<char> piece(64);
  for (const std::int32_t token : generated) {
    int32_t length = llama_token_to_piece(vocab_, token, piece.data(),
                                          static_cast<int32_t>(piece.size()), 0, false);
    if (length < 0) {
      piece.resize(static_cast<std::size_t>(-length));
      length = llama_token_to_piece(vocab_, token, piece.data(),
                                    static_cast<int32_t>(piece.size()), 0, false);
    }
    if (length > 0) {
      text.append(piece.data(), static_cast<std::size_t>(length));
    }
  }
  return text;
}

std::size_t LlamaExecutor::parked_blocks() const noexcept {
  std::size_t parked = 0;
  for (const auto& slots : holder_slots_) {
    parked += static_cast<std::size_t>(std::ranges::count_if(
        slots, [](const PagedKvCache::BlockId block) { return block != kNoBlock; }));
  }
  return parked;
}

bool LlamaExecutor::KvEmpty() const {
  llama_memory_t memory = llama_get_memory(context_);
  const std::size_t sequences = config_.max_sequences + config_.holder_sequences;
  for (std::size_t id = 0; id < sequences; ++id) {
    if (llama_memory_seq_pos_max(memory, static_cast<llama_seq_id>(id)) >= 0) {
      return false;
    }
  }
  return true;
}

std::string LlamaExecutor::model_description() const {
  char buffer[256];
  const int32_t length = llama_model_desc(model_, buffer, sizeof(buffer));
  return length > 0 ? std::string(buffer, static_cast<std::size_t>(length)) : std::string{};
}

std::uint64_t LlamaExecutor::model_bytes() const { return llama_model_size(model_); }
std::uint64_t LlamaExecutor::model_parameters() const { return llama_model_n_params(model_); }

}  // namespace infersched::engine
