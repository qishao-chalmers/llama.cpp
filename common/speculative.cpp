#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_layer_inp / llama_get_embeddings_nextn / llama_model_target_layer_ids

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

// EAGLE3 speculative decoding state
//
// Decoder input convention: pairs (token[P+1], g_embd[P]) at memory pos P, where g_embd[P] is the
// draft's fused projection ("fc") of the target's captured hidden states (at target_layer_ids) for
// the token at position P. This matches the training convention used by real EAGLE3 checkpoints.
//
// Simplification vs. a fully-integrated implementation: instead of a process()-style hook that
// captures the target's layer features for every row of every ctx_tgt decode (prefill and verify
// batches alike), this state captures them lazily - each draft() call reads whatever the target's
// most recent decode captured (llama_get_embeddings_layer_inp only ever reflects that one call, see
// prev_prompt_len below), and begin() does one extra re-decode of the prompt to backfill the prefill
// features it otherwise couldn't have captured (its constructor runs after the caller's own prefill
// decode - see the priming comment below). This costs a handful of extra small forward passes
// through the draft's encoder, negligible next to the draft's own decode cost.
//
// Root cause of a long-standing near-zero-signal bug (fixed alongside this state's own logic, in
// llama_context::encode() in src/llama-context.cpp): llama_encode() never populated embd_nextn at
// all - only decode() did - so every llama_get_embeddings_nextn() call made right after an encode()
// call in this file (the seed/refresh/priming steps below) was silently reading stale leftover data
// from whatever decode() call last ran on ctx_dft, not the actual encoder output just computed.
struct common_speculative_state_eagle3 : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch batch;

    int32_t n_embd_tgt = 0; // target model hidden size
    int32_t n_embd_dec = 0; // draft model hidden size
    int32_t n_embd_enc = 0; // target_layer_ids_n * n_embd_tgt

    const int32_t * target_layer_ids   = nullptr;
    uint32_t        target_layer_ids_n = 0;

    std::vector<float> feat_buf; // scratch: concatenated target features for the seed token
    std::vector<float> g;        // current g_embd (fused feature state), chained across draft steps

    // size of prompt_tgt as observed at the end of the previous draft() call. llama_get_embeddings_layer_inp
    // only ever reflects the target's *most recent* llama_decode() call (the verify batch
    // [id_last_prev, draft_prev...]), never a persistent history. Since the caller (e.g.
    // speculative-simple.cpp) pushes exactly one token per row of that batch it keeps onto
    // prompt_tgt before calling draft() again, prompt_tgt's growth since last call equals that
    // batch's row count, and the row we need (the last accepted row, whose features seed the
    // next draft) is (growth - 1). This is a pure position computation - deliberately not a
    // search for a matching token value, which breaks on repeated tokens (common in real text)
    // and desyncs silently if the caller ever discards/truncates what draft() returned (e.g. the
    // n_min check in speculative-simple.cpp).
    // < 0 means "no verify batch captured yet for this state to use" (first call after begin()).
    int64_t prev_prompt_len = -1;

    // ctx_dft's KV cache is NOT reset between rounds - it must stay continuous, mirroring how
    // ctx_tgt's own cache grows across the whole generation, because the (single-layer) decoder's
    // self-attention relies on that accumulated history for context; without it, the decoder only
    // ever sees the handful of tokens drafted in the current round and predicts far worse (verified
    // empirically: ~4% acceptance with a per-round reset vs 56% with a continuous cache, on the
    // same GGUF files and prompt, cross-checked against the upstream reference implementation).
    // dft_seed_pos is the absolute position where this round's seed step (id_last, g) belongs -
    // i.e. the position that held the last *accepted* draft step from the previous round's chain
    // (or 0 before any round has run). Everything at or beyond it is this state's own leftover
    // speculation from last round and gets trimmed before writing the new seed, rather than wiping
    // the whole cache.
    llama_pos dft_seed_pos = 0;

    // Deferred boundary primed by begin() from the prompt: the caller's own prompt prefill runs
    // before this state's constructor turns on target feature capture, so those features are
    // lost by the time we could read them - begin() recovers them by re-decoding the prompt
    // through ctx_tgt (now with capture on) and priming ctx_dft's KV cache from it, exactly like
    // the reference's process()-hook does on every ctx_tgt decode including the prefill. This
    // gives the very first draft() call real accumulated context to attend to instead of a
    // completely empty cache (which measurably starves the first several rounds - see docs).
    // pending_g/pending_pos hold the one deferred (g, pos) pair priming couldn't complete (its
    // token isn't known until the caller's first draft() call supplies id_last); pending_pos < 0
    // means "no primed boundary to use" (priming skipped/failed, or already consumed).
    std::vector<float> pending_g;
    llama_pos           pending_pos = -1;

    common_speculative_state_eagle3(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_dft = llama_get_model(ctx_dft);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);

        if (target_layer_ids_n != 3) {
            throw std::runtime_error("eagle3: draft model must declare exactly 3 target_layer_ids, got " +
                    std::to_string(target_layer_ids_n));
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;

        // llama_batch_init allocates only one of token/embd; the decoder step needs both.
        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/* n_tokens_alloc = */ n_b, /* embd = */ n_embd_dec, /* n_seq_max = */ 1);
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        common_params_sampling sparams;
        sparams.no_perf  = false;
        sparams.top_k    = 10;
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

        smpl = common_sampler_init(model_dft, sparams);

        const int32_t n_layer_tgt = llama_model_n_layer(model_tgt);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] >= n_layer_tgt) {
                throw std::runtime_error("eagle3: target_layer_ids[" + std::to_string(k) + "] = " +
                        std::to_string(target_layer_ids[k]) + " exceeds target n_layer = " + std::to_string(n_layer_tgt));
            }
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, /* masked = */ true);
    }

    ~common_speculative_state_eagle3() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        // new session: nothing accepted yet, ctx_dft's cache (if any, from a previous session
        // reusing this state) is stale in full.
        prev_prompt_len = -1;
        dft_seed_pos    = 0;
        pending_g.clear();
        pending_pos = -1;
        llama_memory_clear(llama_get_memory(ctx_dft), true);

        const int32_t N = (int32_t) prompt.size();
        if (N < 2) {
            return; // nothing to prime - need at least one (t[k+1], g[k]) pair
        }

        // Re-decode the prompt through ctx_tgt (capture is already on - the constructor enabled
        // it before this call) to recover target features for it. llama_decode requires new
        // positions to start exactly at (existing seq max + 1) - it has no in-place-overwrite
        // path - so the caller's own prefill entries at these same positions must be removed
        // first; the re-decode then deterministically reconstructs identical KV content at
        // positions 0..N-1, so this is a pure re-computation for the side-channel feature output,
        // not a lasting mutation of ctx_tgt's real cache content or the caller's own n_past
        // bookkeeping.
        {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, N);

            llama_batch prime_batch = llama_batch_init(N, 0, 1);
            for (int32_t i = 0; i < N; ++i) {
                common_batch_add(prime_batch, prompt[i], i, { 0 }, false);
            }
            const int rc = llama_decode(ctx_tgt, prime_batch);
            llama_batch_free(prime_batch);
            if (rc != 0) {
                LOG_ERR("%s: eagle3 prompt re-decode for priming failed (rc=%d)\n", __func__, rc);
                return;
            }
        }

        std::vector<float> feat_all((size_t) N * n_embd_enc);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                LOG_ERR("%s: eagle3 target layer %d input not extracted during priming\n", __func__, target_layer_ids[k]);
                return;
            }
            for (int32_t i = 0; i < N; ++i) {
                std::memcpy(feat_all.data() + (size_t) i * n_embd_enc + (size_t) k * n_embd_tgt,
                        layer + (size_t) i * n_embd_tgt,
                        (size_t) n_embd_tgt * sizeof(float));
            }
        }

        std::vector<float> g_all((size_t) N * n_embd_dec);
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < N; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, N - i);
            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ feat_all.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            if (llama_encode(ctx_dft, enc_batch) != 0) {
                LOG_ERR("%s: eagle3 priming encode failed\n", __func__);
                return;
            }
            const float * g_chunk = llama_get_embeddings_nextn(ctx_dft);
            if (!g_chunk) {
                LOG_ERR("%s: eagle3 priming encoder produced no output\n", __func__);
                return;
            }
            std::memcpy(g_all.data() + (size_t) i * n_embd_dec, g_chunk, (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        // Write decoder KV for rows 0..N-2: (token=prompt[k+1], g=g_all[k]) at position k - the
        // same (t[P+1], g_P) convention draft()'s seed/refresh steps use. Row N-1's pair is
        // deferred: its token (whatever follows the prompt) isn't known here, only once the
        // caller's first draft() call supplies id_last.
        if (N > 1) {
            batch.n_tokens = N - 1;
            for (int32_t k = 0; k < N - 1; ++k) {
                batch.token[k]     = prompt[k + 1];
                batch.pos[k]       = k;
                batch.n_seq_id[k]  = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k]    = false;
                std::memcpy(batch.embd + (size_t) k * n_embd_dec,
                        g_all.data() + (size_t) k * n_embd_dec,
                        (size_t) n_embd_dec * sizeof(float));
            }
            if (llama_decode(ctx_dft, batch) != 0) {
                LOG_ERR("%s: eagle3 priming decode failed\n", __func__);
                return;
            }
        }

        pending_g.assign(g_all.end() - n_embd_dec, g_all.end());
        pending_pos = N - 1;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        result.clear();

        if (prompt_tgt.empty()) {
            prev_prompt_len = -1;
            dft_seed_pos    = 0;
            pending_pos     = -1;
            llama_memory_clear(llama_get_memory(ctx_dft), true);
            return;
        }

        if (pending_pos >= 0) {
            // First call after begin(): use the deferred boundary primed by begin()'s prompt
            // re-decode instead of the normal growth-based row lookup. id_last (the caller's
            // param here) is exactly the token begin() didn't know yet; pending_g is already the
            // encoder's output (begin() ran the encoder itself while priming), so we skip
            // feat_buf/llama_encode entirely and use it directly as g.
            dft_seed_pos = pending_pos;
            g = pending_g;
            pending_pos = -1;
            prev_prompt_len = (int64_t) prompt_tgt.size();
        } else {
            if (prev_prompt_len < 0) {
                // Priming didn't run or failed (e.g. N < 2): id_last hasn't been fed to ctx_tgt
                // yet, so its features don't exist yet. Just record where we are so next call can
                // measure growth from here.
                prev_prompt_len = (int64_t) prompt_tgt.size();
                return;
            }

            const int64_t growth = (int64_t) prompt_tgt.size() - prev_prompt_len;
            prev_prompt_len = (int64_t) prompt_tgt.size();

            if (growth <= 0) {
                // Should not happen (prompt_tgt only grows), but nothing sane to seed from.
                return;
            }

            // row, within the target's most recent verify batch [id_last_prev, draft_prev...], of
            // the last accepted token - equivalently, how many of last round's drafted tokens got
            // accepted.
            const int32_t   row            = (int32_t) (growth - 1);
            const int32_t   n_accepted     = row;
            const int64_t   prompt_len_old = prev_prompt_len - growth; // == prev_prompt_len before the update above
            const llama_pos pos_base       = dft_seed_pos; // position of target-batch row 0 (id_last) this round

            feat_buf.resize(n_embd_enc);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                if (!layer) {
                    LOG_ERR("%s: eagle3 target layer %d input not extracted\n", __func__, target_layer_ids[k]);
                    return;
                }
                std::memcpy(feat_buf.data() + (size_t) k * n_embd_tgt,
                        layer + (size_t) row * n_embd_tgt,
                        (size_t) n_embd_tgt * sizeof(float));
            }

            // Trim everything this call is about to (re)write, *before* writing any of it. Last
            // round's own draft() call speculatively wrote its own guesses at pos_base+1..
            // pos_base+n_max (chained via the decoder's own output) - llama_decode requires a
            // batch's positions to start exactly at (existing max + 1), with no in-place-overwrite
            // path, so both the refresh below (rewrites pos_base+1..pos_base+n_accepted-1 with
            // real target-grounded features) and the seed step further below (rewrites
            // pos_base+n_accepted, the new dft_seed_pos) would otherwise collide with that
            // leftover speculative content and fail.
            //
            // When n_accepted == 0, the new seed position (pos_base+n_accepted) collapses onto
            // pos_base itself: nothing from last round's draft was accepted, so this round's seed
            // writes a *different* (id_last, g) pair to the exact same slot last round's own seed
            // step used - that slot must be trimmed too, not preserved. The "row 0 already holds a
            // valid entry" reasoning below only holds for the refresh loop's row range
            // (1..n_accepted-1), which is empty whenever n_accepted <= 1 anyway.
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos_base + (n_accepted > 0 ? 1 : 0), -1);

            // Refresh KV entries for target-batch rows 1..n_accepted-1 (n_accepted-1 of them; a
            // no-op for n_accepted <= 1) with real target-grounded features, using known tokens
            // from prompt_tgt - no sampling needed, we already know these were accepted.
            //
            // Row 0 of the target's verify batch (= id_last, the token this state was seeded with
            // last round) is deliberately EXCLUDED: dft_seed_pos already holds the correct entry
            // for it, written by last round's own seed step - (token=id_last, g=features of the
            // token *before* id_last). Refreshing row 0 too means overwriting that entry with a
            // DIFFERENT, mismatched one, which silently corrupts self-attention for everything
            // that attends back to this position afterwards.
            if (n_accepted > 1) {
                const int32_t n_refresh = n_accepted - 1;
                std::vector<float> feat_refresh((size_t) n_refresh * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    for (int32_t r = 0; r < n_refresh; ++r) {
                        // target-batch row (r+1): rows 1..n_accepted-1
                        std::memcpy(feat_refresh.data() + (size_t) r * n_embd_enc + (size_t) k * n_embd_tgt,
                                layer + (size_t) (r + 1) * n_embd_tgt,
                                (size_t) n_embd_tgt * sizeof(float));
                    }
                }
                llama_batch enc_refresh = {
                    /*.n_tokens =*/ n_refresh,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ feat_refresh.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };
                if (llama_encode(ctx_dft, enc_refresh) != 0) {
                    LOG_ERR("%s: eagle3 refresh encode failed\n", __func__);
                    return;
                }
                const float * g_refresh = llama_get_embeddings_nextn(ctx_dft);
                if (!g_refresh) {
                    LOG_ERR("%s: eagle3 refresh encoder produced no output\n", __func__);
                    return;
                }
                batch.n_tokens = n_refresh;
                for (int32_t r = 0; r < n_refresh; ++r) {
                    // position pos_base + (r+1): pairs with target-batch row (r+1)'s features,
                    // predicting target-batch row (r+2)'s token - i.e. prompt_tgt[prompt_len_old+r+2].
                    batch.token[r]     = prompt_tgt[(size_t) (prompt_len_old + r + 2)];
                    batch.pos[r]       = pos_base + r + 1;
                    batch.n_seq_id[r]  = 1;
                    batch.seq_id[r][0] = 0;
                    batch.logits[r]    = false;
                    std::memcpy(batch.embd + (size_t) r * n_embd_dec,
                            g_refresh + (size_t) r * n_embd_dec,
                            (size_t) n_embd_dec * sizeof(float));
                }
                if (llama_decode(ctx_dft, batch) != 0) {
                    LOG_ERR("%s: eagle3 refresh decode failed\n", __func__);
                    return;
                }
            }

            // The new seed position is pos_base advanced by however many of last round's drafted
            // steps were actually accepted (0 if none). Everything beyond it was already trimmed
            // above (along with the refresh range) in the same pos_base+1 cut.
            dft_seed_pos = pos_base + n_accepted;

            llama_batch enc_batch = {
                /*.n_tokens =*/ 1,
                /*.token    =*/ nullptr,
                /*.embd     =*/ feat_buf.data(),
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };

            if (llama_encode(ctx_dft, enc_batch) != 0) {
                LOG_ERR("%s: eagle3 llama_encode(ctx_dft) failed\n", __func__);
                return;
            }

            const float * g0 = llama_get_embeddings_nextn(ctx_dft);
            if (!g0) {
                LOG_ERR("%s: eagle3 encoder produced no output\n", __func__);
                return;
            }
            g.assign(g0, g0 + n_embd_dec);
        }

        common_sampler_reset(smpl);

        llama_token cur_token = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            batch.n_tokens      = 1;
            batch.token[0]      = cur_token;
            batch.pos[0]        = dft_seed_pos + i;
            batch.n_seq_id[0]   = 1;
            batch.seq_id[0][0]  = 0;
            batch.logits[0]     = true;
            std::memcpy(batch.embd, g.data(), (size_t) n_embd_dec * sizeof(float));

            if (llama_decode(ctx_dft, batch) != 0) {
                LOG_ERR("%s: eagle3 llama_decode(ctx_dft) failed\n", __func__);
                break;
            }

            common_sampler_sample(smpl, ctx_dft, 0, true);
            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            const float * g_next = llama_get_embeddings_nextn_ith(ctx_dft, 0);
            if (!g_next) {
                break;
            }
            std::copy(g_next, g_next + n_embd_dec, g.begin());

            cur_token = id;
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop: draft() re-derives everything it needs from (prompt_tgt, id_last) each round
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    auto * mem = llama_get_memory(ctx_tgt);
    if (mem == nullptr) {
        return false;
    }

    bool res = true;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft_eagle3 = ctx_dft != nullptr && llama_model_target_layer_ids_n(llama_get_model(ctx_dft)) == 3;
        bool has_draft        = !params.mparams_dft.path.empty() && !has_draft_eagle3;

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type,
                    /* .ctx_tgt = */ ctx_tgt,
                    /* .ctx_dft = */ ctx_dft
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
