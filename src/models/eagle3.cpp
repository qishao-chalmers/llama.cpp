#include "models.h"

// eagle3 Encoder: processes target model features through the feature fusion layer
// Input: target_features e.g. [12288, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [4096, n_tokens] stored in t_h_nextn (read via llama_get_embeddings_nextn(ctx_dft))
llm_build_eagle3_enc::llm_build_eagle3_enc(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;

    // Input: target model features, provided via ubatch->embd
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [4096, n_tokens]
    // store in t_h_nextn (same as MTP) so it can be read via llama_get_embeddings_nextn(ctx_dft)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// eagle3 Decoder: processes draft tokens using g_embeddings from the encoder
// Input: draft tokens + g_embeddings from encoder
// Output: draft logits
llm_build_eagle3_dec::llm_build_eagle3_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_layer == 1); // eagle3 has only one decoder layer

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // eagle3 Decoder receives:
    // 1. Token embeddings (from eagle3's own tok_embd if present, else the target model's)
    // 2. g_embeddings from the encoder
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "EAGLE3 decoder requires token embeddings (own or from target model)");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * inp_embd = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inp_embd, "inp_embd", -1);

    ggml_tensor * inp_g = inp->embd;
    cb(inp_g, "inp_g_embeddings", -1);

    res->add_input(std::move(inp));

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        // Apply input_layernorm to the token embeddings
        ggml_tensor * embd_norm = build_norm(inp_embd,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // Apply hidden_norm to inp_g
        ggml_tensor * g_norm = build_norm(inp_g,
                model.layers[il].attn_norm_2, NULL,
                LLM_NORM_RMS, -1);
        cb(g_norm, "g_norm", il);

        // norm_before_residual: what goes into the residual connection (compatible with RedHatAI eagle3 speculators)
        // - false (default): use raw inp_g for residual
        // - true: use normalized g_norm for residual
        ggml_tensor * inpSA = hparams.norm_before_residual ? g_norm : inpL;

        // Concatenate normalized inp_embd and normalized inp_g
        cur = ggml_concat(ctx0, embd_norm, g_norm, il);
        cb(cur, "concat_embd", il);

        // Self-attention with concatenated input
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }

    cur = inpL;

    // Output prenorm state (for next token's g_embeddings in autoregressive generation)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head - projects to draft vocabulary
    // if the draft has no own output projection, inherit the target model's lm_head
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->output != nullptr && "EAGLE3 decoder requires an output projection (own or from target model)");
        output = model_other->output;
    }
    cur = build_lora_mm(output, cur);

    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
