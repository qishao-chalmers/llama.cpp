# KV Cache Quantization Research — Index

Perplexity/accuracy impact of KV cache compression on Qwen3-8B / Qwen2.5-Coder-7B / Llama-4-Scout-17B.
Code: `research/scripts/` — Python ctypes wrapping `build_release/bin/libllama.so`.

## Sub-files (read on demand)

| File | Contents |
|------|----------|
| `context/cursor_handoff.md` | **Cursor agents: start here.** Full project state, script inventory, what's done/pending, Claude Code memory locations |
| `context/adaptive_kv_design.md` | **Adaptive KV cache design.** Base+delta split storage, adaptive decode scheme, comparison with QuantSpec/TurboQuant/KIVI, performance model, next steps |
| `context/tasks.md` | Task guide: wikitext PPL / GSM8K accuracy / NIAH / code, full commands |
| `context/left_tasks.md` | Deferred ideas; glossary: **bootstrap window** vs **quant rollout window** |
| `context/combined_weight_kv_quant.md` | **Combined weight + KV quant.** Draft model (Q4_K_M) + int2 KV vs verifier (Q8_0) + fp16; base+delta weight storage idea; `--draft-model` implementation plan |
| `context/split_weight_precision.md` | **Split-weight precision.** In-place Q8_0 draft/verify: 2-region (Q4/Q8) and 3-region (Q2/Q4/Q8) nibble split, CUDA kernels, bootstrap verify batch pass, adaptive gen integration, further work |
| `context/base_delta_weight_split.md` | **Base+delta weight split (new).** Q8_0-Q4_K_M as draft, int4 delta recovers ~Q8_0 verify quality; `block_q4km_d4` format, CUDA kernel plan, offline delta computation, 5-phase impl plan |
| `context/analysis.md` | Analysis tools: plot_entropy.py, analyze_alarms.py, plot_niah.py |
| `context/scripts.md` | Core infrastructure: run_sweep.py CLI, adaptive sim/gen (top‑k/p, zones), quant types |
| `context/adaptive_gen_bootstrap.md` | **Adaptive-gen bootstrap:** steer-back, metrics, agree-rate vs **cost** pick, probe |
| `context/bootstrap_ms_quant_example.json` | Example cost JSON: **`recover`** (verifier-quant → ms) + draft quants |
| `context/strategies.md` | Quantization strategies, group-size semantics, three-zone KIVI |
| `context/api.md` | ctypes bindings, struct layouts, state blob format, critical API pitfalls |
| `context/results.md` | All experiment results tables and key findings |
| `context/setup.md` | New machine setup (build, deps, model download) |
| `context/gpu_quantization.md` | GPU-side KV quantization implementation plan |
| `context/gguf_quant_sizes.md` | **GGUF quant size analysis.** Per-tensor type assignments for Q3_K_M/Q4_K_M (src/llama-quant.cpp), effective bpw, Qwen3-8B architecture param counts, why Q3_K_M≈4bpw |
| `context/qwen3_kernel_trace_mapping.md` | **NVBit kernel → layer mapping.** Prefill vs decode repetitive kernels (Qwen3-8B, `-fa 1`), phase markers, launch counts, quant-type GEMV/MMQ notes |
| `context/roofline_calibration.md` | **Roofline vs measured gap analysis.** T_floor=2.2ms (CPU-GPU pipeline bubble), BW efficiency 33–50% of peak, two-parameter model, KV quant overhead, why efficiency does not grow with B |
| `context/selective_weight_precision.md` | **Selective weight precision builder.** N-bit-in-Q8 GGUF: re-quantise attn/ffn/essential tensor groups to configurable bits (2–8), optional first/last layer protection, asymmetric quant via `_uniform_quant_asym`, output stays Q8_0 — no new kernels. Script: `build_selective_precision.py` |
| `context/kv_refresh_teacher.md` | **KV cache refresh from teacher model.** Small model always generates; teacher teacher-forces every W tokens and overwrites small's KV. Isolates KV accumulation error (B) from weight imprecision (A). Cost model, W=1 interpretation, comparison with adaptive gen scheme. Script: `vllm_env/kv_refresh_decode.py` |

## Critical Pitfall (always remember)
`llama_memory_clear` takes `llama_memory_t`, NOT `llama_context*`:
```python
mem = lib.llama_get_memory(ctx)    # required first step
lib.llama_memory_clear(mem, True)  # WRONG: passing ctx directly → silent corruption → segfault
```

## Script Map

```
Datasets                   Evaluation              Analysis
────────────────           ─────────────────       ──────────────────────────
fetch_wikitext.py    ─►    run_sweep.py        ─►  plot_entropy.py   (PPL/diags)
fetch_gsm8k.py       ─►    + --eval-accuracy   ─►  analyze_alarms.py (TPR/FPR)
build_niah_dataset.py─►    + --skip-ppl            plot_niah.py      (U-curve)
fetch_code.py        ─►    + --save-per-example     plot_results.py   (bar chart)
                           + --save-diags
```

## Datasets (research/data/)

| File | Tokens/Examples | Task | Build command |
|------|----------------|------|---------------|
| `wikitext2_test.txt` | 245K tokens | flat PPL | `fetch_wikitext.py` |
| `c4_val.txt` | 500K tokens | flat PPL / NIAH haystack | `fetch_wikitext.py` |
| `gsm8k_test.jsonl` | 1319 examples | accuracy (8-shot math) | `fetch_gsm8k.py` |
| `niah_4096.jsonl` | 110 examples (11pos×10) | accuracy / U-curve | `build_niah_dataset.py` |
| `code_longcode.jsonl` | 143 examples | structured PPL | `fetch_code.py` |

## Key Results (prefill=1024, score_windows=512/1024/2048, Qwen3-8B, 3 chunks)
| Quant | ppl@512 | ppl@1024 | ppl@2048 | time |
|-------|---------|----------|----------|------|
| fp16 | 5.50 | 6.97 | 8.06 | 48s |
| int8_ch | 5.49 | 6.97 | 8.07 | 83s |
| int4_ch:int4_tok | 5.53 | 7.07 | 8.15 | 84s |
| int3_ch | 5.90 | 7.41 | 8.75 | 85s |
| int2_ch | 49.1 | 47.5 | 45.4 | 83s |
