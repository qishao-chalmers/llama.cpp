# KV Cache Quantization Research — Index

Perplexity/accuracy impact of KV cache compression on Qwen3-8B / Qwen2.5-Coder-7B / Llama-4-Scout-17B.
Code: `research/scripts/` — Python ctypes wrapping `build_release/bin/libllama.so`.

## Sub-files (read on demand)

| File | Contents |
|------|----------|
| `context/tasks.md` | **Start here** — task guide: wikitext PPL / GSM8K accuracy / NIAH / code, full commands |
| `context/analysis.md` | Analysis tools: plot_entropy.py, analyze_alarms.py, plot_niah.py |
| `context/scripts.md` | Core infrastructure: run_sweep.py CLI flags, module architecture, quant types |
| `context/strategies.md` | Quantization strategies, group-size semantics, three-zone KIVI |
| `context/api.md` | ctypes bindings, struct layouts, state blob format, critical API pitfalls |
| `context/results.md` | All experiment results tables and key findings |
| `context/setup.md` | New machine setup (build, deps, model download) |
| `context/gpu_quantization.md` | GPU-side KV quantization implementation plan |

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
