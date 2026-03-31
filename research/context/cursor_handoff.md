# Cursor Agent Handoff — KV Cache Quantization Research

Last updated: 2026-03-27

## Project Summary

Studying perplexity/accuracy impact of KV cache quantization on LLMs (Qwen3-8B, Qwen3-14B,
Llama-4-Scout-17B, GPT-OSS-20B). All experiment code is Python ctypes wrapping
`build_release/bin/libllama.so`. Branch: `kv-quant-study`.

## How to find prior work

This project was primarily built in a **Claude Code** session. Cursor has limited transcript
history. The authoritative source of prior conversation and decisions is:

- **Claude Code memory**: `~/.claude/projects/-home-qshao-Project-Fun-llama-cpp/memory/`
  - `MEMORY.md` — master index, always-remember rules (API pitfalls, chat format, quant gotchas)
  - `scripts.md` — every script's role, CLI flags, hook call flow, auto-detection logic
  - `results.md` — all experimental results tables, corpus files, pending experiments
  - `api.md` — C API pitfalls (llama_memory_clear, logits_ith, etc.)
  - `quantization.md` — quant types, granularities, KIVI three-zone, asymmetric mode
  - `cluster.md` — MN5 SLURM setup
  - `project_gpu_quantization.md` — GPU-side CuPy quant implementation

- **Claude Code transcript** (51MB, 10K lines):
  `~/.claude/projects/-home-qshao-Project-Fun-llama-cpp/c972303c-932d-46b4-93fc-74b0ea4f42f8.jsonl`
  Search with grep, don't read whole file. Each line is JSON: `{"message":{"role":"user"|"assistant","content":...}}`

- **Cursor transcripts** (limited):
  `~/.cursor/projects/home-qshao-Project-Fun-llama-cpp/agent-transcripts/`
  The earliest Cursor session (`9ee7f33a`, Mar 10) covered llama.cpp architecture overview and
  state_write/state_read API understanding.

## Script Inventory

### Core infrastructure
| Script | Purpose |
|--------|---------|
| `llama_bindings.py` | ctypes bindings, tokenize, log_softmax, chat format detection |
| `strategies.py` | Decode strategies: batch prefill, token-by-token, generate |
| `quant.py` | Quantization functions, dynamic name parsing, BinTracker |
| `parse_state.py` | CPU KV state blob: read → apply fn → write back |
| `gpu_quant.py` | CuPy in-place GPU quant (~400× faster than CPU) |
| `run_sweep.py` | **Main CLI** — orchestrates sweeps, hooks, baselines, accuracy eval |

### Dataset fetchers
| Script | Output |
|--------|--------|
| `fetch_wikitext.py` | `wikitext2_test.txt`, `c4_val.txt` |
| `fetch_gsm8k.py` | `gsm8k_test.jsonl` (8-shot) |
| `fetch_humaneval.py` | `humaneval.jsonl` (164 problems) |
| `fetch_code.py` | `code_longcode.jsonl` (143 examples) |
| `fetch_longbench.py` | LongBench tasks (qasper, etc.) |
| `fetch_aime.py` | `aime_2022_2024.jsonl` (73 problems) |
| `build_niah_dataset.py` | `niah_4096.jsonl` (110 examples, needs model for tokenization) |

### Analysis & plotting
| Script | Purpose |
|--------|---------|
| `plot_entropy.py` | H/lp/p_max vs position from `--save-diags` |
| `analyze_alarms.py` | TPR/FPR for alarm signals (H, p_max, rep_rate) |
| `plot_niah.py` | Accuracy vs needle position U-curve |
| `plot_bins.py` | Bin hit distribution visualization |
| `plot_results.py` | Bar charts from results JSON |

### Performance modeling
| Script | Purpose |
|--------|---------|
| `perf_model.py` | **High-level** analytical model. Bandwidth-bound decode, compute-bound prefill. Supports adaptive quant sim. Has `--sweep` mode for benchmark × batch grid. |
| `roofline_layer.py` | **Per-operation** roofline model. Breaks each transformer layer into ops (QKV, attn, FFN, norms). Separate weight/activation/KV bandwidth streams. Flash attention model. Three stages (prefill/decode/verify). **Validated against Qwen2-7B/A100 measured data — see validation section below.** |

### Batch execution
| Script | Purpose |
|--------|---------|
| `run_all_benchmarks.sh` | Full suite locally (wikitext, gsm8k, longbench, niah, humaneval, aime) |
| `submit_benchmarks.sh` | SLURM submission for MN5 cluster |

## Key Results (Qwen3-8B)

### Wikitext PPL (prefill=1024, score-windows 512/1024/2048)
| Quant | ppl@512 | ppl@2048 | Notes |
|-------|---------|----------|-------|
| fp16 | 5.50 | 8.06 | baseline |
| int8_ch | 5.49 | 8.07 | essentially lossless |
| int4_ch:int4_tok | 5.53 | 8.15 | good |
| int3_ch | 5.90 | 8.75 | noticeable |
| int2_ch | 49.1 | 45.4 | catastrophic |

### Key findings
- **Per-channel quantization is critical**: int4 per-tensor = +937% PPL, int4 per-channel = +3%
- **K cache has 64/1024 outlier dims** (persistent across layers); V cache has none
- **int8_ch is lossless**, int4_ch surprisingly good, int3_ch shows real degradation, int2_ch catastrophic
- **int3_half_1357** (4-level non-uniform, avoids zero bin) is an experimental middle ground

### Performance model findings (perf_model.py --sweep)
- At B=1: almost everything is weight-dominated (kv% < 30%) — KV quant gives ≤1.32×
- At B=32: AIME flips to 92% KV-dominated → int3_half gives 3–5× speedup
- At B=128: AIME up to 6.35×, LongBench up to 4.95×

## Current State of Benchmark Runs

| Benchmark | Status | Last run | Issue |
|-----------|--------|----------|-------|
| WikiText PPL | Done | Mar 11 | `kv_sweep.json` |
| Code (LongCodeArena) | Done | Mar 18 | `sweep_code.json` |
| GSM8K | Done (all models) | Mar 31 | `server_results/{model}/gsm8k_*.json` |
| Qasper (LongBench) | Done (all models) | Mar 31 | `server_results/{model}/longbench_*.json` |
| NIAH | Partial | Mar 31 | Qwen3-8B stale (all 0%, needs rerun with thinking fix) |
| HumanEval | Not run | — | — |
| AIME | Partial | Mar 31 | Only fp16 for Qwen3; full sweep running on MN5 |
| Adaptive-sim | In progress | Mar 31 | See below |

## CURRENT TASK: Adaptive KV Sim — fix in progress (2026-03-31)

**Goal**: Measure acceptance rate of int3_half_1357_ch draft vs int4_ch verifier using the window-based adaptive scheme.

**What was implemented** (in this session):
- `strategies.py`: added `save_kv_state`, `restore_kv_state`, `_single_decode`, `verify_window`
- `run_sweep.py`: rewrote `run_adaptive_sim` as interleaved window loop with KV state save/restore
- Fixed position consistency bug (two sub-bugs):
  1. Window 0: batch-prefill only `pt[:-1]`, save `kv0`, then single-decode `pt[-1]` (so re-prime is at X+1)
  2. Windows 1+: `prime_pos = pos` not `pos-1` (token not yet decoded at save time)

**Current blocker**: crash in `strategies.verify_window` → `_flush_hook` when `n_pending_k=0` or `n_pending_v=0` causes `_int3_half_quant` to receive a zero-size array.

Traceback:
```
File "strategies.py", line 185, in verify_window
    _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v)
File "strategies.py", line 74, in _flush_hook
    kv_hook(ctx, n_new_k=n_pending_k if do_k else None, ...)
...
File "quant.py", line 373, in _int3_half_quant
    xmin = f32.min(axis=0)
ValueError: zero-size array to reduction operation minimum which has no identity
```

**Root cause**: `_flush_hook` calls the hook with `n_new_k=n_pending_k` (could be 0 if only V is pending, or some value < group_size). The window hook then calls `apply_kv_hook` which computes `arr[start:end]` where end-start=0. The quant function can't handle empty arrays.

**Fix needed** in `strategies.py` `_flush_hook`: guard against `n_pending_k == 0` or `n_pending_v == 0` before calling hook:
```python
def _flush_hook(kv_hook, ctx, n_pending_k, n_pending_v):
    do_k = n_pending_k > 0
    do_v = n_pending_v > 0
    if do_k or do_v:
        kv_hook(ctx,
                n_new_k=n_pending_k if do_k else None,
                n_new_v=n_pending_v if do_v else None)
```
Wait — that IS the current code. So the bug is that `n_pending_k > 0` but `n_pending_k < group_size`, and `apply_kv_hook` computes `start:end` = empty slice. Investigate `apply_kv_hook` in `parse_state.py` for how it handles partial groups (< group_size pending tokens).

**Test command** (once fixed):
```bash
python3 research/scripts/run_sweep.py /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
    research/data/gsm8k_test.jsonl --corpus-mode structured \
    --quants int3_half_1357_ch --adaptive-sim --verifier-quant int4_ch \
    --adaptive-window 32 --eval-accuracy --skip-ppl --n-ctx 2048 \
    --out results_adaptive.json --save-per-example results_adaptive_per_ex.json
```

Early results (before crash, first 3 examples): acc=1.00, draft_frac=0.95-1.00 — very promising.

## What Was Being Worked On Previously (performance model)

1. `perf_model.py` — high-level bandwidth model with `--sweep` grid ✅ done
2. `roofline_layer.py` — per-op roofline validated against Qwen2-7B/A100 ✅ done
3. Adaptive sim implementation ← current focus

## Critical API Pitfalls (always remember)

1. `llama_memory_clear(mem, bool)` — pass `llama_memory_t` from `llama_get_memory(ctx)`, NOT ctx directly → silent corruption → segfault
2. `llama_get_logits_ith(ctx, i)` — i is batch position; use `n_prompt-1` after prefill, `0` in decode
3. Default quant group size is **64** — both per-channel and per-token
4. Per-channel quant has zero effect if `n_decode < group_size` and `n_prompt = 0`; always prefill ≥ 1024
5. `--prompt-prefix "" --prompt-suffix ""` required for HumanEval (suppress chat auto-detect)

## Validation: roofline_layer.py vs Real Benchmarks (Mar 27)

### Source data

| Source | Model | GPU | Framework | Setting |
|--------|-------|-----|-----------|---------|
| [Qwen Speed Benchmark](https://qwen.readthedocs.io/en/v2.0/benchmark/speed_benchmark.html) | Qwen2-7B-Instruct | A100-80G | vLLM 0.4.2 | B=1, BF16, generate 2048 tokens |
| [LMDeploy docs](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html) | llama2-7b-chat | A100 | LMDeploy | bs=256, 10K prompts |
| [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) | Llama-2-7B/13B | A100 | Custom CUDA | 2-bit KV, K per-channel / V per-token |

### 1. Decode tok/s — Qwen2-7B, A100, BF16, B=1

We ran `roofline_layer.py` with Qwen2-7B parameters (28 layers, d=3584, nh=28, nkv=4, hd=128, ffn=18944).

| avg_ctx | Measured | Pred (eff=85%) | err | Pred (eff=55%/20%) | err |
|---------|----------|----------------|-----|---------------------|-----|
| 1,025   | 80.5     | 129.7          | +61% | 83.2               | +3% |
| 7,168   | 76.4     | 126.3          | +65% | 77.6               | +2% |
| 15,360  | 66.5     | 122.0          | +83% | 71.1               | +7% |
| 31,744  | 55.8     | 114.3          | +105% | 60.9              | +9% |
| 64,512  | 41.2     | 101.5          | +146% | 47.4              | +15% |
| 130,048 | 25.0     | 82.9           | +231% | 32.8              | +31% |

**Key finding**: With `mem_eff=55%` (weight BW utilization) and `attn_eff=20%` (attention/KV BW utilization), the model matches short-context reality to within 3-9%. Long-context error grows to +31% because flash attention efficiency degrades further at very long sequences.

### 2. KV cache quantization throughput impact

| Source | Method | Throughput gain | Our model prediction | Match? |
|--------|--------|-----------------|---------------------|--------|
| LMDeploy | INT8 KV, llama2-7b, bs=256 | +27% RPS | — | — |
| LMDeploy | INT4 KV, llama2-7b, bs=256 | +39% RPS | — | — |
| Our model | INT4 KV, B=32, ctx=32K | — | +44% tok/s | Close to LMDeploy +39% |
| Our model | INT4 KV, B=1, ctx=4K  | — | +7%  | Correctly small (weights dominate) |
| Our model | INT2 KV, B=32, ctx=32K | — | +44% | Diminishing returns vs INT4 (weights still 57%) |
| KIVI | 2-bit KV | 2.35–3.47× throughput | — | Achieved by 4× batch size, not per-token speedup |

### 3. Weight quantization (GPTQ-Int4) — model limitation

| avg_ctx | Measured (Int4 wt) | Pred (Int4 wt) | Over-prediction |
|---------|--------------------|----------------|-----------------|
| 1,025   | 143.4              | 321.1          | 2.2× |
| 130,048 | 29.4               | 46.3           | 1.6× |

Our model does **not** account for dequantization compute overhead of weight-quantized models (GPTQ, AWQ). Real Int4 weights give ~1.8× speedup over BF16, not 4× as pure bandwidth would suggest. This is a known limitation — the model correctly handles KV quantization (no dequant needed at inference time) but not weight quantization overhead.

### 4. Recommended efficiency calibration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `mem_eff` (weight BW) | 0.50–0.55 | Matches vLLM/LMDeploy reality |
| `attn_eff` (KV BW) | 0.20–0.25 | Flash attention is less BW-efficient than weight loading |
| `compute_eff` | 0.60–0.70 | Prefill compute utilization |

With these settings, the model is within 10% for short/medium contexts (≤32K) and 15-30% for very long contexts (64K-128K). The systematic over-prediction at long contexts comes from: (a) flash attention tiling inefficiency, (b) paged attention overhead (vLLM), (c) cache pollution effects.

### 5. Cross-model comparison: perf_model.py vs roofline_layer.py

| Metric | perf_model.py | roofline_layer.py |
|--------|---------------|-------------------|
| Qwen3-8B, B=1, fp16, 4K ctx decode tok/s | 104.5 | ~71 (eff=55%/20%) |
| KV quant speedup (int4_ch, B=1) | 1.035× | 1.07× |
| Granularity | Whole model | Per-op, per-layer |
| Efficiency | Single eff param | Separate weight/attn/compute |
| Better for | Quick sweep comparisons | Understanding bottleneck breakdown |

## Models

| Key | GGUF path | Notes |
|-----|-----------|-------|
| qwen3-8b | `/home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf` | Primary test model |
| qwen3-14b | registered in run_all_benchmarks.sh | |
| llama4-scout-17b | registered in run_all_benchmarks.sh | |
| gpt-oss-20b | registered in run_all_benchmarks.sh | Special chat format (`<\|start\|>`/`<\|end\|>`/`<\|return\|>`) |

## Build

```bash
# Release build (for research scripts)
cmake -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release -j $(nproc)
# Scripts use: build_release/bin/libllama.so

# CUDA build (for GPU quant path)
cmake -B build_release -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release -j $(nproc)
```
