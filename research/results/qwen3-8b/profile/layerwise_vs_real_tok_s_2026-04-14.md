# Layerwise vs real throughput tables (Qwen3-8B, H100-SXM)

**Date:** 2026-04-14

This note collects the throughput tables produced during the session and the
small “sim physics” knobs added to `research/scripts/layerwise_roofline_sim.py`.

## Setup (shared across tables)

- **Model preset:** `qwen3-8b` (measured file uses **Q8_0 weights**, `weight_bits=8.5`)
- **Hardware:** `h100-sxm`
- **mid_ctx:** 4096 (from `kv_timing_h100.json`)
- **Measured source:** `research/results/qwen3-8b/profile/kv_timing_h100.json`
- **Layerwise η:** `research/results/qwen3-8b/profile/layerwise_eta_h100.json`
- **Perf model:** `research/scripts/benchmark_kv_timing.py::roofline_ms`

## Table A — real vs layerwise vs perf-model (layerwise default dequant model)

Layerwise defaults include:
- `kv_attn_byte_mode=fp16_equiv_dequant` (native q8/q4 attention reads modeled as fp16 bytes × overhead)
- no attn overlap correction (`attn_time_scale=1`, `attn_time_scale_inv_batch=0`)

| KV | B | real tok/s | layerwise sim tok/s | perf_model sim tok/s |
|:---|--:|-----------:|-------------------:|---------------------:|
| f16 | 1 | 131.6 | 116.8 | 293.7 |
| f16 | 4 | 359.3 | 351.7 | 957.6 |
| f16 | 8 | 559.8 | 529.0 | 1536.2 |
| f16 | 16 | 874.5 | 707.3 | 2201.4 |
| f16 | 32 | 1225.1 | 850.7 | 2809.7 |
| q8_0 | 1 | 121.5 | 115.8 | 304.5 |
| q8_0 | 4 | 334.9 | 342.9 | 1082.7 |
| q8_0 | 8 | 495.8 | 509.4 | 1886.0 |
| q8_0 | 16 | 746.2 | 672.7 | 2998.3 |
| q8_0 | 32 | 1007.6 | 801.1 | 4252.2 |
| q4_0 | 1 | 121.8 | 115.8 | 310.6 |
| q4_0 | 4 | 339.7 | 342.9 | 1163.9 |
| q4_0 | 8 | 501.8 | 509.4 | 2146.7 |
| q4_0 | 16 | 747.2 | 672.7 | 3715.7 |
| q4_0 | 32 | 997.8 | 801.1 | 5855.3 |

## Table B — add a layerwise column *without* dequant overhead

Same as Table A, but adds `layerwise sim (no dequant)` which uses:
- `kv_attn_byte_mode=storage` (legacy bytes-only KV attention read)
- `attn_scale_by_batch={8: 0.9954, 16: 0.8063, 32: 0.7033}` (see Table C)

| KV | B | real tok/s | layerwise sim (dequant model) tok/s | layerwise sim (no dequant) tok/s | perf_model sim tok/s |
|:---|--:|-----------:|------------------------------------:|---------------------------------:|---------------------:|
| f16 | 1 | 131.6 | 116.8 | 116.8 | 293.7 |
| f16 | 4 | 359.3 | 351.7 | 351.7 | 957.6 |
| f16 | 8 | 559.8 | 530.2 | 530.2 | 1536.2 |
| f16 | 16 | 874.5 | 808.3 | 808.3 | 2201.4 |
| f16 | 32 | 1225.1 | 1104.9 | 1104.9 | 2809.7 |
| q8_0 | 1 | 121.5 | 115.8 | 123.1 | 304.5 |
| q8_0 | 4 | 334.9 | 342.9 | 416.3 | 1082.7 |
| q8_0 | 8 | 495.8 | 510.6 | 691.2 | 1886.0 |
| q8_0 | 16 | 746.2 | 771.7 | 1134.8 | 2998.3 |
| q8_0 | 32 | 1007.6 | 1045.7 | 1681.9 | 4252.2 |
| q4_0 | 1 | 121.8 | 115.8 | 126.7 | 310.6 |
| q4_0 | 4 | 339.7 | 342.9 | 460.0 | 1163.9 |
| q4_0 | 8 | 501.8 | 510.6 | 819.8 | 2146.7 |
| q4_0 | 16 | 747.2 | 771.7 | 1433.7 | 3715.7 |
| q4_0 | 32 | 997.8 | 1045.8 | 2302.5 | 5855.3 |

## Table C — fitted attention overlap correction for B=8/16/32 (all KV types)

We introduced a per-batch override:
- `attn_scale_by_batch[B]` multiplies only `attn_core` time (weights/FFN unchanged)

Fit (using kv ∈ {f16, q8_0, q4_0}, B ∈ {8,16,32}):

| B | attn_scale (multiplies attn_core time only) |
|--:|--------------------------------------------:|
| 8 | 0.9954 |
| 16 | 0.8063 |
| 32 | 0.7033 |

Saved as: `research/results/qwen3-8b/profile/layerwise_sim_physics_h100_attn_scale_8_16_32.json`

## Table D — layerwise what-if: q8_0 no-dequant baseline, scale memory BW ×2/×4

This is a “what-if” on the **layerwise** simulator:
- KV: `q8_0`
- `kv_attn_byte_mode=storage` (no dequant overhead)
- same `attn_scale_by_batch` as Table C
- only change is scaling hardware `memory_bw_gbps` by ×2 and ×4

| B | real tok/s | layerwise tok/s (×1 BW) | layerwise tok/s (×2 BW) | layerwise tok/s (×4 BW) |
|--:|-----------:|------------------------:|------------------------:|------------------------:|
| 1 | 121.5 | 123.1 | 246.3 | 492.6 |
| 4 | 334.9 | 416.3 | 832.7 | 1665.3 |
| 8 | 495.8 | 691.2 | 1382.5 | 2764.9 |
| 16 | 746.2 | 1134.8 | 2269.5 | 4539.1 |
| 32 | 1007.6 | 1681.9 | 3363.8 | 6727.7 |

