# Qwen3-8B CUDA Kernel Trace Mapping (NVBit / llama-bench)

Traces from BSC NVBit runs on Qwen3-8B-Q8_0, H100, `-ngl 99`, `-fa 1` unless noted.

| Trace file | Command regime |
|------------|----------------|
| `research/results/kernel_name.log` | `-p 1024 -n 32 -b 16 -fa 1` (prefill chunked) |
| `research/results/kernel_name_flash_attention_batch_2048.log` | `-p 1024 -n 32 -b 2048 -fa 1` (single-chunk prefill) |
| `research/results/kernel_name_batch_2048.log` | `-p 1024 -b 2048` **no `-fa`** (classic attn: CUTLASS + softmax) |
| `research/results/kernel_name_batchsize1.log` | `-p 1024 -n 32 -b 1 -fa 1` (decode-style: 1 token / graph) |

## Critical: `-b` is prefill chunk size, not parallel sequences

`-b` = `n_batch` = max tokens per `llama_decode()` during **prefill** only.
Decode (`-n`) always uses **1 token** per `llama_decode()` regardless of `-b`.

---

## Prefill stage (`-p 1024 -b 2048 -fa 1`)

**One full layer ≈ 34 launches** (layer 0); **≈ 33 launches** (layers 1–35, no `cpy_scalar` in ATTN).

**Full model (36 layers, 1 prefill forward): ≈ 1189 kernels**

### ATTN block — 22 launches (layer 0) / 21 (layers 1+)

| # | CUDA kernel | Qwen3-8B op | CUDA PERF label |
|---|-------------|-------------|-----------------|
| 1 | `rms_norm_f32<1024>` | `blk.L.attn_norm` | `norm:RMS_NORM` |
| 2 | `quantize_mmq_q8_1` | Q act quant | (helper) |
| 3 | `mul_mat_q<Q8_0, tile=128>` | `blk.L.attn_q.weight` | `Qcur:MUL_MAT` |
| 4 | `mul_mat_q_stream_k_fixup` | MMQ fixup | (helper) |
| 5 | `rms_norm_f32<256>` | `blk.L.attn_q_norm` | `norm:RMS_NORM` |
| 6 | `rope_neox<f32→f32>` | RoPE(Q) | `Qcur:ROPE` |
| 7–9 | quant + mmq + fixup | `blk.L.attn_k.weight` | `Kcur:MUL_MAT` |
| 10–12 | quant + mmq + fixup | `blk.L.attn_v.weight` | `Vcur:MUL_MAT` |
| 13 | `rms_norm_f32<256>` | `blk.L.attn_k_norm` | `norm:RMS_NORM` |
| 14 | `rope_neox<f32→f16>` | RoPE(K) | `Kcur:ROPE` |
| 15 | `k_set_rows<f16>` | KV cache write | `cache_k/v:SET_ROWS` |
| 16 | `cpy_scalar_contiguous` | attn staging | `(copy):CPY` **(layer 0 only)** |
| 17 | `flash_attn_ext_f16<ncols=16>` | flash attention | `__fattn__:FLASH_ATTN_EXT` |
| 18 | `flash_attn_stream_k_fixup` | flash fixup | (helper) |
| 19–21 | quant + mmq + fixup | `blk.L.attn_output.weight` | `result_output:MUL_MAT` |
| 22 | `k_bin_bcast ADD` | attn residual | `l_out:ADD` |

### FFN block — 12 launches (every layer)

| # | CUDA kernel | Qwen3-8B op | CUDA PERF label |
|---|-------------|-------------|-----------------|
| 1 | `rms_norm_f32<1024>` | `blk.L.ffn_norm` | `norm:RMS_NORM` |
| 2–4 | quant + mmq + fixup | `blk.L.ffn_up.weight` | `ffn_up:MUL_MAT` |
| 5–7 | quant + mmq + fixup | `blk.L.ffn_gate.weight` | `ffn_gate:MUL_MAT` |
| 8 | `unary_gated_op_kernel<silu>` | SiLU(gate) × up | `ffn_swiglu:GLU` |
| 9–11 | quant + mmq + fixup | `blk.L.ffn_down.weight` | `ffn_out:MUL_MAT` |
| 12 | `k_bin_bcast ADD` | ffn residual | `ffn_inp:ADD` |

**Note:** `-b 16` uses `mul_mat_q<tile=16>` instead of `tile=128`; same structure.

**Prefill without `-fa`:** replace steps 17–18 with 9 classic-attn kernels (cast, `k_compute_batched_ptrs`, CUTLASS QK, `soft_max_f32`, CUTLASS ×V, cast, cpy) → **40 launches/layer**.

---

## Decode stage (`-n *`, any `-b`, `-fa 1`)

**One full layer ≈ 26–28 launches** (1 token per step).

**One decode token × 36 layers ≈ 950–1000 kernels**

### ATTN block — ~17 launches

| # | CUDA kernel | Qwen3-8B op |
|---|-------------|-------------|
| 1 | `rms_norm_f32<1024>` | `blk.L.attn_norm` |
| 2 | `quantize_q8_1` | Q act quant |
| 3 | `mul_mat_vec_q` | `blk.L.attn_q.weight` |
| 4 | `rms_norm_f32<256>` | `blk.L.attn_q_norm` |
| 5 | `rope_neox<f32→f32>` | RoPE(Q) |
| 6–7 | quant + gemv | `blk.L.attn_k.weight` |
| 8–9 | quant + gemv | `blk.L.attn_v.weight` |
| 10 | `rms_norm_f32<256>` | `blk.L.attn_k_norm` |
| 11 | `rope_neox<f32→f16>` | RoPE(K) |
| 12 | `k_set_rows<f16>` | KV cache write |
| 13 | `flash_attn_ext_vec` | flash attention (ncols=1) |
| 14 | `flash_attn_combine_results` | flash vec combine |
| 15–16 | quant + gemv | `blk.L.attn_output.weight` |
| 17 | `k_bin_bcast ADD` | attn residual |

### FFN block — ~11 launches

| # | CUDA kernel | Qwen3-8B op |
|---|-------------|-------------|
| 1 | `rms_norm_f32<1024>` | `blk.L.ffn_norm` |
| 2–3 | quant + gemv | `blk.L.ffn_up.weight` |
| 4–5 | quant + gemv | `blk.L.ffn_gate.weight` |
| 6 | `unary_gated_op_kernel<silu>` | SiLU(gate) × up |
| 7–8 | quant + gemv | `blk.L.ffn_down.weight` |
| 9 | `k_bin_bcast ADD` | ffn residual |

No MMQ fixup in decode — GEMV has no `stream_k_fixup`.

---

## Phase markers (unique kernel families)

| Shared (both) | Prefill only (`-fa 1`) | Decode only |
|---------------|------------------------|-------------|
| `rms_norm_f32<1024/256>` | `quantize_mmq_q8_1` | `quantize_q8_1` |
| `rope_neox` Q/K | `mul_mat_q` + fixup | `mul_mat_vec_q` |
| `k_set_rows` | `flash_attn_ext_f16` + fixup | `flash_attn_ext_vec` |
| `unary_gated silu` | | `flash_attn_combine_results` |
| `k_bin_bcast ADD` | | |
| `cpy_scalar_contiguous` (L0) | | |

```
See mul_mat_vec_q or flash_attn_ext_vec  →  decode
See mul_mat_q or flash_attn_ext_f16      →  prefill
```

---

## Kernel counts (Qwen3-8B, 36 layers)

| Scenario | Approx launches |
|----------|-----------------|
| One prefill forward (`-p 1024 -b 2048 -fa 1`) | **~1189** |
| One decode token (`-fa 1`) | **~1000** |
| `-n 8` decode after prefill | **~8000** |
| `-p 1024 -b 16` (64 chunk replays) | **~64 × 1189 ≈ 76k** |

### llama-bench warmup (default, no `--no-warmup`)

Before measured `-n 8` decode: warmup prefill (~1189) + warmup gen 1 token (~1000) + measured prefill (~1189) ≈ **3400 kernels** before timed decode phase.

---

## Compact per-layer templates

**Prefill (`-fa 1`):**
```
norm → [quant+MMQ+fixup]×4 (Q,K,V,O) + q/k norms + rope + kv + [cpy L0] + flash_f16×2 + residual
     → norm → [quant+MMQ+fixup]×3 (up,gate,down) + swiglu + residual
```

**Decode (`-fa 1`):**
```
norm → [quant+GEMV]×4 (Q,K,V,O) + q/k norms + rope + kv + flash_vec×2 + residual
     → norm → [quant+GEMV]×3 (up,gate,down) + swiglu + residual
```

---

## Quant type note (Q2_K / Q4_K / Q8_0)

Same **kernel family** structure; only the `ggml_type` template parameter and inner `vec_dot_*` change.
See `research/context/qwen3_kernel_trace_mapping.md` § Quant types below, and `gguf_quant_sizes.md` for per-tensor assignments in K-quant blends.

### GEMM/GEMV dispatch (all quant weights)

| Condition | Path | Act quant |
|-----------|------|-----------|
| `src1->ne[1] <= 8` (decode) | `mul_mat_vec_q` (GEMV) | `quantize_q8_1` |
| `src1->ne[1] > 8` (prefill) | `mul_mat_q` (MMQ) | `quantize_mmq_q8_1` |

Mangled name examples:
- Q8_0: `mul_mat_q<ggml_type8>` / `mul_mat_vec_q<ggml_type8>`
- Q2_K: `mul_mat_q<ggml_type10>`
- Q4_K: `mul_mat_q<ggml_type12>`

**Speedup from lower quant:** mainly **less weight bytes loaded** (memory-bandwidth bound), not a different kernel algorithm. K-quants (Q2_K, Q4_K) use heavier dequant inside the same kernel shell than Q8_0.

**Q8_K:** legacy/internal type; standard GGUF models use **Q8_0**. Q8_K is not in the MMQ/MMVQ switch — falls back to cuBLAS + dequant if used.

**Mixed models (Q4_K_M):** one forward pass uses multiple `ggml_type` instantiations (Q4_K, Q5_K, Q6_K per tensor group) — see `gguf_quant_sizes.md`.
