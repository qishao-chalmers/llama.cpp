# Decode Timing Model (GGUF-native, simple, no fit file)

## Goal

Build a transparent decode estimator that:

1. Loads model dimensions directly from the local GGUF.
2. Uses real tensor byte sizes from GGUF tensors (supports mixed quant per layer).
3. Computes per-op FLOPs and bytes for one decode step.
4. Estimates op time from simple roofline math with fixed in-script efficiencies.
5. Outputs sweep tables compatible with `decode_sweep*.txt` and per-op detail text.

No calibration JSON, no learned hyperparameter table.

---

## Source of truth

Model metadata and tensors are loaded from GGUF:

- `general.architecture`
- `{arch}.block_count`
- `{arch}.embedding_length`
- `{arch}.feed_forward_length`
- `{arch}.attention.head_count`
- `{arch}.attention.head_count_kv` (fallback to `head_count`)
- `{arch}.attention.key_length` (fallback `d_model / n_heads`)
- `{arch}.attention.value_length` (fallback `key_length`)

Per-layer weight bytes come from exact tensor records:

- `blk.{L}.attn_q.weight`
- `blk.{L}.attn_k.weight`
- `blk.{L}.attn_v.weight`
- `blk.{L}.attn_output.weight`
- `blk.{L}.ffn_gate.weight`
- `blk.{L}.ffn_up.weight`
- `blk.{L}.ffn_down.weight`
- plus norm tensors where present.

This means different quantization per layer/tensor is naturally captured via actual `n_bytes`.

---

## One-step op list

Per layer, one decode step emits:

1. `attn_norm`
2. `attn_q`
3. `attn_k`
4. `attn_v`
5. `attn_q_norm` (if tensor exists, otherwise fallback size)
6. `attn_k_norm` (if tensor exists, otherwise fallback size)
7. `rope`
8. `kv_write`
9. `attn_qk`
10. `attn_softmax`
11. `attn_pv`
12. `attn_output`
13. `residual_attn`
14. `ffn_norm`
15. `ffn_gate`
16. `ffn_up`
17. `ffn_swiglu`
18. `ffn_down`
19. `residual_ffn`

Optional fixed runtime overhead can be added as `runtime_other`.

---

## FLOPs and bytes

### GEMM

For shape `[in_dim, out_dim]` from GGUF:

- `FLOPs = 2 * B * in_dim * out_dim`
- `bytes = weight_bytes + activation_in_bytes + activation_out_bytes`

Where activation bytes assume fp16 by default.

### Attention

Let `L = ctx_len`, `B = batch`, `H = n_heads`, `D = head_dim`, `Dkv = n_kv_heads * head_dim`, `Dv = n_kv_heads * value_dim`.

- `attn_qk FLOPs = 2 * B * H * L * D`
- `attn_softmax FLOPs = 5 * B * H * L`
- `attn_pv FLOPs = 2 * B * H * L * value_dim`

Bytes include Q, score buffers, and KV reads. KV bytes use CLI `--kv-bits`.

### Elementwise / norm / rope

Simple linear formulas on tensor size, all explicit in script.

---

## Timing formula per op

For each op:

- `Tcomp_min = FLOPs / peak_compute`
- `Tmem_min  = bytes / peak_bw`
- `Tmin = max(Tcomp_min, Tmem_min)`

Estimated time (using fixed family efficiencies):

- `Tcomp_est = Tcomp_min / eff_comp[family]`
- `Tmem_est  = Tmem_min  / eff_mem[family]`
- `Test = max(Tcomp_est, Tmem_est)`

Step time is strict sum of `Test` across emitted ops.

`ms/token = step_ms / B`  
`tok/s = 1000 * B / step_ms`

---

## Category aggregation for sweep table

Ops map to categories for output columns:

- `QKV+O`: `attn_q`, `attn_k`, `attn_v`, `attn_output`
- `RoPE`: `rope`
- `Attn`: `attn_qk`, `attn_softmax`, `attn_pv`
- `FFN`: `ffn_gate`, `ffn_up`, `ffn_swiglu`, `ffn_down`
- `Norm`: `attn_norm`, `attn_q_norm`, `attn_k_norm`, `ffn_norm`
- `Other`: `kv_write`, residuals, optional runtime overhead

This matches the row/cell style used by existing `decode_sweep*.txt`.

---

## Generated artifacts

Script: `research/scripts/simple_decode_timing.py`

Outputs:

1. Sweep text (duration + percentage tables), diff-friendly with `vimdiff`.
2. Per-op report with:
   - FLOPs
   - bytes
   - arithmetic intensity
   - estimated compute/memory/total ms
   - ideal min compute/memory/total ms at 100% efficiency
   - aggregate totals by op and category
