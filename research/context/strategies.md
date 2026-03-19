# Quantization Strategies

## Quantization Granularities

KV cache shape: [n_tokens, head_dim] (e.g. [n_tokens, 128] for Qwen3-8B)

| Name | axis | Scale per | group_size | Notes |
|------|------|-----------|------------|-------|
| `int4_ch` | 0 | column (dim) | 128 tokens | K outlier dims handled well |
| `int4_tok` | 1 | row (token) | 1 token | each token independent |
| `int4` | None | whole matrix | 128 tokens | catastrophic: outlier ruins all |
| `int4_ch:int4_tok` | K=0, V=1 | K per-col, V per-row | K=128, V=1 | best 4-bit |

**K cache**: 64/1024 dims are outliers across all 36 layers → per-channel critical for K.
**V cache**: more uniform per-dim → per-token works well for V.

## Group Size Semantics

`group_size` = number of decode tokens to accumulate before calling the quantization hook.

- **Per-channel (axis=0)**: needs group_size >= 128 for meaningful scale estimation.
  With group_size=1: each column has 1 value → scale = |val|/7 → exact (not real compression).
- **Per-token (axis=1)**: each row is independent → group_size=1 is correct.
  group_size=1 means quantize immediately after each token — correct for autoregressive:
  token N+1 is generated using quantized KV of token N, as a real system would do.

**Mixed K:V** (e.g. `int4_ch:int4_tok`):
- k_group_size = 128 (per-channel K needs group context)
- v_group_size = 1  (per-token V quantized immediately each token)
- Hook fires every 1 token (due to V); K is only quantized when k_pending >= 128

## Always-On Quantization (current default)

```
Prefill phase (n_prompt tokens, batch decoded):
  → quantize ALL n_prompt K cells at once (n_new_k=None)
  → quantize ALL n_prompt V cells at once (n_new_v=None)

Decode phase (token by token):
  → each token: n_pending_k += 1, n_pending_v += 1
  → when n_pending_k >= k_group_size: quantize last k_group_size K cells
  → when n_pending_v >= v_group_size: quantize last v_group_size V cells
  → each cell quantized exactly once — no re-quantization
```

## Multi-Window Scoring (current primary mode)

```
chunk = [prefill_tokens | decode_tokens (max_window)]
       e.g. [4096 prompt | 2048 decode]

1. Batch prefill 4096 tokens → quantize KV
2. Decode 2048 tokens (one at a time), collecting log_probs[0..2047]
3. ppl@512  = exp(-mean(log_probs[0:512]))
   ppl@1024 = exp(-mean(log_probs[0:1024]))
   ppl@2048 = exp(-mean(log_probs[0:2048]))
→ single run gives PPL at all three window sizes
```

## Strategy D — Prompt-only quantization (for ablation)

```
prefill → quantize prompt KV once → decode WITHOUT quantizing new tokens
```
Shows impact of prompt KV compression alone, vs full always-on.
Flag: `--prefill-batch --quantize-prompt-only`

## Why Per-Token V Needs group_size=1

In autoregressive decoding, quantizing token N's KV immediately (group_size=1) means
token N+1 is generated using quantized token N's KV — exactly what a real system does.

With group_size=128, tokens 1-127 are generated using UNQUANTIZED KV of each other,
then all 127 get quantized at once. This underestimates the error because 127 tokens
see better-than-real-system precision. Results would be optimistic.

group_size=1 is slow on CPU (2048 PCIe round-trips) but correct.
GPU-side quantization solves this — see context/gpu_quantization.md.

## Performance Notes

| Quant type | group_size | Hook calls (2048 decode) | Approx time |
|---|---|---|---|
| int4_ch | 128 | 16 | ~85s |
| int4_ch:int4_tok | K=128, V=1 | 2048 | ~3800s (CPU) / ~2s (GPU) |
| int4_tok | 1 | 2048 | ~3800s (CPU) / ~2s (GPU) |
