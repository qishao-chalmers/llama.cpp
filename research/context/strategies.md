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

## Group Size Semantics (Fairness Fix)

`group_size` = number of decode tokens to accumulate before calling the quantization hook.

Both per-channel and per-token quants now use `--quant-group-size` (default 128) for fair comparison.
Previously per-token forced group_size=1 (unfair: int4_tok saw a fresh hook every token while int4_ch saw only 1 call per 128 tokens).

- **Per-channel (axis=0)**: needs group_size >= 128 for meaningful scale estimation.
  With group_size=1: each column has 1 value → scale = |val|/7 → not real compression.
- **Per-token (axis=1)**: each row is independent. group_size=128 means 128 new tokens are accumulated, then all 128 get quantized together — same cadence as per-channel.

**Mixed K:V** (e.g. `int4_ch:int4_tok`):
- Both K and V now use `default_group_size` (e.g. 128)
- Hook fires every 128 tokens; K and V each quantize 128 cells at that point

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

## Three-Zone Quantization (KIVI-style)

Divides the KV cache into three zones with different quant aggressiveness:
```
[0, n_sink)          → sink zone   : fp16 or --quant-sink   (attention sinks)
[n_sink, T-n_recent) → stale zone  : main quant (--quants)  (heaviest compression)
[T-n_recent, T)      → recent zone : fp16 or --quant-recent (freshest tokens)
```
Flags: `--sink-tokens N`, `--recent-tokens R`, `--quant-sink`, `--quant-recent`.

The `_apply_window` wrapper in run_sweep.py implements this. During decode:
- Stale zone = tokens that just fell out of the recent window; quantized with `start_k` absolute offset.
- Recent zone tokens are quantized immediately with recent_hook (can be re-quantized to stale later).
- When `group_size > n_recent`, some new tokens bypass the recent zone directly to stale.

**Motivation**: "Lost in the middle" problem — aggressive middle compression may hurt RAG retrieval.
Protecting sinks (attention drain) and recent tokens (working memory) at higher precision recovers quality.

## Performance Notes

| Quant type | group_size | Hook calls (2048 decode) | Approx time |
|---|---|---|---|
| int4_ch | 128 | 16 | ~85s |
| int4_tok (fairness fix) | 128 | 16 | ~85s |
| int4_ch:int4_tok | 128 | 16 | ~85s |
| int4_tok (group_size=1, old behavior) | 1 | 2048 | ~3800s (CPU) / ~2s (GPU) |

With the fairness fix all per-channel and per-token quants use the same hook cadence — old per-token timing numbers are no longer relevant for the default configuration.
