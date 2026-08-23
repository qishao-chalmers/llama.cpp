# KV Cache Refresh from Teacher Model

Last updated: 2026-05-04

## 1. Core Idea

During small-model decode, periodically **overwrite** the small model's KV cache for
recently generated positions with the teacher (big) model's KV for those same positions.
The teacher computes its KV via a single parallel teacher-forcing pass (cheap), not
autoregressive generation.

```
Position:   0 ──── P ──── P+W ──── P+2W ──── P+3W ────
            [  prompt  ] [win 1 ] [ win 2  ] [ win 3  ]

Step 1: Teacher prefills prompt           → teacher_KV[0..P]
Step 2: Small model starts with teacher_KV[0..P] as its initial cache
Step 3: Small decodes W tokens autoregressively  → small generates tokens[P..P+W]
Step 4: Teacher teacher-forces on those W tokens → teacher_KV[P..P+W]  (one parallel pass)
Step 5: Overwrite small_KV[P..P+W] with teacher_KV[P..P+W]
Step 6: Repeat from Step 3
```

The small model always **generates** tokens (never blocked). The teacher's role is solely
to supply higher-quality KV — it never generates tokens in this scheme.

---

## 2. Motivation: Isolating Error Sources

Small model disagreement with the teacher has two independent causes:

| Source | Description | Accumulates? |
|--------|-------------|--------------|
| **(A) Weight imprecision** | W_small ≠ W_big → Q/K/V projections differ | No — fixed per layer |
| **(B) KV accumulation error** | Small's past wrong tokens → wrong K,V in cache → corrupts all future attention | Yes — compounds over time |

Baseline (no refresh) measures A + B together.
KV refresh eliminates B by periodically correcting the cache.

**The improvement = fraction of errors caused by KV accumulation.**

If `refresh_agreement >> baseline_agreement`:
→ KV accumulation is dominant; periodic refresh is highly effective.

If `refresh_agreement ≈ baseline_agreement`:
→ Weight imprecision dominates; better KV doesn't help much.

---

## 3. Comparison with Existing Adaptive Generation Scheme

| Scheme | Small model role | Big model role | Big model cost per W tokens |
|--------|-----------------|----------------|-----------------------------|
| **Adaptive gen** (existing) | Generates until rejected | Verifies; takes over for M tokens on rejection | Up to M autoregressive steps |
| **KV refresh** (this scheme) | Always generates | Teacher-forces every W tokens to produce KV | 1 parallel pass over W tokens (W× cheaper) |

KV refresh has a **fixed, predictable** big-model cost: one teacher-forcing pass per W tokens,
regardless of how often the small model "errs." No recovery window, no switching overhead.

---

## 4. Cost Model

Let:
- `small_ms` = small model ms/token (autoregressive)
- `big_tf_ms` = big model ms per token in teacher-forcing mode (parallel over W tokens)
  - Empirically `big_tf_ms ≈ big_ms / W` at small batch sizes (prefill is faster than decode)
- `W` = refresh window size

```
cost_per_token = small_ms + big_tf_ms
               = small_ms + big_ms / W

speedup_vs_pure_big = big_ms / (small_ms + big_ms / W)
```

At W=8, big_ms=4×small_ms (fp8 big, W4A16 small):

```
cost = small_ms + 4*small_ms/8 = 1.5 * small_ms
speedup = 4*small_ms / 1.5*small_ms ≈ 2.7×
```

Compare to adaptive gen with p_accept=0.85, W=8:

```
steady_S ≈ 0.85  →  effective cost ≈ 0.85*small_ms + 0.15*big_ms = 0.85+0.6 = 1.45*small_ms
speedup ≈ 2.8×
```

Both schemes give similar throughput, but KV refresh has **no variability** (fixed cost per
window vs stochastic recovery windows in adaptive gen).

---

## 5. KV Compatibility

Both models must have **identical architecture** (same num_layers, num_kv_heads, head_dim).
They differ only in weight precision (fp16 vs W4A16). This is always true when the small
model is a quantized version of the teacher.

The Q-K mismatch at decode time:
```
Q = X_small × W_Q_small   (small model weights)
K = X_teacher × W_K_big   (teacher's KV from cache)
```

For high-quality quantization (W4A16 GPTQ, W8A16), `W_Q_small ≈ W_Q_big` so the
dot product `Q·Kᵀ` is nearly correct. For aggressive quantization (W2, W3), mismatch
grows. The experiment measures this empirically.

---

## 6. What window=1 Means

`W=1` (refresh every token): small model attends to teacher's KV at every position.
The only remaining error source is weight imprecision in the Q/K/V projections themselves.
This gives the **lower bound** on how much KV refresh can help: the `W=1` agreement rate
is essentially the "pure weight imprecision" baseline.

---

## 7. Implementation

Script: `/home/qshao/Project/Fun/vllm_env/kv_refresh_decode.py`

Uses **HuggingFace transformers** (not vLLM) because the KV cache is a directly
accessible `DynamicCache` object with `key_cache[layer_idx]` and `value_cache[layer_idx]`
tensors that can be read and overwritten in-place.

```
DynamicCache.key_cache[layer_idx]   shape: (batch, num_kv_heads, seq_len, head_dim)
DynamicCache.value_cache[layer_idx] shape: (batch, num_kv_heads, seq_len, head_dim)
```

vLLM stores KV in paged blocks with no public read/write API — not suitable for this
research prototype.

### Run command

```bash
python3 kv_refresh_decode.py \
    --small-model ./Qwen3-8B-W4-simulated \
    --big-model   /path/Qwen3-8B-fp16 \
    --dataset     ../llama.cpp/research/data/gsm8k_test.jsonl \
    --window 1 4 8 16 \
    --max-gen-toks 128 \
    --n-examples 100 \
    --output kv_refresh_gsm8k.jsonl
```

### Output per example

```json
{
  "example_id": 0,
  "window": 8,
  "baseline":   {"agreement_rate": 0.705, "p_accept": 0.62, "window_rates": [...]},
  "kv_refresh": {"agreement_rate": 0.855, "p_accept": 0.79, "window_rates": [...]},
  "improvement": 0.150
}
```

### Interpretation printed at end of each window sweep

```
W=8 summary:
  baseline   agreement rate: 0.705
  kv_refresh agreement rate: 0.855
  improvement: +0.150
  Interpretation: 51% of baseline errors eliminated by KV refresh
```

---

## 8. Relation to Other Work

| Work | Relation |
|------|----------|
| **measure_adaptive_agreement.py** | Scheme B (p_accept) is this experiment's baseline — same measurement without the refresh intervention |
| **Speculative decoding** | Draft model generates, target model verifies/corrects — similar spirit but target generates tokens; here big model only provides KV |
| **Adaptive KV base+delta** (`adaptive_kv_design.md`) | Corrects KV precision within one model's cache; this corrects across two models |
| **KV cache distillation** (research area) | Using teacher attention patterns to guide student — this is a runtime version |
