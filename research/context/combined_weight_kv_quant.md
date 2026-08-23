# Combined Weight + KV Cache Quantization (Draft Model Idea)

Last updated: 2026-04-10

## 1. Motivation

Current adaptive-gen compresses only the **KV cache** (int2/int3) while keeping model
weights at Q8_0 (the baseline "full precision" for our experiments).  Two orthogonal
compression axes exist:

| Axis | What it saves | Current status |
|------|--------------|----------------|
| KV cache quant (int2/int3) | KV bandwidth during attention | ✅ implemented |
| Weight quant (Q4_K_M etc.) | Matrix-multiply compute + weight load BW | ❌ not yet tested |

The key question: **can a lower-precision draft model (Q4_K_M weights) combined with
int2/int3 KV be accepted by a higher-precision verifier (Q8_0 weights + fp16/int8 KV)?**
If yes, both compression axes contribute simultaneously and the speedup compounds.

## 2. Research Proposal

### 2.1 Base+Delta Weight Storage (novel, longer term)

Store model weights in two regions — mirroring the KV base+delta idea:

```
Full weight W_fp16 = W_base (Q4_K_M) + delta (small fp16 residual)

Storage layout:
  ┌───────────────────────┬───────────────────────┐
  │  Base (Q4_K_M)        │  Delta (fp16 diff)    │
  │  ~4 bits/param        │  ~16 bits/param × sparsity │
  └───────────────────────┴───────────────────────┘

Draft decode:  multiply using W_base only        → Q4_K_M speed
Verify decode: reconstruct W_fp16 = W_base+delta → full precision
```

Delta can be sparse (only large corrections stored) → total storage < fp16 but > Q4_K_M.
Same architecture, no small draft model needed. Weight memory shared between draft and verifier.

### 2.2 Simplified Measurement (near term, implementable now)

Skip base+delta storage. Use **two existing GGUF files** loaded as separate llama contexts:

- **Verifier context (`ctx`)**: `Qwen3-8B-Q8_0.gguf` — current baseline model
- **Draft context (`ctx_draft`)**: `Qwen3-8B-Q8_0-Q4_K_M.gguf` — quantized from Q8_0

Draft generates with Q4_K_M weights + int2/int3 KV hook.
Verifier replays with Q8_0 weights + fp16/int8 KV.
Accept/reject same as current adaptive-gen (greedy token match or top-k).

#### Experiment matrix

| Experiment | Weight (draft) | KV (draft) | Speedup axes |
|------------|---------------|------------|--------------|
| Baseline | Q8_0 | fp16 | — |
| KV only (current) | Q8_0 | int2/int3 | KV BW |
| Weight only | Q4_K_M | fp16 | matmul |
| Combined | Q4_K_M | int2/int3 | matmul + KV BW |

Acceptance rate of "combined" vs "KV only" shows the cost of additionally
quantizing weights. If acceptance stays ≥ 0.9, both compression axes are free.

## 3. Implementation Plan

### 3.1 One-Time Draft Model Creation

Use `llama-quantize` (magnitude-based, no imatrix needed for initial experiments):

```bash
./build_release/bin/llama-quantize \
    models/Qwen3-8B-Q8_0.gguf \
    models/Qwen3-8B-Q8_0-Q4_K_M.gguf \
    Q4_K_M
```

Helper script: `research/scripts/make_draft_model.py`
- Accepts source GGUF + quant type
- Auto-derives output path: `src.replace(".gguf", f"-{qt}.gguf")`
- Skips if output already exists

Quant levels to try: Q4_K_M, Q3_K_M, Q2_K (increasing aggression).

### 3.2 run_sweep.py Changes

1. **`--draft-model PATH`** — path to lower-precision GGUF. When set:
   - Load second `llama_model` + `llama_context` at startup (alongside existing `ctx`)
   - Pass `draft_ctx` into `run_adaptive_gen`

2. **`run_adaptive_gen` with `draft_ctx`**:
   - Prefill both `ctx` and `draft_ctx` with same prompt at example start
   - Draft phase: use `draft_ctx` + KV quant hook (int2/int3)
   - Verify phase: use `ctx` (Q8_0) + fp16 or int8 KV hook
   - **KV sync at window boundaries**: `save_kv_state(lib, ctx)` →
     `restore_kv_state(lib, draft_ctx, blob)` so draft's KV history always
     derives from the verifier's fp16 ground truth (same invariant as current adaptive-gen)

3. **KV compatibility**: Q8_0 and Q4_K_M are same architecture → identical
   `n_kv_heads`, `n_embd_head_k`, `n_layers` → KV blobs are layout-compatible
   between the two contexts. Direct restore works.

### 3.3 Memory Budget

Loading two contexts simultaneously:
- `ctx` (Q8_0): ~8 GB
- `draft_ctx` (Q4_K_M): ~4.5 GB
- Total: ~12.5 GB + KV caches (~2 GB each at 2048 ctx)
- Feasible on A100-40GB or 2× RTX 3090

## 4. Expected Results

Based on typical speculative decoding literature (same-architecture draft/target):
- Q4_K_M vs Q8_0 greedy token match rate: ~85–95% (model-dependent)
- Combined (Q4_K_M weights + int2 KV) vs Q8_0+fp16: lower, ~70–90%
- If combined acceptance ≥ 0.85: speedup from both axes simultaneously

Speedup model at B=1, ctx=4K:
- Weight quantization: ~1.5–2× (Q4_K_M vs Q8_0 matmul)
- KV quantization: ~6% (marginal at short context)
- Combined: ~1.5–2× (KV quant has small effect at short ctx)

At B=1, ctx=32K:
- Weight quantization: ~1.5–2×
- KV quantization: ~15%
- Combined: ~1.7–2.3×

## 5. Relation to Prior Work

| System | Draft model | KV quant | Acceptance mechanism |
|--------|------------|----------|---------------------|
| Standard SpecDec | Small separate model | No | Probabilistic rejection |
| KIVI | Same model | int2 rolling | N/A (always-on) |
| **Our adaptive-gen** | Same model | int2/int3 adaptive | Greedy window verify |
| **This proposal** | Same model, lower weight quant | int2/int3 adaptive | Greedy window verify |

Key difference from standard speculative decoding: no small draft model required —
the draft is the **same architecture** at lower weight precision, benefiting from the
existing KV adaptive framework without architectural changes.

## 6. Open Questions

1. Does int2 KV + Q4_K_M weight quant compound errors or are they independent?
2. Is the acceptance rate of "combined" roughly the product of individual acceptance rates?
3. Which weight quant level (Q4_K_M vs Q3_K_M) is the right operating point?
4. Does the imatrix (importance matrix) make Q4_K_M acceptable where magnitude-only isn't?
