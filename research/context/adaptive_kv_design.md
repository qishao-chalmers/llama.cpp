# Adaptive KV Cache Quantization via Base+Delta Split Storage

Last updated: 2026-03-31

## 1. Core Idea

Store the KV cache in a **split memory layout** — a base region and a delta region —
within the same total memory footprint as standard quantization. By selectively reading
only the base during decode, we halve KV bandwidth. Periodic verification reads the
full base+delta and corrects the cache, maintaining accuracy without extra storage.

```
Standard int4 KV cache:   [████████████████]  4 bits per element

Our split layout:          [base][delta]
                            2-bit  2-bit   = 4 bits total (same size)

Draft decode:  read base only   → 2-bit bandwidth
Verify:        read base+delta  → 4-bit bandwidth (same as standard)
```

This also works with int3 storage: 2-bit base + 1-bit delta = 3 bits total.

## 2. The Adaptive Decode Scheme (revised 2026-03-31)

Verification uses **prefill** (one batched forward pass over W tokens), not a second
autoregressive decode — same insight as speculative decoding's batch verification.

### Phase 1 — Bootstrap (conservative, first W tokens)

**Terminology (aligned with `context/left_tasks.md` and `run_adaptive_gen`):** the first **W** fp16 tokens form the **bootstrap window**; each subsequent **W**-token block generated in draft quant is a **quant rollout window**, verified under fp16 (or `--verifier-quant`).

```
fp16:  autoregressive decode W tokens     → tokens[0:W]   (slow, ground truth)
int4:  prefill tokens[0:W] with int4 KV  → compare logit distributions  (fast)
```

- Check: KL(int4_logits, fp16_logits) < threshold (or exact token match)
- Accepted → switch to fast phase. Rejected → stay fp16.

### Phase 2 — Fast phase (repeating quant rollout windows)

```
int4:  autoregressive decode W tokens             → tokens_draft[P:P+W]  (fast)
fp16:  prefill tokens_draft[P:P+W] with fp16 KV  → verify + advance fp16 KV to P+W
       compress fp16 KV[P:P+W] → base+delta
       overwrite low-prec KV[P:P+W] with base+delta   ← KV correction
```

- Accepted → continue fast phase for next window
- Rejected → fall back to fp16 (or re-bootstrap after cooldown)

### KV Cache Invariant

base+delta ALWAYS holds fp16-derived values, never draft values.
The low-prec draft reads only the base (2-bit quantized fp16 ground truth).
Drift never accumulates — overwrite happens after every W tokens.

### Phase 3 — Fallback

If acceptance fails: stay fp16 for remaining tokens, or re-bootstrap after cooldown.

### Note on --adaptive-sim (current implementation is WRONG)

Current --adaptive-sim runs draft and verifier independently to full completion, then
compares post-hoc. After first divergence, both sequences are in different states —
the comparison is meaningless and does not simulate the KV overwrite.

Fix needed: rewrite as interleaved prefill-verification loop (see Section 7).
Real acceptance rates will be HIGHER than current sim reports.

## 3. Memory Layout

### int4 split (2+2)

```
Per KV element (4 bits total):

  Region 1 (base):   int3_half_1357  — 2 bits
                      Bins: {1, 3, 5, 7} (non-uniform, avoids zero)

  Region 2 (delta):  2 bits
                      Encodes the correction from base to full int4 (16 levels)

  Total: 4 bits = same as standard int4
```

### int3 split (2+1)

```
Per KV element (3 bits total):

  Region 1 (base):   int3_half_1357  — 2 bits
                      Bins: {1, 3, 5, 7}

  Region 2 (delta):  1 bit
                      Encodes which of 2 nearby int3 levels the value maps to

  Total: 3 bits = same as standard int3
```

### Physical memory organization

```
KV cache for one layer, one head (n_ctx tokens × head_dim elements):

  Contiguous in memory:
  ┌─────────────────────────┬─────────────────────────┐
  │     Base region         │     Delta region         │
  │  (2 bits × n_elements)  │  (1-2 bits × n_elements) │
  └─────────────────────────┴─────────────────────────┘

  Draft decode:  read [base] only          → pointer to base, stride = 2 bits
  Full decode:   read [base] + [delta]     → two reads, reconstruct int3/int4
  Write-back:    overwrite [base] + [delta] after verification
```

The two regions can be stored either interleaved (base+delta per element) or
split (all bases contiguous, all deltas contiguous). Split layout is better
for bandwidth because:
- Draft reads are sequential over the base region (better cache line utilization)
- Delta reads during verification can be a separate sequential stream

## 4. Why int3_half_1357 as the Base

Standard int2 symmetric quantization uses 3 bins: {-1, 0, +1}.
~80% of values land in the zero bin for normally-distributed KV data.
This wastes 80% of the representational capacity.

int3_half_1357 uses 4 bins: {1, 3, 5, 7} mapped onto int3's step size:
- Completely avoids the zero bin
- All 4 levels are utilized roughly equally
- 2 bits encode 4 levels — maximum information per bit
- Experimentally: int3_half_1357_ch achieves PPL close to int3_ch on Qwen3-8B

## 5. Comparison with Prior Work

### vs QuantSpec (ICML 2025)

| Aspect | QuantSpec | Our scheme |
|--------|-----------|------------|
| Draft model | Same arch, **4-bit weights** + 4-bit KV | Same model, **same weights**, 2-bit KV read |
| Target model | **FP16 weights** + FP16 KV | Same model, same weights, 3-4 bit KV read |
| Verify cost | Load FP16 weights (~16 GB for 8B model) | Read delta region only (1-2 extra bits per KV element) |
| KV storage | Hierarchical FP16→4-bit | Split base+delta, same total as int3/int4 |
| Starting precision | FP16 (then quantize to 4-bit for draft) | Already quantized (int3 or int4) |
| Extra memory | FP16 KV + 4-bit KV (hierarchical sharing) | Zero — same footprint as standard int3/int4 |
| KV correction | No — just accept/reject tokens | Yes — overwrite KV with ground truth after verify |

Key advantage: QuantSpec's verification requires loading full FP16 model weights,
which is the dominant cost at short-medium contexts. Our verification only reads
extra delta bits — negligible overhead compared to weight loading.

### vs TurboQuant (Google, ICLR 2026)

| Aspect | TurboQuant | Our scheme |
|--------|------------|------------|
| Approach | PolarQuant (random rotation) + QJL residual | Split storage, no transformation |
| Compression | 3-bit, near-optimal distortion | 2-bit draft from 3-4 bit storage |
| Overhead | Must rotate every KV vector | Zero — just address offset |
| Hardware complexity | Custom kernel for rotation + QJL | Simple: read from region 1 or region 1+2 |
| Accuracy | Zero loss at 3-bit | Depends on acceptance rate |

Key advantage: no data transformation required. Base+delta is a pure memory
layout decision, implementable as a pointer offset in the attention kernel.

### vs KIVI (ICML 2024)

KIVI established that K cache should be quantized per-channel and V cache per-token.
We adopt this finding. Our contribution is the adaptive precision switching and
split-storage layout on top of KIVI's quantization strategy.

## 6. Performance Model

### Bandwidth during fast phase (draft)

Only reading 2-bit base per KV element:
```
KV bytes per token per layer (draft):
  = n_kv_heads × head_dim × (2/8) × 2  (K + V)
  = 8 × 128 × 0.25 × 2 = 512 bytes     (Qwen3-8B)

vs standard int4:
  = 8 × 128 × 0.5 × 2 = 2048 bytes      (4× more)

vs fp16:
  = 8 × 128 × 2.0 × 2 = 4096 bytes      (8× more)
```

### Verification window cost

For W tokens of verification:
- Read W tokens × delta region (1-2 bits per element) — small
- Re-run W tokens through the model — like a short prefill of W tokens
- Write back corrected base+delta — same as normal KV write

The amortized overhead per generated token:
```
overhead_per_token = T_verify_window / W

For W=32: one verification every 32 tokens
  T_verify ≈ T_prefill(32 tokens) + T_delta_read
  Amortized overhead ≈ T_verify / 32
```

### When does this help?

The scheme helps when **KV bandwidth is a significant fraction of total decode time**.
From our roofline model validation:

| Scenario | KV % of decode | Draft speedup | Worth it? |
|----------|---------------|---------------|-----------|
| B=1, ctx=4K | ~12% | ~6% | Marginal |
| B=1, ctx=32K | ~30% | ~15% | Moderate |
| B=32, ctx=4K | ~12% | ~6% | Marginal |
| B=32, ctx=32K | ~55% | ~28% | **Yes** |
| B=128, ctx=32K | ~70% | ~35% | **Strongly yes** |

The sweet spot is **large batch × long context** — exactly the serving scenario
where KV cache is the dominant bottleneck.

## 7. What Needs to Be Done

### Experiments (using existing infrastructure)

1. **Acceptance rate measurement** (highest priority):
   ```bash
   python3 research/scripts/run_sweep.py \
       --model /path/to/Qwen3-8B-Q8_0.gguf \
       --quants int3_half_1357_ch \
       --adaptive-sim --verifier-quant int4_ch --adaptive-window 32 \
       --eval-accuracy --skip-ppl --corpus-mode structured \
       --corpus gsm8k --out results_adaptive.json \
       --save-per-example results_adaptive_per_ex.json
   ```
   Run on: gsm8k, aime, humaneval, niah
   Key metrics: acceptance_rate, first_fail_pos, draft_fraction

2. **Acceptance rate with int3 verifier** (also test int3 as the "full" precision):
   ```bash
   --quants int3_half_1357_ch --verifier-quant int3_ch
   ```

3. **Feed acceptance stats into performance model**:
   ```bash
   python3 research/scripts/perf_model.py \
       --model qwen3-8b --hw a100-80g \
       --sim-json results_adaptive_per_ex.json \
       --draft int3_half_1357_ch --verifier int4_ch
   ```

### Implementation (future — for paper submission)

4. **CUDA kernel**: Custom attention kernel that reads from split base+delta layout
   - Draft mode: read base region only (2-bit)
   - Verify mode: read base + delta, reconstruct full int3/int4
   - Write-back mode: split quantized value into base + delta, write both regions

5. **End-to-end benchmark**: Integrate into llama.cpp or vLLM, measure real speedup

6. **Accuracy evaluation**: Full benchmark suite with adaptive scheme active
   (not just simulated acceptance rate, but actual generated text quality)

## 8. Target Venues

- **MICRO 2026** / **HPCA 2027**: Hardware-efficiency story with CUDA kernel + roofline model
- **MLSys 2027**: Systems angle with end-to-end integration
- **Workshop paper** (ISCA/MICRO workshops): Establish priority with simulation results first

The strongest angle for architecture venues: **same memory, half the bandwidth,
zero extra storage, simple hardware implementation** — just a pointer offset
and a precision-mode bit in the attention kernel.
