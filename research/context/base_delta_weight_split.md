# Base+Delta Weight Split: Q4_K_M Draft / Q8_0-Quality Verify from One Model

## 0. Implementation Status (as of 2026-04-28)

The offline tooling and ggml types/kernels are **fully implemented**.
Runtime switching and inference integration are **pending**.

| Component | Status | Notes |
|-----------|--------|-------|
| `build_q4km_delta.py` offline script | **DONE** | Phase 0 complete |
| `GGML_TYPE_Q4_K_RES` (type 46, verify) in `ggml.h` | **DONE** | implemented as Q4_K_RES, not Q4_K_D4 |
| `GGML_TYPE_Q4_K_RES_DRAFT` (type 45, draft) in `ggml.h` | **DONE** | reads base Q4_K block only |
| CUDA MMVQ kernels (`mmvq.cu`) | **DONE** | `vec_dot_q4_k_res_{draft,verify}_q8_1` |
| GPU convert kernels (`convert.cu`) | **DONE** | |
| Built GGUF on disk | **DONE** | `models/Qwen3-8B-Q8_0-Q4_K_M_q4k_res.gguf` |
| Runtime mode switching API (`llama_set_weight_mode`) | **PENDING** | Phase 3 |
| Python binding exposure | **PENDING** | Phase 3 |
| `run_adaptivegen.py` integration | **PENDING** | Phase 4 |
| Evaluation (PPL, GSM8K, tok/s) | **PENDING** | Phase 5 |

**Key format note:** The implemented format is `Q4_K_RES` (two Q4_K blocks: base + residual Q4_K),
NOT `Q4_K_D4` (base Q4_K + int4 integer delta in Q8_0 units) as originally designed.
Layout: 288 bytes per 256-element superblock in both cases (144 B base + 144 B residual/delta).
The Q4_K residual (implemented) is lossy; the int4 delta design (§4b) would give exact Q8_0 reconstruction.

**Runtime switching:** Draft/verify is controlled by tensor type id — type 45 → draft kernel (reads only
first 144 B of each block), type 46 → verify kernel (reads all 288 B). The GGUF on disk has type 46
tensors. To run in draft mode, tensor type fields must be mutated to 45, or a `llama_set_weight_mode`
API must dispatch the draft kernel regardless of stored type. This is the only missing piece.

---

## 1. Motivation

The split2 design (see `split_weight_precision.md`) shows how to get a Q4-quality draft
and Q8_0-quality verify from a single model by nibble-splitting each int8 weight.  That
works cleanly because the upper nibble is a **bit-subset** of the Q8_0 int8, so the delta
(lower nibble) has zero error and exactly 4 bits.

This document extends the idea to a different starting point: a standard **Q8_0-Q4_K_M
mixed-quantization GGUF** (produced by `tools/quantize` with imatrix importance weighting).
In this format the less-important FFN tensors are already stored as Q4_K_M (~4.5 bpw),
while the more-sensitive attention tensors stay at Q8_0 (~8 bpw).

Idea: **store a per-element delta alongside each Q4_K_M block**.  Draft reads only the
Q4_K_M base (fast, 4.5 bpw).  Verify reads base + delta and reconstructs weights close to
full Q8_0 precision.  The Q8_0 tensors require no delta at all — draft and verify both use
the same Q8_0 path.

Result: one model file, two runtime decode paths, shared KV cache, no second model in VRAM.


## 2. Why This Is Harder Than Split2

In split2 the relationship between draft and verify is exact and bitwise:

```
Q8_0 int8 byte:  [b7 b6 b5 b4 | b3 b2 b1 b0]
                  ^-- Region A (draft) --^  ^-- Region B (delta) --^

draft:  scale_A × nibble_A           (scale_A = scale × 16)
verify: scale   × (nibble_A<<4 | nibble_B)   (exact Q8_0 reconstruction)
delta:  nibble_B  ∈ [0..15],  always non-negative,  always 4 bits
```

Q4_K_M does **not** have this bitwise relationship to Q8_0:

| property | split2 | Q4_K_M + delta |
|---|---|---|
| Base format | upper nibble of Q8_0 int8 | Q4_K_M (different block structure, different scale hierarchy) |
| Delta domain | integer (nibble_B, exact) | float (Q8_0_float − Q4_K_M_float) |
| Delta magnitude | bounded ∈ [0..15] × scale | depends on Q4_K_M error; typically ≤ 5% of weight range |
| Verify precision | **exact** Q8_0 | **approximate** Q8_0 (delta itself is quantized) |
| Delta size | 4 bits/element (free from nibble packing) | requires its own quantized storage |

The delta values are small-magnitude (they are quantization residuals), which means they
can be stored at 4-bit precision with much finer effective resolution than the original
weights.  This is what makes the scheme viable despite living in float-space.


## 3. Delta in Q8_0 Coordinate Frame: Exact Reconstruction

The delta is NOT defined in float-space.  It is defined in the **integer space of Q8_0**,
which allows exact reconstruction of the Q8_0 dequantized value.

For a weight element `i` in Q8_0 block `b` (32 elements, scale `d_q8[b]`):

```
q8_int[i]     — the true Q8_0 signed int8 weight (from the Q8_0 GGUF)
q4km_float[i] — the Q4_K_M dequantized float (from the mixed GGUF)

projection:   q4km_proj[i] = clamp(round(q4km_float[i] / d_q8[b]), -128, 127)
delta[i]      = q8_int[i] - q4km_proj[i]   ← signed integer
```

**Verify reconstruction (exact Q8_0)**:

```
q8_int_rec[i] = q4km_proj[i] + delta[i]   = q8_int[i]   (if delta doesn't saturate)
w_verify[i]   = q8_int_rec[i] * d_q8[b]   = q8_int[i] * d_q8[b]   ← exact Q8_0 dequant
```

**Bound on delta magnitude**:

```
Q4_K_M has 16 levels covering the same signed range as Q8_0's 256 levels.
Each Q4_K_M step ≈ 16 Q8_0 steps.
The Q4_K_M dequant for element i lands within ±8 Q8_0 integer units of q8_int[i].

→  |delta[i]| ≤ 8  in Q8_0 integer units, for typical weight distributions.
```

Signed int4 covers −8..7.  The saturation case `delta = −8` (which would need int4 value
`−8`) fits exactly when stored as `0x8` in a biased encoding; clamping only occurs when
`delta = +8` → stored as `+7` (1-ULP error on one element).  The Phase 0 script will
measure how often this actually saturates across all Qwen3-8B FFN tensors.


## 4. Storage Layout

### 4a. Tensor classification

| tensor role | standard assignment in Q8_0-Q4_K_M | draft path | verify path |
|---|---|---|---|
| attn Q/K/V/O projections | Q8_0 | Q8_0 (unchanged) | Q8_0 (unchanged) |
| FFN gate / up / down | Q4_K_M | Q4_K_M base only | Q4_K_M + int4 delta |
| token embeddings | Q6_K or fp32 | unchanged | unchanged |
| LM head output | Q8_0 or fp32 | unchanged | unchanged |
| RMSNorm gamma | fp32 | unchanged | unchanged |

Only the Q4_K_M tensors carry delta blocks.  Q8_0 tensors are untouched.

### 4b. New block type: `block_q4km_d4`

Q4_K_M stores weights in super-blocks of 256 elements (144 bytes).  Each super-block
covers 8 Q8_0-sized sub-blocks of 32 elements.  The delta is stored at Q8_0 sub-block
granularity so that the Q8_0 scale `d_q8` per 32 elements is available for exact integer
reconstruction:

```
struct block_q8_0_delta {            // NEW — 18 bytes per 32 elements
    ggml_half  d_q8;                 // Q8_0 scale for this 32-element sub-block  (2 bytes)
    uint8_t    dq[16];               // 32 × int4 delta, packed 2/byte            (16 bytes)
};
// int4 delta: biased unsigned storage, bias=8  →  stored value = delta + 8  ∈ [0..15]
// signed range: delta ∈ [−8, 7]  (covers all expected Q4_K_M → Q8_0 residuals)

struct block_q4km_d4 {               // NEW — 144 + 8×18 = 288 bytes per 256 elements
    block_q4_K          base;        // standard Q4_K super-block                (144 bytes)
    block_q8_0_delta    delta[8];    // one delta sub-block per 32-element group (144 bytes)
};
// Verify bpw:  288 × 8 / 256  =  9.0 bpw
// Draft bpw:   144 × 8 / 256  =  4.5 bpw  (reads base only, skips delta entirely)
```

Verify reconstruction for sub-block `b` of 32 elements:

```c
float d_q8 = delta[b].d_q8;
for (int i = 0; i < 32; i++) {
    uint8_t packed  = delta[b].dq[i / 2];
    int     d_raw   = (i % 2 == 0) ? (packed & 0xF) : (packed >> 4);
    int     d_int   = d_raw - 8;                         // signed, range −8..7
    int q4km_proj   = (int)roundf(q4km_float[b*32+i] / d_q8);
    q4km_proj       = clamp(q4km_proj, -128, 127);
    int q8_rec      = q4km_proj + d_int;                 // = original Q8_0 int8 (exact)
    w_verify[b*32+i] = q8_rec * d_q8;                   // exact Q8_0 dequant
}
```

### 4c. Effective bandwidth comparison

| mode | Q8_0 tensors | Q4_K_M+delta tensors | blended (typical Qwen3-8B) |
|---|---|---|---|
| verify (full) | 8.06 bpw | 9.0 bpw | ~8.6 bpw |
| draft (base only) | 8.06 bpw | 4.5 bpw | ~5.8 bpw |
| speedup (draft vs verify) | 1.0× | 2.0× | ~1.5× |

Verify is ~7% slower than plain Q8_0 for the FFN tensors, but gives **exact** Q8_0
reconstruction — not approximate.  The delta overhead (144 bytes per super-block) is the
price of lossless recovery without holding a full Q8_0 model in memory.

The blended speedup (1.4×) accounts for Q8_0 attention tensors being the same in both
modes.  If FFN layers dominate weight bandwidth (true for MLP-heavy models like Qwen3),
this approaches 1.7–1.9× for the FFN portion.

### 4d. Per-layer precision variation

The Q8_0-Q4_K_M assignment already varies per layer in some quantization recipes
(first/last layers may retain Q8_0 for more tensors).  The scheme handles this naturally:

- Any tensor typed `GGML_TYPE_Q4_K` → gets a `block_q4km_d4` block (with delta)
- Any tensor typed `GGML_TYPE_Q8_0` → no delta, same kernel in both modes
- A per-tensor metadata flag marks whether a delta block is present

This flag lives in the GGUF key-value store as a string-set of tensor names that carry
deltas, or as a new tensor type `GGML_TYPE_Q4_K_D4`.


## 5. Offline Delta Computation

Given the original full-precision model (fp16/bf16) and the Q8_0-Q4_K_M GGUF:

```python
for each Q4_K_M tensor T:
    W_fp   = load_fp16_tensor(original_model, T.name)      # [rows, cols]
    W_q4km = dequant_q4km(T.data)                          # [rows, cols], float32
    delta  = W_fp - W_q4km                                 # residual in float32

    # Quantize delta to signed int4 at super-block granularity
    for each 256-element super-block b:
        max_abs = max(|delta[b]|)
        d_delta = max_abs / 7.0                            # signed int4 range is -8..7
        dq[b]   = round(delta[b] / d_delta).clip(-8, 7)   # int4
        # pack dq[b] as two nibbles per byte into dq_packed

    # Write block_q4km_d4: base Q4_K block + d_delta + dq_packed
```

**Note**: the delta can also be computed relative to the Q8_0 dequantized value (not fp16)
if the target is to match Q8_0 exactly rather than recover fp16 quality:

```python
W_q8_0  = dequant_q8_0(quantize_q8_0(W_fp))   # round-trip through Q8_0
delta   = W_q8_0 - W_q4km                      # target: match Q8_0, not fp16
```

This is preferable because it sets a well-defined verify target (Q8_0) and means the
delta values are smaller (Q4_K_M → Q8_0 gap is smaller than Q4_K_M → fp16 gap).

**Tooling**: implement as a Python script `research/scripts/build_q4km_delta.py` that
reads both GGUFs and writes a new GGUF with `GGML_TYPE_Q4_K_D4` tensor types.


## 6. CUDA Kernels

### 6a. Draft kernel (Q4_K_M only, existing kernel)

Draft mode uses the standard `mul_mat_q4_K_q8_1` MMVQ kernel unchanged.  The delta block
bytes are skipped (they follow the base block in memory but are never loaded).  No new
kernel is needed for draft.

### 6b. Verify kernel (Q4_K_M + int4 delta → dequant → GEMV)

For each weight element `i` in a super-block:

```cuda
// Load base Q4_K dequantized value (existing logic)
float w_base = dequant_q4k_element(base, i);

// Load delta
uint8_t packed = dq[i / 2];
int4_t  d_raw  = (i % 2 == 0) ? (packed & 0xF) : (packed >> 4);
int     d_sign = (int)d_raw - 8;          // signed -8..7
float   w_delta = d_delta * (float)d_sign;

// Final dequantized weight
float w = w_base + w_delta;
```

The verify kernel can be structured as a modified `dequant_q4_K_f32` that loads the
interleaved delta block from `base_ptr + sizeof(block_q4_K)`.

### 6c. Kernel dispatch

Runtime mode is controlled by `ggml_backend_cuda_context::q4km_verify` (bool), set via
the same `ggml_backend_cuda_reg_get_proc_address` / proc-address mechanism as split2:

```cpp
// In ggml-cuda.cu
static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    ...
    if (strcmp(name, "ggml_backend_cuda_set_q4km_verify") == 0) {
        return (void *)ggml_backend_cuda_set_q4km_verify_impl;
    }
    ...
}
```

Public API:

```c
// include/llama.h
void llama_set_weight_mode(llama_context * ctx, int mode);
// mode 0 = draft (Q4_K_M base only for Q4_K_D4 tensors)
// mode 1 = verify (Q4_K_M + delta for Q4_K_D4 tensors)
```

### 6d. Interaction with cuBLAS path

For batch sizes > 1 (prefill), cuBLAS is used.  The cuBLAS path needs a `to_fp16_cuda`
dequant function for `GGML_TYPE_Q4_K_D4`.  Two variants:

- `to_fp16_q4km_d4_draft`: dequant base only (for draft prefill)
- `to_fp16_q4km_d4_verify`: dequant base + delta (for verify prefill / bootstrap)

These are register in `ggml_cuda_op_mul_mat` alongside other type dequant functions.


## 7. GGUF Format Extension

### 7a. New tensor type

Register `GGML_TYPE_Q4_K_D4` (= `GGML_TYPE_Q4_K` with interleaved int4 delta):

```c
// ggml/include/ggml.h
GGML_TYPE_Q4_K_D4 = 33,   // Q4_K base + int4 delta per super-block
```

Block size: 274 bytes per 256 elements.

### 7b. Metadata key

A GGUF key-value entry lists which tensors carry delta blocks:

```
llama.weight.delta_tensors  (string, comma-separated tensor names)
```

Or more simply: any tensor with type `GGML_TYPE_Q4_K_D4` implicitly carries a delta.
Model loaders check the tensor type and dispatch accordingly.

### 7c. Backward compatibility

A GGUF with `GGML_TYPE_Q4_K_D4` tensors loaded without delta-aware code falls back to
treating the block as Q4_K (reading only the first 144 bytes of each 274-byte block).
This gives Q4_K_M-quality inference without crashing — a safe degraded mode.


## 8. Integration with run_adaptivegen.py

The switch mode in `run_adaptivegen.py` maps cleanly onto this scheme:

```
bootstrap (verifier, mode=1):   generate W tokens at verify quality
agree-rate estimation (mode=0): single-decode loop through draft context
                                  → measures Q4_K_M-only quality
rollout:
  if agree_rate >= threshold:
    draft window at mode=0  (Q4_K_M base, fast)
    verify window at mode=1  (Q4_K_M + delta, near-Q8_0 quality)
  else:
    verify-only at mode=1
```

The `_set_mode` helper already exists; it just needs to call `llama_set_weight_mode`
instead of `llama_set_split2_mode`.  No other changes to the Python script logic.


## 9. Comparison with Related Approaches

| approach | draft | verify | memory | accept rate expectation |
|---|---|---|---|---|
| Separate small model | Qwen3-0.6B (Q8_0) | Qwen3-8B (Q8_0) | 2 models | moderate (arch mismatch) |
| split2 (this repo) | Qwen3-8B nibble (~Q4) | Qwen3-8B Q8_0 (exact) | 1 model + 6% | high (same arch+weights) |
| **this scheme** | Qwen3-8B Q4_K_M | Qwen3-8B ~Q8_0 (approx) | 1 model + 6% | high (same arch+weights) |
| KIVI / H2O | fp16 model, int4 KV | fp16 model, fp16 KV | 1 model | N/A (KV not weight) |
| LoRA correction | base model | base + LoRA delta | 1 model + LoRA | N/A (task adaptation) |

The key advantages over split2:
- **Starting GGUF already exists**: Q8_0-Q4_K_M is a standard quantize output.  No need
  to re-encode the entire model into a new nibble format.
- **Attention tensors untouched**: Q8_0 attention stays Q8_0 in both modes.  Precision
  loss is concentrated in FFN layers, which is where Q4_K_M already places it.
- **Smaller delta for FFN tensors**: Q4_K_M error is ~3% RMS vs fp16; the int4 delta
  needs to cover a much smaller range than split2's lower nibble.

Disadvantages vs split2:
- **Verify is approximate, not exact**: the int4 delta is lossy.  Verify quality is
  "near Q8_0" rather than "exact Q8_0".
- **Delta is in float-space**: the delta quantization introduces its own error; its
  interaction with Q4_K_M's complex block structure (sub-scales, mins) is harder to
  reason about than split2's nibble math.
- **More complex kernel**: the verify kernel must load and apply the delta after the
  existing Q4_K_M dequant, adding memory accesses and compute.
- **Q4_K_M tensors in draft = 4.5 bpw, not 4.0 bpw**: the K-quant overhead (scale bytes)
  means draft bandwidth is slightly worse than split2's pure nibble draft.


## 10. Open Questions / Risks

1. **Delta error correlation**: if the int4 delta is too coarse, the remaining error after
   Q4_K_M + delta may be correlated (not random noise), causing systematic logit shifts
   similar to split2's nibble underestimation bias.  Needs empirical measurement.

2. **Accept rate vs split2**: split2 draft has exact-nibble error (always ≤ 0 per element);
   this scheme has float-domain errors with both signs.  Unknown which gives better accept
   rate in practice.

3. **FFN vs attention split**: in Qwen3-8B the attention projections are ~25% of weight
   parameters.  If they stay Q8_0 (mode-invariant), the draft speedup is bounded by the
   FFN fraction (~75% of weight bytes).  Actual decode speedup will be 1.3–1.6×, not 2×.

4. **cuBLAS verify path for prefill**: the `to_fp16_q4km_d4_verify` kernel must be fast
   enough that prefill latency does not dominate.  For prompts >512 tokens, cuBLAS
   prefill time may exceed the draft decode savings.

5. **GGUF size**: the new GGUF is ~6% larger than the Q8_0-Q4_K_M source (delta blocks
   only added to Q4_K_M tensors).  Typically: +0.3 GB on a 7–8B model.


## 11. Implementation Plan

### Phase 0 — Offline tooling (Python, no C++ changes)
- [x] `research/scripts/build_q4km_delta.py`: reads Q8_0 GGUF + Q8_0-Q4_K_M GGUF,
      computes Q4_K residual (or int4 delta), writes new GGUF with `Q4_K_RES` blocks
- [x] Validate delta magnitude / residual distribution (script validates per-tensor)

### Phase 1 — New GGML type + CPU dequant (no CUDA yet)
- [x] Add `GGML_TYPE_Q4_K_RES_DRAFT` (45) and `GGML_TYPE_Q4_K_RES` (46) in `ggml/include/ggml.h`
- [x] Block size registered in `ggml/src/ggml.c`
- [x] CPU dequant in `ggml/src/ggml-quants.c`
- [x] GGUF type registered (build_q4km_delta.py writes type 46 tensors)

### Phase 2 — CUDA MMVQ kernels
- [x] `vec_dot_q4_k_res_draft_q8_1` in `ggml/src/ggml-cuda/mmvq.cu` (reads base only)
- [x] `vec_dot_q4_k_res_verify_q8_1` in `ggml/src/ggml-cuda/mmvq.cu` (reads base + residual)
- [x] GPU convert kernels in `ggml/src/ggml-cuda/convert.cu`
- [x] Registered in `ggml/src/ggml-cuda/ggml-cuda.cu`

### Phase 3 — Runtime mode switching
- [ ] Add `q4km_verify` flag to `ggml_backend_cuda_context`
- [ ] Register `ggml_backend_cuda_set_q4km_verify` via proc-address
- [ ] Add `llama_set_weight_mode(ctx, mode)` to `include/llama.h` / `src/llama-context.cpp`
- [ ] Add `q4km_verify_captured` to `ggml_cuda_graph` for graph invalidation on mode change
- [ ] Expose via Python bindings in `llama_bindings.py`

### Phase 4 — run_adaptivegen.py integration
- [ ] Replace `llama_set_split2_mode` calls with `llama_set_weight_mode` in `_set_mode()`
- [ ] Run end-to-end benchmark: draft / verify / switch on Qwen3-8B with Q4_K_D4 GGUF
- [ ] Measure: agree_rate (bootstrap), accept_rate (rollout), tok/s for all three modes
- [ ] Compare verify quality: PPL of Q4_K_D4 verify vs plain Q8_0 (should be within 0.01)

### Phase 5 — Evaluation
- [ ] PPL on wikitext2 in draft and verify modes (vs Q8_0-Q4_K_M baseline and Q8_0)
- [ ] GSM8K accuracy in all three modes
- [ ] Adaptive gen tok/s vs split2 and vs separate-small-model speculative decoding


## 12. KV orchestration overhead (adaptive / two-context) — **optimize later**

This section is a **reminder**, not part of the weight-split design.  The base+delta scheme
shares one model and KV between draft and verify modes, but **`run_adaptivegen` switch
mode** still pays for **KV state choreography** every window.  Worth revisiting when
tuning end-to-end tok/s.

### Why rollback exists

- The **draft** autoregresses and **appends** K/V at new positions for speculative tokens.
- **Verification** may reject none, part, or all of that suffix.  Rows written for
  **rejected** tokens are **invalid** — they correspond to tokens that will not be committed.
- **`restore_kv_state`** (or equivalent) puts the cache back to the last **known-good**
  prefix (prompt + committed tokens, or partial accept).  Without that, wrong speculative
  K/V would poison later steps.

So rollback is not optional if speculative K/V is written into the same buffers you use
for committed sequence state.

### What “full snapshot” vs “only new tokens” means

- **Per decode step**, the model only **computes** new K/V rows for the current position.
- **`save_kv_state` / `restore_kv_state`** in a naive implementation often **copies the
  entire KV allocation** (all layers, current sequence length) so restore is a single
  correct memcpy.  That is **not** the same as “only the bytes for the last token” — it is
  “everything needed to rewind exactly.”
- **Possible future optimization**: snapshot/restore only the **speculative suffix**
  (e.g. positions from `n_committed` through end of draft window), if prefix is shared and
  immutable.  More bookkeeping; fewer GB copied per window if implemented carefully.

### Heavy work every window (typical cost model)

- Multiple **`save_kv_state`** (e.g. verifier + draft) and **`restore_kv_state`** around
  draft tries, verify, and partial/failed accepts.
- Each copy scales with **layers × heads × sequence length × width** — often **very**
  expensive compared to a single batched matmul.

### Commit path after acceptance

- After a window is accepted, the **verifier** must actually **contain** those tokens in
  order: often a loop of **`_single_decode`** (one step per accepted token) rather than
  one batched verify over the whole extension.
- Then **`_sync_draft_kv_from_verifier`** (or similar) **copies KV** from verifier → draft
  so the next window starts aligned.

So “commit” can add **many single-token decodes** plus **another large KV copy**, on top of
batched verify inside the window.

### TODO — revisit when optimizing adaptive throughput

- [ ] Measure whether snapshots are **full-buffer** or **suffix-only** in our bindings.
- [ ] If full-buffer, prototype **suffix-only** save/restore for speculative ranges only.
- [ ] Profile commit path: batch verifier advance vs single-decode loop; cost of draft sync.
- [ ] Consider architectures where draft **does not** commit wrong K/V into shared buffers
  until after verify (different tradeoffs: memory vs recompute).
