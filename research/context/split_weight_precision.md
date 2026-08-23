# Split-Weight Precision: In-Place Draft/Verify from a Single Q8_0 Model

## 0. Implementation Status (as of 2026-04-28)

**This document describes the raw nibble-split design (Region A / Region B of Q8_0 int8).**
The actual implementation that landed uses `Q4_K_RES` instead (see `base_delta_weight_split.md`).

| Layer | Status |
|-------|--------|
| `GGML_TYPE_Q4_K_RES` (type 46, verify) defined in `ggml.h` | **DONE** |
| `GGML_TYPE_Q4_K_RES_DRAFT` (type 45, draft) defined in `ggml.h` | **DONE** |
| CUDA MMVQ kernels in `ggml/src/ggml-cuda/mmvq.cu` | **DONE** |
| GPU convert kernels in `ggml/src/ggml-cuda/convert.cu` | **DONE** |
| Offline GGUF builder (`build_q4km_delta.py --delta-format q4k --out-layout q4_k_res`) | **DONE** |
| Built GGUF: `models/Qwen3-8B-Q8_0-Q4_K_M_q4k_res.gguf` | **DONE** |
| Runtime mode switching (tensor type 46↔45 API, `llama_set_weight_mode`) | **PENDING** |
| `run_adaptivegen.py` integration | **PENDING** |

**Format difference from this design doc:**
- This doc: raw nibble split — `Region A = (q8_int >> 4) & 0xF`, `Region B = q8_int & 0xF`
- Implemented: `Q4_K_RES` = base Q4_K_M block (144 B) + residual Q4_K block (144 B) = 288 B/superblock
- Both achieve 4.5 bpw draft / ~9 bpw verify, but Q4_K_RES is lossy (residual is re-quantized); raw nibble split gives exact Q8_0 reconstruction.

**Remaining work:** The raw nibble split (this doc) is still a design — not yet implemented.
The Q4_K_RES path is the working implementation. The only missing piece is the runtime switching mechanism to flip between type 45 (draft) and type 46 (verify) without reloading the model.

---

## 1. Motivation

Current adaptive-gen uses a separate smaller GGUF model (e.g. Q3_K_M or Q2_K) as the
draft model and the full Q8_0 as the verifier.  This requires:
- Loading and holding two separate models in VRAM
- A model-size mismatch between draft and verifier (different architecture depth/width
  is not possible; only weight precision differs)
- Separate KV caches

The key observation: **a Q8_0 weight already contains a Q4-quality approximation in its
upper 4 bits**.  If we split each int8 element into two nibble regions stored
contiguously in memory, the GPU can load only the upper-nibble region during drafting
(half the weight bytes) and load both regions during verification (full Q8_0 precision),
all from the same allocated buffer, with the same model architecture and shared KV cache.

This eliminates the second model entirely and gives a clean progressive-precision ladder:

| Pass | Regions read | Effective precision | Weight BW |
|------|-------------|---------------------|-----------|
| Draft (fast)   | A only    | Q4-equivalent | 50% of Q8_0 |
| Verify (exact) | A + B     | Full Q8_0     | 100%       |

Extended to 3 regions (see §3), we get Q2 / Q4 / Q8 progressive levels.


## 2. Background: Q8_0 format

llama.cpp Q8_0 stores weights in blocks of 32 elements:

```
struct block_q8_0 {
    ggml_half scale;      // fp16 block scale
    int8_t    qs[32];     // 32 quantized values: v_i = scale × qs_i
};
// Size: 2 + 32 = 34 bytes per block
```

Each `qs[i]` is a signed int8.  In binary:

```
qs[i] = [b7 b6 b5 b4 b3 b2 b1 b0]   (b7 = MSB, b0 = LSB)
```


## 3. Split-weight storage format

### 3a. 2-Region split  (Q8_0 → Q4 draft / Q8_0 verify)

Split each int8 element into two nibbles stored in separate contiguous regions:

```
Region A  (upper nibble):  [b7 b6 b5 b4]   4 bits per element
Region B  (lower nibble):  [b3 b2 b1 b0]   4 bits per element
```

**Block layout** (32 elements per block):

```
[ scale: fp16 (2 B) ]
[ draft_scale: fp16 (2 B) ]        ← scale × 16, precomputed at transform time
[ Region A: 16 bytes ]             ← upper nibbles, 2 per byte, packed
[ Region B: 16 bytes ]             ← lower nibbles, 2 per byte, packed
Total: 36 bytes/block  (vs 34 for Q8_0 — 6% overhead)
```

**Reconstruction:**
- Draft:  `v_i = draft_scale × nibble_A_i`   where `nibble_A_i ∈ [0,15]` (treat as unsigned 4-bit → remap to signed: subtract 8 before multiply, OR keep as signed by sign-extending bit b7 into 4-bit field)
- Verify: `qs_i = (nibble_A_i << 4) | nibble_B_i`,  `v_i = scale × qs_i`

> **Scale relationship:**  `draft_scale = scale × 16` because dropping the lower nibble
> means `qs ≈ nibble_A << 4`, so `v = scale × (nibble_A << 4) = (scale × 16) × nibble_A`.

**Sign handling:**  Q8_0 int8 values are signed (range −128…127).  The upper nibble
carries the sign bit (b7).  When unpacking nibble_A for the draft pass:
- sign-extend bit 3 of the nibble into a signed int8 before multiplying by draft_scale
- i.e. `signed_nibble = (nibble_A ^ 8) - 8`  (convert 4-bit unsigned to signed)

### 3b. 3-Region split  (Q8_0 → Q2 / Q4 / Q8 progressive)

```
Region A:  [b7 b6]           2 bits per element  →  8 bytes per 32-elem block
Region B:  [b5 b4]           2 bits per element  →  8 bytes per 32-elem block
Region C:  [b3 b2 b1 b0]     4 bits per element  → 16 bytes per 32-elem block
```

**Block layout:**

```
[ scale: fp16 (2 B) ]
[ scale_AB: fp16 (2 B) ]     ← scale × 16 (for A+B = 4-bit draft)
[ scale_A:  fp16 (2 B) ]     ← scale × 64 (for A-only = 2-bit draft)
[ Region A:  8 bytes ]
[ Region B:  8 bytes ]
[ Region C: 16 bytes ]
Total: 38 bytes/block  (12% overhead vs Q8_0)
```

**Precision levels:**

| Level | Regions | Bytes read/block | Effective | Scale |
|-------|---------|-----------------|-----------|-------|
| 1 (Q2) | A only  | 8+overhead | 2-bit | `scale × 64` |
| 2 (Q4) | A + B   | 16+overhead | 4-bit | `scale × 16` |
| 3 (Q8) | A+B+C   | 32+overhead | full 8-bit | `scale` |

**Reconstruction at level 2:**
`nibble = (A_2bit << 2) | B_2bit`,  then sign-extend 4-bit → signed,  `v = scale_AB × nibble`

**Reconstruction at level 3:**
`qs = (A_2bit << 6) | (B_2bit << 4) | C_4bit`,  `v = scale × qs`

### 3c. Which tensors to split

Split all weight matrices in the transformer layers:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `ffn_gate`, `ffn_up`, `ffn_down`

Do **not** split:
- Layer norm / RMSNorm gamma vectors (fp16, small, always loaded)
- Token embedding table (accessed sparsely, not GEMV bottleneck)
- LM head (only used at the last layer during verify)

At load time, a single pass over all Q8_0 blocks transforms them in-place (or into a
new allocation) into the split layout.  The original Q8_0 GGUF data is discarded.


## 4. CUDA kernels

### 4a. 2-region kernels

```
// Draft: load Region A only → dequant to fp16 → GEMV
__global__ void dequant_q8_split2_draft_fp16(
    const uint8_t* __restrict__ region_a,   // packed upper nibbles
    const ggml_half* __restrict__ dscale,   // draft_scale per block
    ggml_half* __restrict__ out,            // dequantized fp16 output
    int n_blocks
);

// Verify: load Region A + B → reconstruct int8 → dequant → GEMV
__global__ void dequant_q8_split2_verify_fp16(
    const uint8_t* __restrict__ region_a,
    const uint8_t* __restrict__ region_b,
    const ggml_half* __restrict__ scale,
    ggml_half* __restrict__ out,
    int n_blocks
);
```

Both feed into the existing GEMV/GEMM infrastructure (or a fused dequant-GEMV kernel
to avoid materializing the fp16 weight matrix in HBM).

### 4b. 3-region kernels

Three variants: `dequant_q8_split3_level1`, `_level2`, `_level3`.  Level 3 is
functionally identical to standard Q8_0 dequant.

### 4b-note. Dequant overhead and realistic speedup

An important empirical lesson from our measurements: **Q2_K is NOT faster than Q4_K_M**
despite reading fewer bytes from HBM.  The dequant overhead in Q2_K dominates the BW
savings.  Q2_K's dequant is expensive because:

- Two-level scaling (sub-block scale × super-block scale) — two multiplies + a load per
  element
- Non-power-of-2 bit packing (6-bit fields) — non-trivial extraction
- Larger 256-element blocks — worse register pressure and warp divergence
- Under-optimized kernels (rarely used in practice → less tuning)

Our nibble-split draft kernel is structurally simpler than Q2_K:

```
Q2_K per element:   load 2-bit val  +  unpack 6-bit sub_scale  +  unpack 6-bit super_scale
                    + 2 multiplies  + 1 add  = expensive

Region A per element: load nibble (shift/mask, 1–2 ops)
                    + sign-extend (2 ops)
                    + 1 multiply  = cheap
```

However, the nibble unpack still adds ~3 integer ops per element vs Q8_0's direct int8
load.  At B=1 (memory-bound) these ops are hidden behind HBM latency — the GPU has spare
compute cycles while waiting for data.  But secondary effects (warp efficiency, SM
occupancy, instruction scheduling) mean the real speedup will be less than the theoretical
2× from halving the byte count.

**Realistic speedup estimate: 1.5–1.8×** rather than 2.0×, depending on how well the
nibble kernel is optimized for coalesced access and instruction-level parallelism.

> **Early validation required (see §11):** Before building the full adaptive gen
> integration, write a standalone microbenchmark that measures the actual achieved HBM
> bandwidth of the nibble-draft kernel vs the Q8_0 kernel on the target GPU.  Only
> proceed if the nibble kernel achieves ≥ 80% of Q8_0's bandwidth efficiency (i.e.
> ≥ 0.80 × 0.44 = 0.35 of peak BW → effective speedup ≥ 1.6×).

### 4c. Kernel selection at runtime

A per-context `precision_mode` enum controls which kernel is dispatched:

```c
typedef enum {
    SPLIT_PRECISION_FULL  = 0,  // all regions (verify / baseline)
    SPLIT_PRECISION_HALF  = 1,  // Region A only (2-region draft)
    SPLIT_PRECISION_Q2    = 2,  // 3-region: A only
    SPLIT_PRECISION_Q4    = 3,  // 3-region: A+B
} split_precision_mode_t;
```

The GGML graph builder (or a wrapper around `llama_decode`) checks this flag and selects
the appropriate kernel for each weight tensor.


## 5. In-memory model transformation

At model load time (after `llama_load_model_from_file`), before the first decode:

```
for each transformer layer:
    for each weight tensor (q/k/v/o/gate/up/down):
        assert tensor->type == GGML_TYPE_Q8_0
        allocate split_buffer[n_blocks × block_size_split]
        for each block b:
            scale[b]       = original_block[b].scale
            draft_scale[b] = scale[b] × 16.0f
            for each element i in block:
                nibble_A = (qs[i] >> 4) & 0x0F
                nibble_B =  qs[i]       & 0x0F
                pack nibble_A into region_a[b]
                pack nibble_B into region_b[b]
        replace tensor->data pointer with split_buffer
        set tensor->type = GGML_TYPE_Q8_0_SPLIT2  (or raw, TBD)
```

This runs entirely on CPU at startup; no GPU copy until the first decode.


## 6. Bootstrap phase

The bootstrap window runs full-precision (`SPLIT_PRECISION_FULL`) for N tokens
(typically 16–64) to establish ground truth.  At the **end** of the bootstrap window:

```
Step 1 — Full precision decode (sequential, N tokens):
    for t in 1..N:
        run forward pass at SPLIT_PRECISION_FULL
        sample token t, record logits[t], append to KV cache

Step 2 — Draft-precision verification (one batched pass):
    feed all N generated tokens as a batch (same as speculative verify step)
    run forward pass at SPLIT_PRECISION_HALF (or SPLIT_PRECISION_Q4)
    for each position i:
        agree[i] = (argmax(draft_logits[i]) == token[i+1])
    agree_rate = mean(agree)

    optionally also run SPLIT_PRECISION_Q2 batch pass and record agree_rate_Q2

Step 3 — Candidate selection:
    use same cost/agree_rate ranking as adaptive_gen bootstrap
    cost from bootstrap_ms_quant_h100_layerwise.json (A-only is ~50% BW of Q8)
    pick precision level that maximises agree_rate / cost
```

The two batch passes in Step 2 are cheap: weight bytes read once for all N tokens
(amortised), KV bytes = N × ctx × kv_bpt (same as a verify step in speculative decoding).


## 7. Rolling window (adaptive gen) phase

After bootstrap, the selected draft precision drives the rolling window:

```
loop:
    Draft K tokens at precision P:
        for t in 1..K:
            run forward pass at precision P (A-only or A+B)
            sample token t

    Verify K tokens (one batched pass at SPLIT_PRECISION_FULL):
        feed [last_accepted, draft_1, ..., draft_K] as batch
        for each position i:
            if full_argmax[i] != draft_token[i]:
                reject at position i, re-sample from full logits[i]
                break

    n_accepted = number of accepted draft tokens (0..K)
    append n_accepted+1 tokens to output (including bonus token from verify)

    Precision adaptation (every M windows):
        if agree_rate < low_threshold:  upgrade P (e.g. HALF → FULL)
        if agree_rate > high_threshold: try downgrade P (FULL → HALF)
```

The KV cache is **shared** between draft and verify passes — same model architecture,
same KV head count, same head dim.  No copy or reconciliation needed.


## 8. Key differences from current adaptive-gen script

| Aspect | Current script (Python ctypes) | This design (C++/CUDA) |
|--------|-------------------------------|------------------------|
| Draft model | Separate GGUF (Q3_K_M, Q2_K) | Same model, Region A of split buffer |
| VRAM | 2× model | 1× model + 6-12% overhead |
| KV cache | Shared (same arch) | Shared (identical) |
| Draft BW | Full weight read of small model | 50% weight read of full model |
| Bootstrap verify | Run draft model on N tokens | Batch pass at reduced precision |
| Precision ladder | Discrete separate models | Continuous via region selection |


## 9. Integration points in llama.cpp

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Add `GGML_TYPE_Q8_0_SPLIT2`, `_SPLIT3` (or defer, use raw) |
| `ggml/src/ggml-cuda/dequant.cu` | New draft/verify dequant kernels |
| `ggml/src/ggml-cuda/mmq.cu` | Fused dequant-GEMV variants for split types |
| `src/llama-model.cpp` | `split_weights_inplace()` called after load |
| `src/llama-context.cpp` | `precision_mode` field, passed into graph build |
| `src/llama-graph.cpp` | Kernel dispatch conditioned on `precision_mode` |
| `include/llama.h` | `llama_set_precision_mode(ctx, mode)` public API |
| New: `src/llama-split-weights.cpp` | Transform logic, buffer management |
| New: `src/llama-adaptive-gen.cpp` | Bootstrap + rolling window loop |


## 11. Early validation: nibble kernel microbenchmark

Before integrating into the adaptive gen loop, validate the core assumption that
nibble-region loading actually delivers meaningful speedup on real hardware.

**What to build:**
A standalone CUDA benchmark (`research/scripts/bench_nibble_gemv.cu`) that:
1. Allocates a Q8_0-shaped weight matrix and transforms it into split format
2. Runs 1000 iterations of GEMV at Q8_0 precision → measures achieved GB/s
3. Runs 1000 iterations of GEMV using Region A nibbles only → measures achieved GB/s
4. Reports: bytes loaded, time, GB/s, ratio vs Q8_0, ratio vs peak HBM

**Pass criteria:**
- Region A GEMV achieves ≥ 80% of Q8_0 GEMV bandwidth efficiency
- i.e. if Q8_0 kernel achieves 44% of 3350 GB/s = 1474 GB/s, Region A should achieve
  ≥ 0.80 × 1474 = 1179 GB/s on half the data → effective 1.6× wall-clock speedup
- If Region A BW efficiency < 70% of Q8_0, redesign the packing layout before proceeding

**What might cause failure:**
- Nibble unpack disturbs memory coalescing (fix: ensure packed bytes are 128-byte aligned)
- Extra instructions reduce SM occupancy (fix: tune block/thread dimensions)
- Scale load latency not hidden (fix: prefetch scale into shared memory)

This microbenchmark is the **go/no-go gate** for the full implementation.


## 12. Further work (deferred)

### GGUF serialization
Store split-weight tensors natively in GGUF so the startup transformation is not needed.
Requires registering new tensor types in the GGUF spec.

### Hardware-native low-precision compute
When hardware supports native INT4/INT2 GEMMs (e.g. via CUTLASS INT4 GEMM on H100),
the draft pass compute time drops proportionally, not just the memory bandwidth.
Currently we assume compute ceiling stays at INT8 TFLOPS (memory-bound anyway for
typical decode batch sizes).

### Per-layer precision selection
Some layers (first and last few transformer layers, attention projections) may need
higher precision than FFN layers.  The `precision_mode` could become a per-layer
array rather than a single global flag.

### KV cache split
Combine with split-weight draft/verify: during draft, use quantized KV (int4_ch);
during verify, use fp16 KV.  The two passes share the same KV buffer, with the verify
pass re-reading at full precision.  This halves KV bandwidth during drafting as well.

### Acceptance rate model
The bootstrap agree_rate for split-weight draft vs full-model draft differs because:
- Split draft uses same architecture (no model-size mismatch)
- Error comes purely from weight precision loss, not capacity
Expected: higher agree_rate at given bpw vs a smaller separate model.

### 3-region progressive fallback
Currently the 3-region ladder is described but only partially designed.
Full implementation requires the `SPLIT_PRECISION_Q2` kernel and a mechanism to
fall back mid-window when Q2 agree_rate is too low (without restarting the window).

### Multi-GPU
With tensor parallelism each GPU holds a shard of the split buffer.
Region A and Region B shards stay co-located; no additional inter-GPU traffic.
