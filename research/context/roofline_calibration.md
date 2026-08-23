# Roofline vs Measured Timing: Gap Analysis

**Model:** Qwen3-8B Q8_0 weights, f16 KV cache  
**Hardware:** H100 SXM (3350 GB/s HBM, 989 TFLOPS FP16)  
**Data source:** `research/results/qwen3-8b/profile/kv_timing_h100.json` (B=1,4,8,16,32)

---

## Measured vs Roofline (ms/tok)

| B | Measured | Roofline | Ratio |
|---|----------|----------|-------|
| 1 | 7.599 | 2.593 | 2.93× |
| 4 | 2.783 | 0.808 | 3.44× |
| 8 | 1.786 | 0.510 | 3.50× |
| 16 | 1.143 | 0.362 | 3.16× |
| 32 | 0.816 | 0.287 | 2.84× |

The roofline underpredicts by 2.8–3.5×. The gap has two independent components.

---

## Component 1: T_floor (additive, per-step)

**Value:** ~2.2 ms per decode step, independent of batch size and model size.

**Cause:** CPU-GPU pipeline bubble between every decode step.

Even with CUDA Graphs active (llama.cpp enables them by default on Ampere+, confirmed in
`ggml/src/ggml-cuda/common.cuh:1120`), each step requires a full CPU round-trip:

```
GPU:  [===== graph replay =====]
CPU:                             cudaStreamSynchronize
                                 → read logits
                                 → sample (argmax over 150K vocab)
                                 → rebuild batch
                                 → cudaGraphExecUpdate  (new token, new KV write pos)
                                 → cudaGraphLaunch
GPU:                                                    [===== next graph replay =====]
                                 └──────────── T_floor ─────────────────────────────┘
```

Breakdown:
- `cudaStreamSynchronize` wakeup latency (OS scheduler): 0.5–1.5 ms
- `cudaGraphExecUpdate` (graph node params for new token/position): 0.3–0.5 ms
- Logit copy + batch rebuild + `cudaGraphLaunch`: ~0.2 ms

**Cannot be avoided** with the current `llama_decode` API — it always synchronizes
before returning so logits are CPU-readable. Production serving systems (vLLM,
TensorRT-LLM) pipeline this with async sampling to hide it; llama.cpp does not.

**CUDA Graphs do NOT eliminate T_floor.** They eliminate per-kernel launch overhead
(~15 µs × 360 kernels = ~5 ms without graphs), but the CPU-GPU barrier between steps
remains. Our earlier hypothesis that kernel launch overhead was T_floor was incorrect.

---

## Component 2: BW efficiency lower than assumed (multiplicative)

**Roofline assumption:** 70% of H100 HBM peak = 2345 GB/s effective.  
**Reality:** weight GEMV ~40%, KV flash-attn ~32%.

### Why the roofline assumption is too optimistic

**Weight reads (GEMV regime):**  
For Q8_0 with batch B, arithmetic intensity = `2B` FLOPs/byte. The H100 ridge point
is ~295,000 FLOPs/byte (989 TFLOPS / 3.35 TB/s). At B=32, AI = 64 — we are
**4,600× below the compute-bound threshold**. Every realistic serving batch is
memory-bound for weight reads. Weights never transition to GEMM territory during
token generation.

Actual weight BW efficiency: ~40% of peak (1340 GB/s). GEMV on H100 does not
achieve peak BW due to irregular access patterns and low SM occupancy at small B.

**KV attention (flash attention):**  
Flash attention uses a tile-based algorithm with online softmax reduction. Even when
memory-bound, the complex control flow (masking, normalization across tiles) reduces
effective BW to ~32% of peak (1064 GB/s).

### Effect grows with batch size

As B increases, KV traffic grows linearly (604 MB per sequence at ctx=4096) while
weight traffic is fixed (8.5 GB for Q8_0 Qwen3-8B):

| B | Weight fraction | KV fraction | Overall eff BW |
|---|---|---|---|
| 1 | 93% | 7% | 1689 GB/s (50%) |
| 4 | 78% | 22% | 1223 GB/s (37%) |
| 8 | 64% | 36% | 1104 GB/s (33%) |
| 16 | 47% | 53% | 1130 GB/s (34%) |
| 32 | 30% | 70% | 1164 GB/s (35%) |

**Overall BW efficiency drops from B=1 to B=8, then plateaus at ~33–35%.** It does
not recover at higher B because the low-efficiency flash-attention component dominates
an ever-growing share of traffic. The GEMV→GEMM efficiency improvement intuition does
not apply here.

---

## Two-parameter calibration model

```
ms_per_step = T_floor + scale × roofline_step_ms
```

Fitted values (Q8_0 Qwen3-8B, H100):

| Parameter | Value | Meaning |
|---|---|---|
| T_floor | 2.21 ms | CPU-GPU pipeline overhead per step |
| scale | 2.67 | BW efficiency correction (roofline × scale = reality) |

Fit quality: ±8% for B ≥ 4. B=1 overpredicts by ~20% (T_floor may be slightly
smaller at B=1 due to simpler graph updates for single-sequence decode).

Equivalent: `ms/tok = T_floor/B + scale × roofline_ms/tok`

At large B, T_floor/B → 0 and the multiplicative scale dominates. At B=1, T_floor
dominates (2.21 ms vs ~1 ms GPU time for small weight quant).

---

## Reporting units: ms/tok vs tok/s (nonlinear)

Many tables in this repo use **ms/tok** because it adds/subtracts linearly. If you
convert to throughput:

\[
\text{tok/s} = \frac{1000}{\text{ms/tok}}
\]

then **errors are no longer linear**. When batch increases, `ms/tok` often becomes
small, so the same absolute error (e.g. +0.1 ms/tok) becomes a **large tok/s delta**.
This can make “discrepancy vs batch” *look* worse in tok/s even if ms/tok residuals
are flat.

Practical rule:
- If you are debugging model fit quality, prefer **ms/tok** residuals.
- If you are reporting serving throughput, use **tok/s**, but expect larger-looking
  deltas at high batch.

---

## KV quant overhead (native q8_0 / q4_0 vs f16)

Native KV quants are **slower** than f16 despite fewer bytes, because the
flash-attention kernel must dequantize compressed KV tiles before use. The overhead is
per-kernel-call (fixed), not per-byte — so q4_0 ≈ q8_0:

| B | q8_0 overhead | q4_0 overhead |
|---|---|---|
| 1 | +8% | +8% |
| 8 | +13% | +12% |
| 32 | +22% | +23% |

Overhead grows with B because KV traffic grows. Applied correction in
`roofline_layer.py`: `KV_ATTN_OVERHEAD = {fp16: 1.00, int8_ch: 1.08, int4_ch: 1.08}`.

Soft quants (int2_ch, int3_ch via Python CPU hook) restore fp16 to GPU before each
decode step — GPU always reads fp16, so their overhead factor is 1.00.

---

## What to do with this

**For absolute ms/tok prediction**, use the two-parameter model:
```python
ms_per_tok = (2.21 + 2.67 * roofline_step_ms) / batch_size
```

**For relative speedup between quant types** (e.g., will q4_0 KV be faster?),
T_floor and scale cancel in the ratio — the roofline ratio is accurate enough.

**To improve calibration further:**
- Fit `(T_floor, scale)` per model × weight_tag (currently global)
- Collect B=64, 128 to confirm the plateau persists
- Use `nsys profile` to measure pure GPU kernel time and verify T_floor estimate

---

## Batch-size discrepancy: why it can grow at large B

Even if the roofline + calibration matches well at small batch, larger batches can
diverge because **GPU kernel efficiency is batch-regime dependent**:

- **GEMM / dequant kernels**: occupancy, tile shapes, and memory coalescing change
  with `B`. A single constant `η` (or a single global scale) cannot capture all
  regimes.
- **Attention**: flash-attn style kernels have batch-dependent scheduling and
  reduction behavior; an “\(1/B\)” intuition does not always apply.

In the layerwise simulator, the recommended way to separate these effects is:
- keep **per-family efficiencies** (`η` by op family), and
- use `sim_physics` knobs for second-order effects (e.g. attention scaling, weight-tag
  scaling), and
- only then apply a small `(t_floor, scale)` calibration layer.
