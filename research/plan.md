# Plan: Cluster-Only KV Cache Inference — Perplexity Study

## Goal

Measure how different KV cache compression strategies affect **perplexity**, using llama.cpp on CPU.
Memory footprint reduction is a future goal — this phase is purely about understanding quality loss.

## Approach

Use llama.cpp's public state API to intercept and modify the KV cache in Python after each decode step:

```
llama_decode() → save_state() → parse bytes → modify K/V → load_state() → llama_decode() → ...
```

This is **simulated quantization only**: the KV cache storage type stays FP16 throughout.
Values are replaced in-place with lower-precision equivalents (still stored as FP16), so
llama.cpp accepts the state without errors and the next decode uses the modified values.
All processing is in Python (numpy + ml_dtypes). No C++/CUDA changes needed.

Note: llama.cpp does support native quantized KV storage (Q2_K through Q8_0, IQ1 through IQ4)
via `type_k`/`type_v` params, but those require Flash Attention and actually compress storage.
That is out of scope for this phase — we only want to measure quality impact for now.

## KV Tensor Layout (for reference)

After parsing the state blob, each layer's K and V tensors have shape:

```
K: [n_tokens, n_embd_k]    where n_embd_k = n_heads_kv * head_dim
V: [n_tokens, n_embd_v]    where n_embd_v = n_heads_kv * head_dim
```

One row = one token position. This determines how granularities map to the data.

---

## Experiment Dimensions

### 1. Precision / Format

**Baseline is FP16 unmodified** — this is llama.cpp's default KV cache type and the standard used
by all production inference systems (vLLM, TensorRT-LLM, HuggingFace). FP32 KV cache is not used
commercially because it doubles memory with negligible quality gain. FP16 is therefore the fair
"full quality" reference point.

Each lower precision below is simulated by casting the FP16 value down and back to FP16.
The storage type stays FP16 throughout — only the precision of the represented value changes.

| Label       | Bits | Description                                       |
|-------------|------|---------------------------------------------------|
| `fp16`      | 16   | **Baseline**: no modification                     |
| `bf16`      | 16   | BF16 range/precision, stored as FP16              |
| `fp8-e4m3`  | 8    | FP8 E4M3 (higher precision, narrower range)       |
| `fp8-e5m2`  | 8    | FP8 E5M2 (wider range, lower precision)           |
| `int8`      | 8    | Uniform INT8 with scale, stored as FP16           |
| `int4`      | 4    | Uniform INT4 with scale, stored as FP16           |
| `nf4`       | 4    | NormalFloat4 (QLoRA-style), stored as FP16        |
| `int2`      | 2    | Uniform INT2 with scale, stored as FP16           |

These map directly to what llama.cpp supports natively (Q8_0 ≈ int8, Q4_0 ≈ int4, Q2_K ≈ int2),
so results here will be directly comparable if native KV quantization is tested in a later phase.

### 2. Quantization Granularity

How elements are grouped when computing the scale factor.

| Label          | Description                                           |
|----------------|-------------------------------------------------------|
| `token-wise`   | One scale per token (one row of shape `[n_embd]`)     |
| `channel-wise` | One scale per embedding dimension (one column)        |
| `group-G`      | G consecutive elements within a token share one scale |

Group sizes to sweep: G = 32, 64, 128, n_embd (= token-wise).

Channel-wise is the transpose: each of the n_embd channels gets its own scale across all tokens.

### 3. Clustering (K-means)

Apply K-means to replace KV values with k representative scalar values (cluster centers).
Cluster centers are computed per layer, then values are replaced with nearest center.

| Variable       | Values                     |
|----------------|----------------------------|
| Cluster count k| 16, 32, 64, 128, 256       |
| Granularity    | per-layer, per-head, per-token |

### 4. Prefill Strategy

| Label      | Description                                                    |
|------------|----------------------------------------------------------------|
| `batch`    | Batch prefill — quantize after prefill, decode uses quant KV  |
| `token-by-token` | One token at a time — prefill also attends to quant KV  |

### 5. Clustering + Quantization Order

When both clustering and quantization are applied:

| Label                  | Description                                       |
|------------------------|---------------------------------------------------|
| `cluster → quantize`   | K-means on FP16 values, then quantize centers     |
| `quantize → cluster`   | Quantize to FP8/INT8 first, then cluster          |

---

## Experiment Matrix (Phase 1)

Start simple, expand outward.

### Step 1: Precision sweep (no clustering)
Fix granularity = token-wise, group size = 128. Sweep precision from high to low.

| Exp | Precision  | Bits | Granularity | Group |
|-----|------------|------|-------------|-------|
| 1a  | fp16       | 16   | —           | —     | ← unmodified baseline
| 1b  | bf16       | 16   | token-wise  | 128   |
| 1c  | fp8-e4m3   | 8    | token-wise  | 128   |
| 1d  | fp8-e5m2   | 8    | token-wise  | 128   |
| 1e  | int8       | 8    | token-wise  | 128   |
| 1f  | int4       | 4    | token-wise  | 128   |
| 1g  | nf4        | 4    | token-wise  | 128   |
| 1h  | int2       | 2    | token-wise  | 128   |

### Step 2: Granularity sweep (fix precision = fp8-e4m3)

| Exp | Precision | Granularity   | Group         |
|-----|-----------|---------------|---------------|
| 2a  | fp8-e4m3  | token-wise    | 128           |
| 2b  | fp8-e4m3  | token-wise    | 64            |
| 2c  | fp8-e4m3  | token-wise    | 32            |
| 2d  | fp8-e4m3  | token-wise    | n_embd (full) |
| 2e  | fp8-e4m3  | channel-wise  | —             |

### Step 3: Clustering sweep (no quantization)

| Exp | Method       | k   | Granularity |
|-----|--------------|-----|-------------|
| 3a  | cluster only | 16  | per-layer   |
| 3b  | cluster only | 32  | per-layer   |
| 3c  | cluster only | 64  | per-layer   |
| 3d  | cluster only | 128 | per-layer   |
| 3e  | cluster only | 256 | per-layer   |
| 3f  | cluster only | 64  | per-head    |

### Step 4: Cluster + Quantize order (best settings from Steps 2 & 3)

| Exp | Order                | Precision | k   |
|-----|----------------------|-----------|-----|
| 4a  | cluster → quantize   | fp8-e4m3  | 64  |
| 4b  | quantize → cluster   | fp8-e4m3  | 64  |

---

## Metrics

Single metric for this phase: **perplexity** on WikiText-103 test set.

- Lower perplexity = better quality
- Baseline is unmodified FP16 KV cache (no `save_state`/`load_state` call)
- Report absolute perplexity and delta: `Δppl = ppl(method) - ppl(fp16_baseline)`
- Also record: wall-clock time per experiment (to understand overhead of the Python hook)

---

## Implementation Steps

### Step 1: Environment setup
- Install `llama-cpp-python` (CPU build)
- Install `ml_dtypes`, `numpy`, `scikit-learn`
- Download a small model for fast iteration (e.g. Llama-3.2-1B-Instruct GGUF)
- Download WikiText-103 test set

### Step 2: State parser
Write `research/poc/parse_state.py`:
- Given `save_state()` bytes, parse the binary format
- Return per-layer K and V as numpy float16 arrays
- Write back: pack modified arrays back into the blob, call `load_state()`

### Step 3: Quantization functions
Write `research/poc/quant.py`:
- `simulate_fp8(arr, format, group_size, granularity)` → FP16 array
- `simulate_int8(arr, group_size, granularity)` → FP16 array
- Handles token-wise, channel-wise, group-G granularities

### Step 4: Clustering functions
Write `research/poc/cluster.py`:
- `cluster_kv(arr, k, granularity)` → FP16 array with values replaced by cluster centers
- Uses `sklearn.cluster.MiniBatchKMeans` for speed

### Step 5: Perplexity harness
Write `research/poc/perplexity.py`:
- Load model with `llama-cpp-python`
- Tokenize WikiText-103 test set
- Decode token by token, apply KV modification hook after each step
- Compute perplexity: `exp(-mean(log_probs))`

### Step 6: Run experiments and log results
Write `research/poc/run_experiments.py`:
- Iterate over experiment matrix
- Save results to `research/results/results.csv`

---

## File Layout

```
research/
  plan.md                          ← this file
  research_idea_cluster_kv_cache_inference.md
  poc/
    parse_state.py                 ← state blob parser
    quant.py                       ← FP8/INT8 simulation
    cluster.py                     ← K-means on KV
    perplexity.py                  ← perplexity measurement loop
    run_experiments.py             ← experiment runner
    requirements.txt
  results/
    results.csv
```

---

## Known Limitations

### Prefill quantization strategy

`llama_decode()` for the prefill stage computes attention internally before returning, so our hook
can only run after the fact. This gives two strategies with different tradeoffs:

**Strategy A — Batch prefill (default, faster):**
Process all prompt tokens in one `llama_decode()` call. Quantize the full KV cache afterward.
Prefill attention uses FP16 K/V. Only decode-stage attention uses quantized K/V.

```
decode([t0..tN])  →  quantize KV(t0..tN)  →  decode(tN+1) uses quantized K/V  →  ...
```

**Strategy B — Token-by-token prefill (slower, fully quantized):**
Process the prompt one token at a time. Quantize after each token so that every subsequent
token attends to quantized K/V, including during prefill itself. Matches what native
`type_k`/`type_v` quantization does.

```
decode([t0])  →  quantize KV(t0)
decode([t1])  →  attends to quantized K/V(t0)  →  quantize KV(t0, t1)
decode([t2])  →  attends to quantized K/V(t0, t1)  →  quantize KV(t0, t1, t2)
...
```

Strategy B is N times slower during prefill (N separate decode calls instead of one batched call),
but gives a more faithful simulation of real quantized inference. Both strategies will be run and
compared. Strategy A results will be noted as slightly optimistic.

### Re-quantization on every decode step

Each `save_state()`/`load_state()` call modifies the entire KV cache, including tokens that were
already quantized in previous steps. For fixed-precision quantization (FP8, INT8) this is
idempotent — a value already at FP8 precision rounds to the same value again. For clustering,
re-running K-means on already-clustered values is also stable once converged. No meaningful
error accumulation is expected, but this should be verified empirically.

### Performance overhead

`save_state()`/`load_state()` serializes and deserializes the full KV state on every decode step.
This adds significant Python-side overhead and makes token generation much slower than normal.
This is acceptable for a perplexity study but not representative of real inference throughput.

---

## What is NOT in scope for this phase

- Actual memory footprint reduction (values are still stored as FP16)
- GPU / CUDA changes
- C++ modifications to llama.cpp internals
- Throughput optimization

These are future phases once the quality impact is understood.
