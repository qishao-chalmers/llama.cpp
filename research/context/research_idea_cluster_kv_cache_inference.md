# Cluster-Only KV Cache Inference with Speculative Decoding for LLMs

## 1. Project Overview

We propose a **cluster-only KV cache inference framework** for large language models (LLMs) to reduce memory footprint and bandwidth while maintaining output quality.

The core idea is:

1. Start from **already quantized weights** (e.g., FP8) and KV cache (FP16 or FP8).
2. Replace KV cache values with a **small set of representative numbers** obtained via **clustering**.
3. Optionally implement **speculative decoding**:
   - Run inference with cluster-only KV cache.
   - Check outputs for correctness or regularity.
   - If necessary, fallback to full precision KV cache.

The goal is to **minimize memory usage and computation** while keeping **perplexity and outputs close to baseline**.

## 2. Motivation

- LLM KV cache often dominates **memory footprint**, especially for long sequences and large models (7B, 13B).
- Traditional low-bit quantization (INT8, FP8) reduces memory but may still store redundant numbers.
- By using **cluster-only representation**, we reduce the number of unique KV values, potentially compressing memory and cache bandwidth.
- Speculative decoding allows us to **avoid reconstructing full precision weights/KV cache** unless needed.

## 3. Proposed Method

### Step 1: Baseline Measurements
- Run model with:
  - **Weights:** FP8  
  - **KV cache:** FP16  
- Record baseline performance (perplexity, memory usage, throughput).

### Step 2: Low-Precision KV Cache
- Run model with:
  - **Weights:** FP8  
  - **KV cache:** INT8 or FP8  
- Compare performance with baseline to understand the effect of low-precision KV cache.

### Step 3: Cluster-Only KV Cache
- After computing each KV cache:
  1. Flatten KV cache tensor.  
  2. Apply **K-means clustering** to extract `k` representative values.  
  3. Replace KV cache entries with **nearest cluster center**.  
  4. Use the cluster-only KV cache in the **next token computation**.  
- Measure **perplexity, memory savings, and error propagation**.

### Optional Step 4: Speculative Decoding
- Use cluster-only KV cache to generate **draft outputs**.  
- Validate outputs using:
  - Top-k token probabilities, entropy checks, or output regularity.  
- If validation fails, **fallback to full precision KV cache** for accurate computation.

## 4. Experimental Variables

| Variable | Values / Options |
|----------|-----------------|
| Cluster size `k` | 16, 32, 64, 128, 256 |
| KV cache precision | FP16, FP8, INT8 |
| Weight precision | FP8 |
| Speculative decoding threshold | Top-k probability, entropy-based, custom regularity metric |
| Evaluation metrics | Perplexity, token-level accuracy, memory footprint, throughput, fallback frequency |

## 5. Framework Recommendation

- **Primary:** vLLM + PyTorch
  - Full Python access to KV cache.
  - Easy on-the-fly manipulation of KV cache for clustering and quantization.
  - GPU / CPU acceleration.
- **Secondary (for benchmarking / deployment):** llama.cpp
  - Memory-optimized, C++ inference.
  - Useful for hardware-oriented memory & latency evaluation.

## 6. Implementation Notes for Claude Code

- Intercept KV cache per forward step using hooks (vLLM) or internal buffer (llama.cpp).  
- Apply clustering (e.g., K-means) per layer per token.  
- Replace KV cache values with cluster centers before next step.  
- Optionally, implement speculative decoding fallback logic.  
- Log metrics:
  - Memory footprint reduction (%)
  - Perplexity / output difference vs baseline
  - Cluster center usage statistics
  - Fallback frequency (if speculative decoding used)

## 7. Hardware / Conference Angle

- Target conferences: **MICRO, HPCA, ISCA**.  
- Focus:
  - Memory footprint reduction and bandwidth savings.
  - Minimal accuracy loss or controlled fallback.
  - Pipeline / accelerator-friendly design: can be integrated with GPU / CPU inference kernels.  
- Emphasize **cluster-only KV cache** as a hardware-efficient approximation strategy.

## 8. Future Extensions

- Delta / residual encoding on cluster values for **lossless reconstruction**.  
- Adaptive cluster size per layer or per token.  
- Integration with **speculative decoding thresholds** for throughput optimization.  
- Hardware simulation using **Accel-sim or similar framewo