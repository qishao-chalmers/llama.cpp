# Experiment Results

## Precision Sweep — Qwen3-8B-Q8_0, Strategy B, 20 chunks × 128 tokens
File: `research/poc/results_128ctx_20chunks.json`

| Format   | PPL    | vs FP16  | Notes |
|----------|--------|----------|-------|
| fp16     | 20.63  | baseline | |
| bf16     | 20.65  | +0.07%   | safe |
| fp8_e4m3 | 20.77  | +0.65%   | safe |
| fp8_e5m2 | 20.92  | +1.40%   | safe |
| int8     | 20.79  | +0.75%   | safe |
| int8_ch  | 20.63  | -0.03%   | essentially lossless |
| int4     | 213.87 | +937%    | catastrophic |
| int4_ch  | 21.26  | +3.04%   | surprisingly good |
| nf4      | 85.80  | +316%    | degraded |

Key findings:
- Per-channel granularity is critical for INT4: 937% → 3%
- INT8 per-tensor is already safe (+0.75%); per-channel is nearly lossless
- BF16/FP8 all safe; INT4 per-tensor and NF4 are unusable; INT2 catastrophic

## INT4 Per-Channel Explanation
int4_ch is genuinely accurate (not a bug). Verified:
- Each of 1024 KV feature dims gets its own scale → 15 distinct values per column
- Per-tensor: outlier dims (max ~200) set the global scale → 960/1024 normal dims round to 0
- Per-channel: each dim scaled independently → all dims use all 15 levels

## KV Outlier Analysis (Qwen3-8B, 128 tokens FP16)
File: `research/poc/kv_outliers.png`
- K cache: **64/1024 dims** are outliers (max > 3× median), consistent across all 36 layers
- V cache: 0 outlier dims — much more uniform distribution
- Outlier dims form persistent vertical stripes in the (layer × dim) heatmap
- This structural outlier pattern explains why per-tensor INT4 fails: one global scale
  is dominated by the 64 outlier dims, crushing the other 960

## INT2 Sweep — Qwen3-8B-Q8_0, Strategy B, 20 chunks × 128 tokens
File: `research/poc/results_int2_sweep.json`

| Format  | PPL     | vs FP16  | Notes |
|---------|---------|----------|-------|
| int2    | 15,838  | +76,700% | completely unusable |
| int2_ch | 293     | +1,322%  | 54× better than per-tensor, still unusable |

Key finding: Per-channel helps (54× improvement) but INT2 is fundamentally too coarse — only 4 levels
total (3 positive) cannot represent attention key distributions adequately.

## Pending Experiments
1. Context-length sweep: n_ctx ∈ {128, 512, 4096} — does degradation scale with context?
2. Qwen2.5-Coder-7B-Instruct: coding-agent scenario with code corpus
3. Strategy C vs D comparison: how much comes from prompt KV vs decode KV?
