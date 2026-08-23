# Hybrid Trace-Calibrated Performance Model

## Purpose

This note describes a possible successor or companion to
`research/scripts/layerwise_roofline_sim.py`.

The current layerwise roofline simulator is useful because it exposes FLOPs,
bytes, and per-operation attribution. Its main weakness is that it estimates a
decode step as a serial sum of per-op roofline times:

```text
step_time = sum_i max(compute_time_i, memory_time_i)
```

That is easy to inspect, but it does not fully represent real GPU execution.
Real inference time depends on kernel fusion, launch overhead, dequantization,
occupancy, memory/compute overlap, backend-specific kernels, and batch-size
regimes. A new model should preserve the explainability of the roofline model
while calibrating against measured llama.cpp traces.

The goal is not to replace measurement. The goal is to make predictions and
what-if comparisons more accurate, more honest, and easier to validate.

## Contribution Note

This repository does not accept pull requests that are fully or predominantly
AI-generated. Any implementation based on this document should be designed,
written, tested, and justified by a human contributor. AI assistance should be
disclosed according to `CONTRIBUTING.md`.

## High-Level Design

Use a hybrid model with three layers:

1. Analytic feature extraction from model structure, quantization, hardware,
   batch size, and context length.
2. A resource-stream performance model that separates compute, weight memory,
   KV memory, dequantization, and fixed overhead.
3. Trace calibration against measured llama.cpp benchmark and profiler data.

The model should support two operating modes:

1. Explainability mode: report bottleneck attribution and per-group costs.
2. Prediction mode: report calibrated latency with uncertainty and warnings
   when extrapolating beyond the calibration data.

## Scope

The first version should focus on single-GPU inference. Multi-GPU, RPC, CPU, and
heterogeneous backends can be added later using the same schema.

Initial target workloads:

- Decode one generated token for batch size B at context length L.
- Prefill a prompt of length S.
- Speculative verification of K draft tokens in one verifier pass.

Initial target backends:

- CUDA first, because the existing measurements and roofline scripts are
  mostly GPU-oriented.
- Other backends later, with backend-specific kernel group definitions.

## Main Concept: Kernel Groups

The current layerwise simulator models logical transformer operations:

```text
rms_norm_pre_attn
q_proj
k_proj
v_proj
attn_core
o_proj
ffn_gate
ffn_up
ffn_down
residual
```

The new model should group work closer to how llama.cpp executes kernels.
Suggested groups:

```text
norm_qkv_group
attention_kv_group
o_proj_group
ffn_group
elementwise_group
kv_write_group
sampling_group_optional
```

The exact grouping should be backend-specific. For example, a CUDA build with
fused kernels may use different groups than Metal, Vulkan, CPU BLAS, or a future
backend.

Each group should expose:

- FLOPs.
- Weight bytes.
- Activation bytes.
- KV read bytes.
- KV write bytes.
- Temporary or spill bytes.
- Dequantization work.
- Number of kernels.
- Shape descriptors.
- Quantization descriptors.
- Backend and kernel-family identifiers.

## Decode Resource Model

For decode, separate the major resources:

```text
T_decode_step =
    T_fixed
  + T_nonoverlap
  + max(T_compute_stream, T_weight_stream, T_kv_stream)
```

Where:

```text
T_compute_stream = compute_flops / effective_compute
T_weight_stream  = weight_bytes / effective_weight_bandwidth
T_kv_stream      = kv_read_bytes / effective_kv_bandwidth
```

This avoids the biggest weakness of the serial layerwise model: summing many
independent memory bottlenecks even when the hardware can overlap or pipeline
some work.

A more expressive version can include partial overlap terms:

```text
T_decode_step =
    T_fixed
  + max(T_compute_stream, T_weight_stream, T_kv_stream)
  + alpha_weight_nonoverlap * min(T_compute_stream, T_weight_stream)
  + alpha_kv_nonoverlap     * min(T_compute_stream, T_kv_stream)
  + T_small_kernel_tail
```

The alpha terms should be fitted from data and constrained to reasonable ranges.
The first implementation can start without them.

## Batch Scaling

Do not model batch scaling only as "wall time divided by B". Different cost
terms scale differently:

- Weight reads are mostly per step and are amortized over B tokens.
- KV reads scale approximately with B times context length.
- Compute scales with B but efficiency usually improves as B increases.
- Kernel launch overhead is mostly per step.
- Dequantization may scale with weights, KV rows, or both depending on the
  quantization format and kernel path.

Suggested decode formula:

```text
T_decode_step(B, L) =
    T_fixed(B)
  + W_bytes / BW_weight_eff(B, weight_quant, shape)
  + B * L * KV_bytes_per_token / BW_kv_eff(B, L, kv_quant)
  + FLOPs(B, L) / Compute_eff(B, shape, quant)
  + T_dequant(B, L, weight_quant, kv_quant)
  + T_tail(B, L)

ms_per_token = T_decode_step(B, L) / B
```

This keeps the term-level interpretation clear while allowing each term to have
its own scaling behavior.

## Prefill Model

Prefill should use the same framework, not a separate unrelated formula.

Decode attention for one new token is approximately linear in context length.
Prefill attention over a prompt is causal and has different geometry. The model
should explicitly distinguish:

```text
decode:  one query position attends to L cached positions
prefill: S query positions attend causally within S prompt positions
verify:  K draft positions attend to shared context and to each other
```

Suggested prefill resource terms:

```text
T_prefill(S, B) =
    T_fixed_prefill
  + max(T_compute_prefill, T_weight_prefill, T_attention_prefill)
  + T_nonoverlap_prefill
```

Prefill matrix shapes are larger than decode shapes, so GEMMs may reach a
different efficiency regime. The fitted efficiency tables should therefore use
different regimes for decode and prefill.

## Speculative Verification

For speculative decoding, the verifier processes K draft tokens in one forward
pass. The model should treat this as its own workload type, not merely "decode
with batch size K" unless validation confirms that approximation.

Important terms:

- Verifier weight reads are mostly paid once for the verification step.
- KV reads scale with K times context length.
- Attention geometry includes the K draft positions.
- Accepted-token throughput depends on the acceptance distribution, not only
  verifier latency.

Useful output:

```text
verify_step_ms(K)
expected_accepted_tokens(K)
effective_ms_per_accepted_token(K)
break_even_K
```

The acceptance model can remain outside the performance model initially. The
performance model only needs to expose verifier step latency.

## Quantization Model

Quantization should not be represented only as effective bits per weight.

For each weight and KV quantization format, track:

- Storage bytes.
- Scale and metadata bytes.
- Alignment and padding overhead.
- Dequantization operation count.
- Whether dequantization is fused into the main kernel.
- Whether tensor cores or specialized kernels are available.
- Expected memory coalescing behavior.
- Backend-specific kernel family.

For example, q4 is not automatically twice as fast as q8. It may use a different
kernel path, different dequant cost, and different achieved bandwidth.

Suggested quantization descriptor:

```text
quant_kind
storage_bits_per_value
effective_bits_per_value
block_size
scale_bytes_per_block
zero_point_bytes_per_block
layout_family
dequant_family
backend_kernel_family
```

The model should continue to support GGUF-exact tensor byte extraction. Synthetic
bits-per-weight accounting is useful for quick estimates, but GGUF-exact bytes
should be preferred for calibrated prediction.

## Effective Efficiency Tables

Avoid a single global eta value per operation family. Instead, use small fitted
tables or smooth functions over the most important regimes.

Suggested axes:

```text
backend
hardware
workload_type: decode, prefill, verify
kernel_group
weight_quant
kv_quant
batch_bucket
context_bucket
shape_bucket
```

Example buckets:

```text
batch:   1, 2, 4, 8, 16, 32, 64
context: 512, 1k, 2k, 4k, 8k, 16k, 32k
```

Each table cell can store:

```text
compute_efficiency
weight_bandwidth_efficiency
kv_bandwidth_efficiency
launch_overhead_ms
dequant_cost_scale
residual_error_stats
sample_count
```

Sparse cells should fall back to broader defaults, such as same hardware and
kernel group but neighboring batch/context buckets.

## Calibration Data

The model should fit against measured data, not only final aggregate throughput.

Useful data sources:

- Existing `benchmark_kv_timing.py` JSON rows.
- Existing layerwise eta and calibration JSONs.
- llama.cpp timing output.
- CUDA event timing if available.
- Nsight Systems or Nsight Compute summaries if available.
- Per-kernel timing grouped into the kernel groups above.

Minimum benchmark row fields:

```text
hardware
backend
llama_cpp_commit
build_flags
gpu_name
driver_version
model_preset
model_path_optional
weight_quant
kv_quant
batch_size
prompt_len
decode_len
context_mid
measured_decode_ms_per_token
measured_prefill_ms_optional
measured_total_ms_optional
```

Profiler-derived rows can add:

```text
kernel_name
kernel_group
kernel_ms
achieved_bandwidth_optional
achieved_flops_optional
occupancy_optional
dram_bytes_optional
```

## Fitting Strategy

Use a hierarchical fitting process:

1. Fit hardware/backend global constants.
2. Fit kernel-group constants.
3. Fit quant-format corrections.
4. Fit batch/context bucket corrections.
5. Fit model-specific residuals only if necessary.

The model should prefer fewer parameters over perfect training error. A perfect
fit that does not generalize is not useful.

Recommended validation splits:

- Hold out some batch sizes.
- Hold out some context lengths.
- Hold out at least one KV quantization format if enough data exists.
- Hold out at least one model size if enough data exists.

Report residuals by regime, not only average error.

## Uncertainty

The model should output uncertainty, not just a point estimate.

Example:

```text
predicted_ms_per_token: 1.42
prediction_interval_ms_per_token: [1.28, 1.63]
confidence: medium
```

Uncertainty should increase when:

- The hardware is not present in calibration data.
- The backend or kernel family differs from calibration data.
- Batch size is outside the fitted range.
- Context length is outside the fitted range.
- Quantization format is unseen.
- The model architecture is substantially different.
- The prediction uses synthetic bytes instead of GGUF-exact bytes.

The uncertainty system can start simple. For each fitted table cell, store
residual mean and residual standard deviation. For extrapolated predictions,
apply a penalty factor.

## Output Schema

The prediction output should be machine-readable and explainable.

Suggested top-level fields:

```text
model_version
hardware
backend
workload_type
model
request_shape
quantization
prediction
resource_breakdown
kernel_group_breakdown
calibration_status
warnings
```

Suggested prediction fields:

```text
decode_step_ms
ms_per_token
tokens_per_second
prediction_interval_ms_per_token
confidence
dominant_resource
regime
```

Suggested resource breakdown:

```text
compute_ms
weight_memory_ms
kv_memory_ms
dequant_ms
fixed_overhead_ms
nonoverlap_ms
tail_ms
```

Suggested warnings:

```text
using_synthetic_weight_bytes
unseen_kv_quant
context_extrapolation
batch_extrapolation
backend_mismatch
low_calibration_sample_count
```

## Relationship to Existing Scripts

The new model should reuse existing pieces where possible:

- `model_structures.json` for model shape presets.
- `perf_model.kv_bytes_per_token` for KV byte accounting.
- `gguf_layerwise_weights.py` for GGUF tensor byte extraction.
- `benchmark_kv_timing.py` JSON output for calibration data.
- Existing eta and calibration JSON files as baseline inputs or comparison data.

The current `layerwise_roofline_sim.py` should remain useful as:

- A transparent baseline.
- A per-operation attribution tool.
- A comparison target for the new model.
- A fallback when no trace calibration exists.

## Development Phases

### Phase 1: Feature Extractor

Build a feature extractor that takes:

```text
model structure
hardware preset
weight quantization
KV quantization
batch size
context length
workload type
optional GGUF tensor sizes
```

and outputs resource features:

```text
FLOPs
weight bytes
KV read bytes
KV write bytes
activation bytes
dequant descriptors
shape descriptors
kernel-group estimates
```

This phase should not fit anything yet.

### Phase 2: Baseline Resource-Stream Predictor

Implement the first predictor:

```text
T = T_fixed + max(T_compute, T_weight_mem, T_kv_mem) + T_tail
```

Use conservative default efficiencies. Compare it against the current layerwise
roofline model on the same benchmark rows.

### Phase 3: Calibration Loader

Load benchmark JSON files and normalize them into one calibration schema. Include
metadata needed to detect incompatible measurements, such as hardware, backend,
commit, quantization, and model preset.

### Phase 4: Fitting

Fit efficiency tables and overhead terms. Start with a small parameter set:

```text
fixed_overhead_ms by backend/hardware/workload
weight_bandwidth_efficiency by backend/hardware/weight_quant/batch_bucket
kv_bandwidth_efficiency by backend/hardware/kv_quant/context_bucket
compute_efficiency by backend/hardware/workload/batch_bucket
```

Add more parameters only when residual analysis shows a consistent missing
effect.

### Phase 5: Validation Report

Produce a validation report with:

- Predicted vs measured scatter plots.
- Residuals by batch size.
- Residuals by context length.
- Residuals by weight quantization.
- Residuals by KV quantization.
- Comparison against `layerwise_roofline_sim.py`.
- Warnings for regimes with low sample counts.

### Phase 6: CLI and JSON Output

Expose a CLI that can:

- Predict latency for a requested setup.
- Explain bottleneck attribution.
- Compare to measured rows.
- Write JSON output.
- Write a validation report.

The CLI should make the model mode explicit:

```text
analytic_layerwise
resource_stream
trace_calibrated
```

### Phase 7: Prefill and Verify

After decode works, extend the same framework to prefill and speculative
verification. Keep these as explicit workload types with separate validation.

## Evaluation Criteria

The new model should be considered useful only if it improves at least one of:

- Lower prediction error than the current calibrated layerwise model.
- Better generalization to held-out batch sizes or context lengths.
- Better explanation of q8/q4/q3/q2 KV behavior.
- Better distinction between weight-bound and KV-bound regimes.
- More honest uncertainty when extrapolating.

Suggested metrics:

```text
mean_absolute_percentage_error
median_absolute_percentage_error
p90_absolute_percentage_error
signed_bias
coverage_of_prediction_interval
```

Always report errors by regime. A low average error can hide a bad model if one
important regime is consistently wrong.

## Risks

Main risks:

- Too many fitted parameters can hide bad assumptions.
- Calibration data may be too narrow for general prediction.
- Backend/kernel changes can invalidate fitted constants.
- Profiler data may be hard to collect consistently.
- Quantization formats may need backend-specific handling.
- Prefill and decode may require separate treatment despite sharing the schema.

Mitigations:

- Start with a small parameter set.
- Keep all calibration metadata.
- Emit warnings for extrapolation.
- Keep the current layerwise model as a baseline.
- Prefer GGUF-exact bytes for real predictions.
- Require validation plots before trusting new fitted constants.

## Recommended First Milestone

The first milestone should be intentionally small:

1. Decode-only.
2. CUDA/H100-focused.
3. Reuse existing Qwen benchmark JSON rows.
4. Compare three predictors:

```text
current raw layerwise roofline
current calibrated layerwise roofline
new resource-stream model
```

5. Report prediction error by batch size, context length, weight quantization,
   and KV quantization.

If the resource-stream model does not beat the current calibrated model, keep it
as an explanatory experiment rather than expanding it.

