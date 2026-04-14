#!/usr/bin/env python3
"""perf_model.py — Analytical performance model for KV cache quantization.

Models decode as memory-bandwidth-bound (KV reads + weight reads) and prefill as
compute-bound.  Supports the adaptive quantization scheme (draft + periodic
verification + rollback on mismatch) via statistics from --adaptive-sim.

Usage examples:
    # Baseline comparison of quant types, Qwen3-8B, A100
    python3 research/scripts/perf_model.py \\
        --model qwen3-8b --hw a100-80g \\
        --n-prompt 4096 --n-decode 512

    # Adaptive sim: feed per-example JSON from --adaptive-sim run
    python3 research/scripts/perf_model.py \\
        --model qwen3-8b --hw a100-80g \\
        --n-prompt 4096 --n-decode 512 \\
        --sim-json results_adaptive_per_example.json \\
        --draft int3_half_1357 --verifier int4_ch

    # Override hardware / model parameters directly
    python3 research/scripts/perf_model.py \\
        --model qwen3-8b \\
        --hw-memory-bw 3350 --hw-compute 989 \\
        --n-prompt 8192 --n-decode 2048 \\
        --quants fp16 int8_ch int4_ch int3_half_1357 int2_ch

Output columns:
    quant          KV quantization name
    bw_ratio       KV bandwidth vs fp16 (lower = more compressed)
    kv_frac        KV bytes / total memory bytes at this context (decode sensitivity)
    T_prefill      Prefill time (s) — compute-bound, quant-independent
    T_decode       Decode time (s) — memory-bound, KV quant matters here
    T_total        Total time (s)
    speedup        Relative to fp16 total time
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Optional

# ─── Hardware presets ─────────────────────────────────────────────────────────

HARDWARE_PRESETS = {
    "a100-80g": dict(
        compute_tflops = 312.0,   # BF16 Tensor Core peak
        memory_bw_gbps = 2000.0,  # HBM2e
        efficiency      = 0.70,   # roofline utilisation fraction
    ),
    "h100-sxm": dict(
        compute_tflops = 989.0,
        memory_bw_gbps = 3350.0,
        efficiency      = 0.70,
    ),
    "a6000": dict(
        compute_tflops = 154.0,
        memory_bw_gbps = 768.0,
        efficiency      = 0.65,
    ),
    "rtx4090": dict(
        compute_tflops = 165.0,
        memory_bw_gbps = 1008.0,
        efficiency      = 0.65,
    ),
    "mn5-h100": dict(   # MN5 cluster: 4× H100 SXM5 per node; single-GPU numbers
        compute_tflops = 989.0,
        memory_bw_gbps = 3350.0,
        efficiency      = 0.70,
    ),
    # GH200 (Grace Hopper): same order-of-magnitude as H100 for roofline; tune to your SKU.
    "gh200": dict(
        compute_tflops = 989.0,
        memory_bw_gbps = 3350.0,
        efficiency      = 0.70,
    ),
}

# ─── Model presets ────────────────────────────────────────────────────────────
# n_kv_heads: GQA KV heads (< n_heads for MQA/GQA)
# ffn_dim: intermediate dimension (pre-activation; SwiGLU has 3 × d_model × ffn_dim FLOPs)

MODEL_PRESETS = {
    "qwen3-8b": dict(
        n_layers   = 36,
        d_model    = 4096,
        n_heads    = 32,
        n_kv_heads = 8,
        head_dim   = 128,
        ffn_dim    = 12288,
        ffn_style  = "swiglu",   # 3 matmuls: gate, up, down
        weight_bits = 16,         # weight precision (fp16 by default)
    ),
    "qwen3-14b": dict(
        n_layers   = 40,
        d_model    = 5120,
        n_heads    = 40,
        n_kv_heads = 8,
        head_dim   = 128,
        ffn_dim    = 17920,
        ffn_style  = "swiglu",
        weight_bits = 16,
    ),
    "llama4-scout-17b": dict(
        n_layers   = 48,
        d_model    = 5120,
        n_heads    = 40,
        n_kv_heads = 8,
        head_dim   = 128,
        ffn_dim    = 16384,
        ffn_style  = "swiglu",
        weight_bits = 16,
    ),
    "gpt-oss-20b": dict(
        n_layers   = 40,
        d_model    = 5120,
        n_heads    = 40,
        n_kv_heads = 8,
        head_dim   = 128,
        ffn_dim    = 13824,
        ffn_style  = "swiglu",
        weight_bits = 16,
    ),
    # Qwen3.5-27B — verify against model config.json / GGUF if results look off.
    "qwen3.5-27b": dict(
        n_layers   = 62,
        d_model    = 7168,
        n_heads    = 56,
        n_kv_heads = 8,
        head_dim   = 128,
        ffn_dim    = 18944,
        ffn_style  = "swiglu",
        weight_bits = 16,
    ),
}

# ─── KV quantization configs ──────────────────────────────────────────────────
# bits:       data bits per element (effective; int3_half stores 4 levels = 2 bits)
# scale_bits: bits for each scale/zero-point value (0 = no scale, e.g. fp16)
# group_size: tokens (per-channel) or elements (per-token) sharing one scale
# asym:       asymmetric adds a zero-point (doubles scale storage)

QUANT_CONFIGS = {
    "fp16":               dict(bits=16, scale_bits=0,  group_size=1,  asym=False),
    "bf16":               dict(bits=16, scale_bits=0,  group_size=1,  asym=False),
    "int8_ch":            dict(bits=8,  scale_bits=16, group_size=64, asym=False),
    "int4_ch":            dict(bits=4,  scale_bits=16, group_size=64, asym=False),
    "int4_tok":           dict(bits=4,  scale_bits=16, group_size=64, asym=False),
    "int3_ch":            dict(bits=3,  scale_bits=16, group_size=64, asym=False),
    "int3_half":          dict(bits=2,  scale_bits=16, group_size=64, asym=False),  # 4 levels → 2 bits
    "int3_half_1357":     dict(bits=2,  scale_bits=16, group_size=64, asym=False),
    "int3_half_1357_ch":  dict(bits=2,  scale_bits=16, group_size=64, asym=False),
    "int2_ch":            dict(bits=2,  scale_bits=16, group_size=64, asym=False),
    "int2":               dict(bits=2,  scale_bits=16, group_size=64, asym=False),
}

# ─── Core bandwidth calculations ──────────────────────────────────────────────

def kv_bytes_per_token(quant_name: str, model: dict,
                        group_size: Optional[int] = None,
                        asym: bool = False) -> float:
    """Bytes loaded per KV token pair (K+V), all layers, all KV heads.

    Includes amortised scale overhead (scale bytes / group_size tokens).
    For asymmetric quant the zero-point doubles the scale storage.
    """
    qcfg = QUANT_CONFIGS.get(quant_name)
    if qcfg is None:
        raise ValueError(f"Unknown quant: {quant_name!r}. Known: {list(QUANT_CONFIGS)}")

    bits       = qcfg["bits"]
    scale_bits = qcfg["scale_bits"]
    gs         = group_size if group_size is not None else qcfg["group_size"]
    scale_mult = 2 if asym else 1   # zero-point doubles the scale

    n_kv = model["n_kv_heads"]
    hdim = model["head_dim"]
    nl   = model["n_layers"]

    # Data: K + V, each [n_tokens, n_kv_heads, head_dim]
    data_bytes = 2 * n_kv * hdim * (bits / 8) * nl

    # Scale: per-channel → one scale per head_dim element per group_size tokens
    # Amortised to per-token: scale_bytes / group_size
    if scale_bits > 0 and gs > 1:
        scale_bytes = 2 * n_kv * hdim * (scale_bits / 8) * scale_mult / gs * nl
    else:
        scale_bytes = 0.0

    return data_bytes + scale_bytes


def weight_bytes(model: dict) -> float:
    """Total bytes for all transformer weights (per-decode-step reload for batch=1).

    At batch_size=1, decode is memory-bound: all weights reloaded each step.
    Includes QKV proj, output proj, FFN (SwiGLU = 3 matmuls), layer norms.
    """
    nl    = model["n_layers"]
    d     = model["d_model"]
    nkv   = model["n_kv_heads"]
    nh    = model["n_heads"]
    hdim  = model["head_dim"]
    ffn   = model["ffn_dim"]
    style = model.get("ffn_style", "swiglu")
    wbits = model.get("weight_bits", 16)

    bytes_per_elem = wbits / 8

    # Q proj: d × (n_heads × head_dim); K,V proj: d × (n_kv_heads × head_dim)
    qkv_bytes = (d * nh * hdim + 2 * d * nkv * hdim) * bytes_per_elem
    out_bytes  = nh * hdim * d * bytes_per_elem  # output projection
    # FFN: SwiGLU = gate + up + down; standard = up + down
    ffn_mats   = 3 if style == "swiglu" else 2
    ffn_bytes  = ffn_mats * d * ffn * bytes_per_elem

    return nl * (qkv_bytes + out_bytes + ffn_bytes)


# ─── Timing models ────────────────────────────────────────────────────────────

def prefill_time(n_prompt: int, model: dict, hw: dict) -> float:
    """Prefill time (seconds) — compute-bound.

    FLOPs: attention O(n²·d) + QKV+FFN projections O(n·d²).
    KV quantisation has negligible impact on prefill time.
    """
    nl   = model["n_layers"]
    d    = model["d_model"]
    nkv  = model["n_kv_heads"]
    nh   = model["n_heads"]
    hdim = model["head_dim"]
    ffn  = model["ffn_dim"]
    style = model.get("ffn_style", "swiglu")

    # Per-layer FLOPs (forward pass, factor-of-2 for multiply-add)
    flops_qkv   = 2 * n_prompt * (d * nh * hdim + 2 * d * nkv * hdim)
    flops_attn  = 2 * nh * (n_prompt ** 2) * hdim        # QK^T + softmax·V
    flops_out   = 2 * n_prompt * nh * hdim * d
    ffn_mats    = 3 if style == "swiglu" else 2
    flops_ffn   = ffn_mats * 2 * n_prompt * d * ffn

    total_flops = nl * (flops_qkv + flops_attn + flops_out + flops_ffn)

    peak_flops_s = hw["compute_tflops"] * 1e12 * hw["efficiency"]
    return total_flops / peak_flops_s


def model_step_flops(model: dict, n_ctx: int) -> float:
    """FLOPs for one decode step (one token, one sequence).

    Counts multiply-adds as 2 FLOPs each.  Includes QKV projection, attention
    (QK scores + softmax·V), output projection, and FFN.
    """
    nl, d, nh, nkv, hd, ffn = (model["n_layers"], model["d_model"], model["n_heads"],
                                model["n_kv_heads"], model["head_dim"], model["ffn_dim"])
    ffn_mats = 3 if model.get("ffn_style") == "swiglu" else 2
    return 2 * nl * (
        d * (nh * hd + 2 * nkv * hd)   # QKV projection
        + nh * n_ctx * hd               # QK^T scores
        + nh * n_ctx * hd               # softmax·V
        + nh * hd * d                   # output projection
        + ffn_mats * d * ffn            # FFN
    )


def critical_batch_size(model: dict, hw: dict,
                         kv_bpt: float, n_ctx: int) -> float:
    """Batch size at which decode flips from memory-bound to compute-bound.

    At B < critical: memory-bound (weights + KV dominate).
    At B > critical: compute-bound (FLOPs scale with B, memory does not scale proportionally).

    Derived from: B × step_flops / compute_peak = (w_bytes + B × kv_bpt × n_ctx) / bw
    → B = compute_peak × w_bytes / (step_flops × bw - compute_peak × kv_bpt × n_ctx)
    Returns inf if KV alone already saturates the compute roof (very long context).
    """
    w_bytes      = weight_bytes(model)
    step_flops   = model_step_flops(model, n_ctx)
    compute_peak = hw["compute_tflops"] * 1e12 * hw["efficiency"]
    bw           = hw["memory_bw_gbps"] * 1e9  * hw["efficiency"]
    denom = step_flops * bw - compute_peak * kv_bpt * n_ctx
    if denom <= 0:
        return float("inf")   # KV alone would exceed compute; never compute-bound
    return compute_peak * w_bytes / denom


def decode_time_for_tokens(n_start: int, n_steps: int,
                            kv_bpt: float, w_bytes: float, hw: dict,
                            batch_size: int = 1,
                            step_flops: float = 0.0) -> float:
    """Decode time (seconds) for n_steps tokens, batch_size parallel sequences.

    Memory model (roofline lower bound):
      - Weights loaded ONCE per step, shared across the batch (batch_size has no effect on weight BW)
      - KV cache loaded for EACH sequence: scales as batch_size × kv_bpt × n_tokens

      T_mem = Σ_{i=0}^{n_steps-1} (w_bytes + batch_size × kv_bpt × (n_start+i)) / bw_eff

    Compute model (roofline upper bound):
      T_compute = batch_size × step_flops × n_steps / compute_peak

    Actual time = max(T_mem, T_compute)  [roofline: bottleneck wins]
    """
    bw_eff = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"]
    T_mem  = (n_steps * w_bytes
              + batch_size * kv_bpt * (n_start * n_steps + n_steps * (n_steps - 1) / 2)
              ) / bw_eff

    if step_flops > 0 and batch_size > 1:
        compute_peak = hw["compute_tflops"] * 1e12 * hw["efficiency"]
        T_compute    = batch_size * step_flops * n_steps / compute_peak
        return max(T_mem, T_compute)
    return T_mem


def decode_time(n_prompt: int, n_decode: int,
                quant_name: str, model: dict, hw: dict,
                group_size: Optional[int] = None, asym: bool = False,
                batch_size: int = 1) -> float:
    """Total decode time (seconds) for standard (non-adaptive) generation."""
    kv_bpt     = kv_bytes_per_token(quant_name, model, group_size, asym)
    w_bytes    = weight_bytes(model)
    step_flops = model_step_flops(model, n_prompt + n_decode // 2)
    return decode_time_for_tokens(n_prompt, n_decode, kv_bpt, w_bytes, hw,
                                  batch_size, step_flops)


def kv_fraction(n_prompt: int, n_decode: int,
                quant_name: str, model: dict,
                group_size: Optional[int] = None, asym: bool = False,
                batch_size: int = 1) -> float:
    """Fraction of total memory bandwidth consumed by KV loads (vs weights).

    Weights are shared across the batch; KV scales with batch_size.
    High kv_frac → KV quant gives large decode speedup.
    Low  kv_frac → weight-bound; KV quant has diminishing returns.
    """
    kv_bpt  = kv_bytes_per_token(quant_name, model, group_size, asym)
    w_bytes = weight_bytes(model)
    avg_ctx = n_prompt + (n_decode - 1) / 2
    total_kv = batch_size * n_decode * kv_bpt * avg_ctx
    total_w  = n_decode * w_bytes            # shared across batch — no B multiplier
    return total_kv / (total_kv + total_w)


# ─── Adaptive quantization model ──────────────────────────────────────────────

@dataclass
class AdaptiveStats:
    """Per-example or aggregate statistics from --adaptive-sim output."""
    acceptance_rate:   float   # fraction of windows where draft == verifier
    draft_fraction:    float   # fraction of decode tokens in accepted windows
    first_fail_fraction: float # first_fail_pos / n_decode (0.0 = fail immediately, 1.0 = never)
    window_size:       int     # W tokens per verification window
    n_examples:        int = 1


def adaptive_stats_from_json(path: str, quant_key: str) -> AdaptiveStats:
    """Load aggregate adaptive sim stats from --save-per-example JSON.

    Looks for key "{quant_key}__adaptive_sim" in the JSON.
    Falls back to scanning all keys for adaptive_sim entries.
    """
    with open(path) as f:
        data = json.load(f)

    sim_key = f"{quant_key}__adaptive_sim"
    if sim_key not in data:
        candidates = [k for k in data if k.endswith("__adaptive_sim")]
        if not candidates:
            raise KeyError(f"No adaptive_sim keys found in {path}. "
                           f"Available: {list(data.keys())[:10]}")
        sim_key = candidates[0]
        print(f"[perf_model] Using sim key: {sim_key!r}", file=sys.stderr)

    records = data[sim_key]
    valid   = [r for r in records if r is not None]
    if not valid:
        raise ValueError(f"No valid adaptive sim records in {sim_key}")

    n_windows_total   = sum(r["n_windows"] for r in valid)
    n_accepted_total  = sum(r["n_accepted_windows"] for r in valid)
    acceptance_rate   = n_accepted_total / n_windows_total if n_windows_total > 0 else 0.0

    # draft_fraction: average across examples
    draft_fraction    = sum(r["draft_fraction"] for r in valid) / len(valid)

    # first_fail_fraction: when does the first fail happen, relative to n_decode?
    # Use first_fail_pos / n_tokens_draft for each example
    ffrac_vals = []
    for r in valid:
        if r["first_fail_pos"] is not None and r["n_tokens_draft"] > 0:
            ffrac_vals.append(r["first_fail_pos"] / r["n_tokens_draft"])
        else:
            ffrac_vals.append(1.0)  # no fail
    first_fail_fraction = sum(ffrac_vals) / len(ffrac_vals)

    # Window size: infer from first record
    w_size = valid[0].get("n_windows", 1)
    # Try to recover window_size from first_fail_window and first_fail_pos
    w_inferred = 32  # default
    for r in valid:
        if r["first_fail_window"] is not None and r["first_fail_pos"] is not None and r["first_fail_window"] > 0:
            w_inferred = r["first_fail_pos"] // r["first_fail_window"]
            break

    return AdaptiveStats(
        acceptance_rate    = acceptance_rate,
        draft_fraction     = draft_fraction,
        first_fail_fraction= first_fail_fraction,
        window_size        = w_inferred,
        n_examples         = len(valid),
    )


def adaptive_decode_time(n_prompt: int, n_decode: int,
                          draft_quant: str, verifier_quant: str,
                          stats: AdaptiveStats,
                          model: dict, hw: dict,
                          verification_mode: str = "sequential",
                          group_size: Optional[int] = None,
                          asym: bool = False,
                          batch_size: int = 1) -> dict:
    """Estimate adaptive scheme decode time and breakdown.

    Phases:
      1. Draft phase (0 → first_fail_pos): generate at draft bw + verify at verifier bw
      2. Verifier tail (first_fail_pos → n_decode): generate at verifier bw

    verification_mode:
      "sequential"  — draft window then verify window back-to-back (worst case)
      "parallel"    — draft and verify overlap (best case); time = max(draft, verify) per window
      "amortised"   — one verify per W draft tokens (ideal pipeline)

    Returns dict with timing breakdown and speedup vs fp16.
    """
    draft_kv   = kv_bytes_per_token(draft_quant,    model, group_size, asym)
    verif_kv   = kv_bytes_per_token(verifier_quant, model, group_size, asym)
    fp16_kv    = kv_bytes_per_token("fp16",          model, group_size, asym)
    w_bytes    = weight_bytes(model)
    sflops     = model_step_flops(model, n_prompt + n_decode // 2)

    def _dt(n_start, n_steps, kv):
        return decode_time_for_tokens(n_start, n_steps, kv, w_bytes, hw,
                                      batch_size, sflops)

    # Phase boundary
    first_fail_pos = int(stats.first_fail_fraction * n_decode)
    n_draft_phase  = first_fail_pos
    n_tail_phase   = n_decode - first_fail_pos

    # Phase 1: draft + verification
    if verification_mode == "parallel":
        heavy_kv = max(draft_kv, verif_kv)
        T1 = _dt(n_prompt, n_draft_phase, heavy_kv)
    elif verification_mode == "amortised":
        W      = stats.window_size
        amort  = 1.0 + verif_kv / (W * draft_kv) if draft_kv > 0 and W > 0 else 1.0
        T1 = _dt(n_prompt, n_draft_phase, draft_kv * amort)
    else:  # sequential
        T1 = _dt(n_prompt, n_draft_phase, draft_kv) + _dt(n_prompt, n_draft_phase, verif_kv)

    # Phase 2: verifier-only tail
    T2 = _dt(n_prompt + n_draft_phase, n_tail_phase, verif_kv)

    T_adaptive = T1 + T2
    T_fp16     = _dt(n_prompt, n_decode, fp16_kv)
    T_verifier = _dt(n_prompt, n_decode, verif_kv)

    return {
        "T_draft_phase":    T1,
        "T_tail_phase":     T2,
        "T_adaptive_decode": T_adaptive,
        "T_fp16_decode":     T_fp16,
        "T_verifier_decode": T_verifier,
        "speedup_vs_fp16":   T_fp16 / T_adaptive if T_adaptive > 0 else float("inf"),
        "speedup_vs_verif":  T_verifier / T_adaptive if T_adaptive > 0 else float("inf"),
        "first_fail_pos":    first_fail_pos,
        "n_draft_phase":     n_draft_phase,
        "n_tail_phase":      n_tail_phase,
        "verification_mode": verification_mode,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def _fmt(x, decimals=3):
    if x is None: return "—"
    if isinstance(x, float) and math.isinf(x): return "∞"
    return f"{x:.{decimals}f}"


def print_baseline_table(n_prompt: int, n_decode: int,
                          quants: list, model: dict, hw: dict,
                          group_size: int = 64, asym: bool = False,
                          batch_size: int = 1):
    """Print a comparison table of quant types for the given workload."""
    w_bytes   = weight_bytes(model)
    fp16_kv   = kv_bytes_per_token("fp16", model, group_size, asym)
    T_prefill = prefill_time(n_prompt, model, hw)

    # ── Roofline analysis ─────────────────────────────────────────────────────
    # Decode has two independent bandwidth consumers per step:
    #   (a) Weights: w_bytes loaded ONCE per step, SHARED across batch
    #       → weight intensity = batch_size × proj_flops / w_bytes  (scales with B)
    #   (b) KV cache: kv_bpt × n_ctx loaded PER SEQUENCE, scales with batch_size
    #       → attention intensity = 2 × (n_heads/n_kv_heads) / bits_per_elem  (fixed!)
    # Attention is ALWAYS memory-bound (intensity << ridge for any practical quant).
    # Weight matmuls flip to compute-bound at large batch_size.
    ridge_pt     = hw["compute_tflops"] * 1e12 / (hw["memory_bw_gbps"] * 1e9)
    avg_ctx      = n_prompt + n_decode / 2
    sflops       = model_step_flops(model, avg_ctx)
    crit_B       = critical_batch_size(model, hw, fp16_kv, avg_ctx)

    # Per-step bytes at this batch size
    step_bytes   = w_bytes + batch_size * fp16_kv * avg_ctx
    # Weight-matmul component intensity (scales with B)
    wt_intensity = batch_size * sflops / step_bytes
    bound        = "compute-bound" if batch_size > crit_B else "memory-bound"

    # Attention intensity (independent of batch_size and n_ctx)
    nh, nkv, hd = model["n_heads"], model["n_kv_heads"], model["head_dim"]
    fp16_bpe     = 2.0
    attn_intens  = 2.0 * (nh / nkv) / fp16_bpe  # FLOPs/byte; fixed by GQA ratio

    W = 108
    print(f"\n{'─'*W}")
    print(f" Performance model  |  model: {args.model}  hw: {args.hw}"
          f"  n_prompt={n_prompt}  n_decode={n_decode}"
          f"  batch={batch_size}  group_size={group_size}"
          + ("  asym" if asym else ""))
    print(f" Weights: {w_bytes/1e9:.2f} GB    fp16 KV/tok: {fp16_kv/1e6:.3f} MB    "
          f"Prefill: {T_prefill:.3f}s")
    print(f" Roofline ridge: {ridge_pt:.0f} FLOPs/B    "
          f"weight-matmul intensity: {wt_intensity:.1f} FLOPs/B (×B={batch_size})  →  {bound}"
          f"  |  crit_batch≈{crit_B:.0f}")
    print(f" Attention always memory-bound: intensity={attn_intens:.1f} FLOPs/B (GQA {nh//nkv}:1)"
          f"  →  KV quant reduces this proportionally")
    mem_kv_frac  = batch_size * fp16_kv * avg_ctx / step_bytes
    print(f" Memory split (fp16, avg ctx {avg_ctx:.0f}):  "
          f"weights={w_bytes/step_bytes:.1%}  KV={mem_kv_frac:.1%}"
          f"  |  KV crossover={int(w_bytes/fp16_kv):,} tok/seq"
          f"  eff crossover (B={batch_size})={int(w_bytes/(fp16_kv*batch_size)):,} tok/seq")
    print(f"{'─'*W}")

    # sys_tok/s label: per-sequence or total system throughput
    sys_label = f"sys_tok/s" if batch_size > 1 else "tot_tok/s"
    hdr = (f"{'quant':<22} {'bw_ratio':>8} {'kv_frac':>8} "
           f"{'T_prefill':>10} {'T_decode':>10} {'dec_tok/s':>10} "
           f"{'T_total':>9} {sys_label:>10} {'speedup':>8}  bound")
    print(hdr)
    print(f"{'─'*W}")

    fp16_total = None
    for q in quants:
        try:
            kv_bpt = kv_bytes_per_token(q, model, group_size, asym)
        except ValueError as e:
            print(f"  {q:<22}  ERROR: {e}")
            continue

        bw_ratio   = kv_bpt / fp16_kv
        kv_frac    = kv_fraction(n_prompt, n_decode, q, model, group_size, asym,
                                  batch_size)
        T_dec      = decode_time_for_tokens(n_prompt, n_decode, kv_bpt, w_bytes, hw,
                                             batch_size, sflops)
        T_total    = T_prefill + T_dec
        dec_toks_s = n_decode / T_dec   if T_dec   > 0 else float("inf")
        # sys_tok/s: total tokens produced per second across the batch
        sys_toks_s = batch_size * n_decode / T_total if T_total > 0 else float("inf")

        if fp16_total is None and q == "fp16":
            fp16_total = T_total
        speedup = fp16_total / T_total if fp16_total else None

        # Bound label per quant (KV bytes change, weight bytes don't)
        q_step_bytes = w_bytes + batch_size * kv_bpt * avg_ctx
        q_intensity  = batch_size * sflops / q_step_bytes
        q_bound = "cmp" if q_intensity > ridge_pt else ("kv>wt" if kv_frac > 0.50 else "wt>kv")

        row = (f"{q:<22} {bw_ratio:>8.3f} {kv_frac:>8.2%} "
               f"{T_prefill:>10.3f} {T_dec:>10.3f} {dec_toks_s:>10.1f} "
               f"{T_total:>9.3f} {sys_toks_s:>10.1f} "
               f"{_fmt(speedup):>8}  {q_bound}")
        print(row)

    print(f"{'─'*W}")
    print(f" dec_tok/s = n_decode/T_decode (per-sequence latency)"
          f"  |  {sys_label} = B×n_decode/T_total (system throughput)")
    print(f" bound: cmp=compute-bound  kv>wt=KV dominates memory  wt>kv=weights dominate memory")
    print()


def print_adaptive_table(n_prompt: int, n_decode: int,
                          draft_quant: str, verifier_quant: str,
                          stats: AdaptiveStats,
                          model: dict, hw: dict,
                          group_size: int = 64, asym: bool = False,
                          batch_size: int = 1):
    """Print adaptive scheme timing breakdown for all verification modes."""
    T_prefill = prefill_time(n_prompt, model, hw)
    modes = ["sequential", "amortised", "parallel"]

    print(f"\n{'─'*90}")
    print(f" Adaptive simulation  |  draft={draft_quant}  verifier={verifier_quant}  "
          f"window={stats.window_size}")
    print(f"   acceptance_rate={stats.acceptance_rate:.3f}  "
          f"draft_fraction={stats.draft_fraction:.3f}  "
          f"first_fail_pos≈{int(stats.first_fail_fraction*n_decode)}/{n_decode}  "
          f"n_examples={stats.n_examples}")
    W = 105
    print(f"{'─'*W}")
    hdr = (f"{'verify_mode':<14} {'T_draft_ph':>10} {'T_tail_ph':>10} "
           f"{'T_decode':>10} {'dec_tok/s':>10} {'T_total':>9} {'tot_tok/s':>10} "
           f"{'vs_fp16':>8} {'vs_verif':>8}")
    print(hdr)
    print(f"{'─'*W}")

    # Compute fp16/verifier reference once (mode-independent)
    r_ref = adaptive_decode_time(n_prompt, n_decode, draft_quant, verifier_quant,
                                  stats, model, hw, "sequential", group_size, asym,
                                  batch_size)
    T_fp16_dec   = r_ref["T_fp16_decode"]
    T_verif_dec  = r_ref["T_verifier_decode"]
    T_fp16_tot   = T_prefill + T_fp16_dec
    T_verif_tot  = T_prefill + T_verif_dec
    sys_s = lambda T_dec: batch_size * n_decode / (T_prefill + T_dec)

    # Reference rows
    print(f"  {'[fp16]':<12} {'':>10} {'':>10} "
          f"{T_fp16_dec:>10.3f} {n_decode/T_fp16_dec:>10.1f} "
          f"{T_fp16_tot:>9.3f} {sys_s(T_fp16_dec):>10.1f} "
          f"{'1.000':>8} {'':>8}")
    print(f"  {'['+verifier_quant+']':<12} {'':>10} {'':>10} "
          f"{T_verif_dec:>10.3f} {n_decode/T_verif_dec:>10.1f} "
          f"{T_verif_tot:>9.3f} {sys_s(T_verif_dec):>10.1f} "
          f"{T_fp16_tot/T_verif_tot:>8.3f} {'1.000':>8}")
    print(f"{'─'*W}")

    for mode in modes:
        r = adaptive_decode_time(n_prompt, n_decode, draft_quant, verifier_quant,
                                  stats, model, hw, mode, group_size, asym,
                                  batch_size)
        T_adap_dec = r["T_adaptive_decode"]
        T_total    = T_prefill + T_adap_dec
        dec_toks_s = n_decode / T_adap_dec          if T_adap_dec > 0 else float("inf")
        sys_toks_s = batch_size * n_decode / T_total if T_total    > 0 else float("inf")
        sp_fp16    = T_fp16_tot  / T_total if T_total > 0 else float("inf")
        sp_verif   = T_verif_tot / T_total if T_total > 0 else float("inf")
        row = (f"  {mode:<12} {r['T_draft_phase']:>10.3f} {r['T_tail_phase']:>10.3f} "
               f"{T_adap_dec:>10.3f} {dec_toks_s:>10.1f} "
               f"{T_total:>9.3f} {sys_toks_s:>10.1f} "
               f"{sp_fp16:>8.3f} {sp_verif:>8.3f}")
        print(row)

    print(f"{'─'*W}")
    print(f"  sequential = draft then verify back-to-back per window (worst case)")
    print(f"  amortised  = verify pipelined with draft; +1/W per step overhead")
    print(f"  parallel   = draft + verify overlap simultaneously (best case)")
    print()


# ─── Benchmark presets ────────────────────────────────────────────────────────
# Characteristic (n_prompt, n_decode) for each task in our sweep suite.
# n_prompt: typical tokenised prompt length after chat template.
# n_decode: max-gen-tokens cap (actual generation may be shorter).

BENCHMARK_PRESETS = {
    "gsm8k":     dict(n_prompt=500,   n_decode=2048,
                      note="8-shot math, short prompt, long CoT"),
    "humaneval": dict(n_prompt=200,   n_decode=2048,
                      note="code completion, very short prompt"),
    "niah-4k":   dict(n_prompt=4000,  n_decode=20,
                      note="needle-in-haystack, short answer"),
    "longbench": dict(n_prompt=8000,  n_decode=2048,
                      note="document QA, long context"),
    "aime":      dict(n_prompt=1000,  n_decode=65536,
                      note="math olympiad, very long thinking trace"),
    "wikitext":  dict(n_prompt=1024,  n_decode=1024,
                      note="flat PPL reference (prefill+score window)"),
}


def print_sweep_table(model: dict, hw: dict,
                      quants: list,
                      batch_sizes: list,
                      group_size: int = 64, asym: bool = False,
                      benchmarks: Optional[list] = None):
    """Compact grid: rows = (benchmark × batch_size), columns = quant speedups.

    Shows for each cell:
      sys_tok/s — total system throughput (B × n_decode / T_total)
      speedup   — vs fp16 at same batch size

    Also highlights where the bottleneck flips: wt>kv → kv>wt.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARK_PRESETS)

    w_bytes = weight_bytes(model)
    fp16_kv = kv_bytes_per_token("fp16", model, group_size, asym)

    # Column layout: fixed cols + one pair per non-fp16 quant
    cmp_quants = [q for q in quants if q != "fp16"]

    # Header
    W = 14 + 7 + 7 + 5 + 8 + len(cmp_quants) * 22
    print(f"\n{'─'*W}")
    print(f" Benchmark sweep  |  model: {args.model}  hw: {args.hw}"
          f"  group_size={group_size}" + ("  asym" if asym else ""))
    print(f" Columns: sys_tok/s (B×n_decode/T_total) and speedup vs fp16 at same batch")
    print(f" Quants compared: fp16  vs  {' / '.join(cmp_quants)}")
    print(f"{'─'*W}")

    # Header row
    hdr_fixed = f"{'benchmark':<14} {'n_p':>6} {'n_d':>7} {'B':>4}  {'kv%':>5}  {'fp16_sys':>9}"
    hdr_quant = "".join(f"  {q:>10} {'spd':>6}" for q in cmp_quants)
    print(hdr_fixed + hdr_quant)
    print(f"{'─'*W}")

    prev_bench = None
    for bench_name in benchmarks:
        bpreset = BENCHMARK_PRESETS[bench_name]
        n_prompt = bpreset["n_prompt"]
        n_decode = bpreset["n_decode"]
        T_prefill = prefill_time(n_prompt, model, hw)

        for B in batch_sizes:
            if prev_bench is not None and prev_bench != bench_name:
                print(f"{'·'*W}")
            prev_bench = bench_name

            avg_ctx    = n_prompt + n_decode / 2
            sflops     = model_step_flops(model, avg_ctx)
            step_bytes = w_bytes + B * fp16_kv * avg_ctx
            kv_frac_fp16 = B * fp16_kv * avg_ctx / step_bytes

            # fp16 baseline
            fp16_kv_bpt = kv_bytes_per_token("fp16", model, group_size, asym)
            T_fp16_dec  = decode_time_for_tokens(n_prompt, n_decode, fp16_kv_bpt,
                                                  w_bytes, hw, B, sflops)
            T_fp16_tot  = T_prefill + T_fp16_dec
            fp16_sys    = B * n_decode / T_fp16_tot if T_fp16_tot > 0 else 0.0

            fixed = (f"{bench_name:<14} {n_prompt:>6} {n_decode:>7} {B:>4}  "
                     f"{kv_frac_fp16:>5.1%}  {fp16_sys:>9.1f}")

            cells = ""
            for q in cmp_quants:
                try:
                    kv_bpt = kv_bytes_per_token(q, model, group_size, asym)
                except ValueError:
                    cells += f"  {'ERR':>10} {'':>6}"
                    continue
                T_dec = decode_time_for_tokens(n_prompt, n_decode, kv_bpt,
                                               w_bytes, hw, B, sflops)
                T_tot = T_prefill + T_dec
                sys_s = B * n_decode / T_tot if T_tot > 0 else 0.0
                spd   = T_fp16_tot / T_tot   if T_tot > 0 else float("inf")
                cells += f"  {sys_s:>10.1f} {spd:>6.2f}×"

            print(fixed + cells)

    print(f"{'─'*W}")
    print(f" kv% = KV fraction of memory bandwidth (fp16 KV).  Higher → KV quant helps more.")
    print(f" All times include prefill.  Batch sizes share weights; KV scales with B.")
    # Show effective crossover for each batch size
    crossover_line = "  KV crossover (tok/seq per batch): " + \
        "  ".join(f"B={B}→{int(w_bytes/(fp16_kv*B)):,}" for B in batch_sizes)
    print(crossover_line)
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global args
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Workload
    p.add_argument("--n-prompt",    type=int, default=1024, help="Prompt length (tokens)")
    p.add_argument("--n-decode",    type=int, default=512,  help="Decode length (tokens)")
    p.add_argument("--batch-size",  type=int, default=1,
                   help="Batch size (parallel sequences). Weights shared across batch; "
                        "KV scales with batch_size. Large batch → KV-bound → KV quant matters more.")

    # Model
    p.add_argument("--model",     default="qwen3-8b",
                   choices=list(MODEL_PRESETS), help="Model preset")
    p.add_argument("--n-layers",   type=int,   default=None)
    p.add_argument("--d-model",    type=int,   default=None)
    p.add_argument("--n-heads",    type=int,   default=None)
    p.add_argument("--n-kv-heads", type=int,   default=None)
    p.add_argument("--head-dim",   type=int,   default=None)
    p.add_argument("--ffn-dim",    type=int,   default=None)
    p.add_argument("--weight-bits",type=int,   default=None,
                   help="Weight precision bits (default 16)")

    # Hardware
    p.add_argument("--hw",         default="a100-80g",
                   choices=list(HARDWARE_PRESETS), help="Hardware preset")
    p.add_argument("--hw-memory-bw",  type=float, default=None,
                   help="Override memory bandwidth (GB/s)")
    p.add_argument("--hw-compute",    type=float, default=None,
                   help="Override compute peak (TFLOP/s)")
    p.add_argument("--hw-efficiency", type=float, default=None,
                   help="Override roofline efficiency (0–1, default 0.70)")

    # Quantization
    p.add_argument("--quants", nargs="+",
                   default=["fp16", "int8_ch", "int4_ch", "int3_ch",
                            "int3_half_1357_ch", "int2_ch"],
                   help="Quant types to compare in baseline table")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--asym",       action="store_true")

    # Adaptive simulation
    p.add_argument("--sim-json",      default=None, metavar="FILE",
                   help="Per-example JSON from --adaptive-sim (run_sweep.py --save-per-example)")
    p.add_argument("--draft",         default="int3_half_1357_ch",
                   help="Draft quant key for adaptive sim (default: int3_half_1357_ch)")
    p.add_argument("--verifier",      default="int4_ch",
                   help="Verifier quant for adaptive sim (default: int4_ch)")
    p.add_argument("--acceptance-rate",   type=float, default=None,
                   help="Override acceptance_rate (0–1); skips --sim-json parsing")
    p.add_argument("--draft-fraction",    type=float, default=None,
                   help="Override draft_fraction (0–1)")
    p.add_argument("--first-fail-frac",   type=float, default=None,
                   help="Override first_fail_fraction (0–1); 1.0 = never fails")
    p.add_argument("--window-size",       type=int,   default=32,
                   help="Verification window size (tokens, default 32)")

    # Sweep mode
    p.add_argument("--sweep", action="store_true",
                   help="Print compact benchmark × batch-size grid instead of per-workload tables")
    p.add_argument("--sweep-batch-sizes", nargs="+", type=int,
                   default=[1, 8, 32, 128],
                   help="Batch sizes in --sweep grid (default: 1 8 32 128)")
    p.add_argument("--sweep-benchmarks", nargs="+",
                   choices=list(BENCHMARK_PRESETS), default=None,
                   help="Benchmarks in --sweep (default: all)")

    args = p.parse_args()

    # ── Build model config ────────────────────────────────────────────────────
    model = dict(MODEL_PRESETS[args.model])
    for attr, key in [("n_layers", "n_layers"), ("d_model", "d_model"),
                       ("n_heads", "n_heads"), ("ffn_dim", "ffn_dim"),
                       ("head_dim", "head_dim"), ("weight_bits", "weight_bits")]:
        val = getattr(args, attr.replace("-", "_"), None)
        if val is not None:
            model[key] = val
    if args.n_kv_heads is not None:
        model["n_kv_heads"] = args.n_kv_heads

    # ── Build hardware config ─────────────────────────────────────────────────
    hw = dict(HARDWARE_PRESETS[args.hw])
    if args.hw_memory_bw  is not None: hw["memory_bw_gbps"] = args.hw_memory_bw
    if args.hw_compute     is not None: hw["compute_tflops"]  = args.hw_compute
    if args.hw_efficiency  is not None: hw["efficiency"]       = args.hw_efficiency

    B = args.batch_size

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if args.sweep:
        print_sweep_table(model, hw,
                          quants      = args.quants,
                          batch_sizes = args.sweep_batch_sizes,
                          group_size  = args.group_size,
                          asym        = args.asym,
                          benchmarks  = args.sweep_benchmarks)
        return

    # ── Baseline table ────────────────────────────────────────────────────────
    print_baseline_table(args.n_prompt, args.n_decode,
                          args.quants, model, hw, args.group_size, args.asym, B)

    # ── Adaptive table ────────────────────────────────────────────────────────
    run_adaptive = (args.sim_json is not None
                    or args.acceptance_rate is not None
                    or args.draft_fraction is not None
                    or args.first_fail_frac is not None)

    if run_adaptive:
        if args.sim_json:
            stats = adaptive_stats_from_json(args.sim_json, args.draft)
            if args.acceptance_rate  is not None: stats.acceptance_rate    = args.acceptance_rate
            if args.draft_fraction   is not None: stats.draft_fraction      = args.draft_fraction
            if args.first_fail_frac  is not None: stats.first_fail_fraction = args.first_fail_frac
            if args.window_size      is not None: stats.window_size         = args.window_size
        else:
            stats = AdaptiveStats(
                acceptance_rate    = args.acceptance_rate    if args.acceptance_rate  is not None else 0.8,
                draft_fraction     = args.draft_fraction     if args.draft_fraction   is not None else 0.8,
                first_fail_fraction= args.first_fail_frac    if args.first_fail_frac  is not None else 0.8,
                window_size        = args.window_size,
            )

        print_adaptive_table(args.n_prompt, args.n_decode,
                              args.draft, args.verifier,
                              stats, model, hw, args.group_size, args.asym, B)


if __name__ == "__main__":
    main()
