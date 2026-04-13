#!/usr/bin/env python3
"""roofline_layer.py — Per-operation roofline model for transformer inference.

For each op in a transformer layer (QKV proj, flash attention, FFN, layer-norm,
KV read/write), computes FLOPs, memory bytes, arithmetic intensity, and latency.
Separate memory-bandwidth parameters for weights/activations vs KV cache, because
KV quantization specifically reduces attention bandwidth.

Three stages modelled:
  prefill   — n_prompt tokens processed in parallel (attention is causal, avg ctx n/2)
  decode    — 1 new token per step; reads KV cache (memory-bound, grows each step)
  verify    — W tokens (like short prefill but on top of existing KV cache)

Batch size:
  Weights are shared across the batch (no BW multiplier on weight ops).
  Activations and KV cache scale with batch_size.

Flash attention model:
  Standard: O(n²) intermediate attention matrix written to HBM.
  Flash:    tiled; O(n×d) HBM traffic; intensity ∝ r×n_q (r = GQA ratio).
            Compute-bound for n_q > ridge/r; memory-bound for short n_q (decode).

Usage:
    python3 research/scripts/roofline_layer.py \\
        --model qwen3-8b --hw a100-80g \\
        --n-prompt 4096 --n-decode 512 \\
        --kv-quant int4_ch --batch-size 1

    # Large-batch serving, long-context, separate attention BW tuning:
    python3 research/scripts/roofline_layer.py \\
        --model qwen3-8b --hw a100-80g \\
        --n-prompt 8192 --n-decode 2048 \\
        --kv-quant int3_half_1357_ch --batch-size 32 \\
        --attn-bw 1600 --padding-efficiency 0.80

    # Verify adaptive window (W=32 tokens on top of existing 1024-tok KV cache):
    python3 research/scripts/roofline_layer.py \\
        --model qwen3-8b --hw a100-80g \\
        --n-prompt 1024 --verify-window 32 \\
        --kv-quant int3_half_1357_ch

    # Adaptive draft: same parameter count as --model, but draft uses a more-compressed
    # GGUF (Q2_K / Q4_K_M / …). Tensor shapes are unchanged; only effective weight bpw differs.
    python3 research/scripts/roofline_layer.py \\
        --model qwen3-8b --hw h100-sxm --n-prompt 2048 --n-decode 4096 \\
        --adaptive --acceptance-rate 0.7 --adaptive-window 32 \\
        --main-gguf-quant Q8_0 --draft-gguf-quant Q2_K \\
        --draft-kv-quant int3_half_1357_ch --verify-kv-quant int4_ch

    # Mixed GGUF (non-uniform per tensor): build a profile from the GGUF, then:
    python3 research/scripts/gguf_roofline_weight_profile.py model.gguf -o /tmp/w.json
    python3 research/scripts/roofline_layer.py --model qwen3-8b --hw h100-sxm \\
        --n-prompt 2048 --n-decode 512 --kv-quant int4_ch \\
        --weight-bpw-profile /tmp/w.json

    # Calibrate to benchmark_kv_timing.py output (rows[] + measured_ms per scenario):
    python3 research/scripts/roofline_layer.py --model qwen3-8b --hw h100-sxm \\
        --n-prompt 2048 --n-decode 4096 --kv-quant fp16 --main-gguf-quant Q8_0 \\
        --calibration-json research/results/qwen3-8b/profile/kv_timing_h100.json

    # Build a report of measured vs analytic ms/tok for every row in downloaded profiles:
    python3 research/scripts/build_roofline_calibration_report.py \\
        -o research/results/roofline_real_calibration_report.json
"""

import argparse, json, math, os, sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

# ── Re-use presets from perf_model.py if available ───────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
try:
    from perf_model import MODEL_PRESETS, HARDWARE_PRESETS, QUANT_CONFIGS
except ImportError:
    # Inline minimal presets so the script is self-contained
    MODEL_PRESETS = {
        "qwen3-8b": dict(n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8,
                         head_dim=128, ffn_dim=11008, ffn_style="swiglu", weight_bits=16),
        "qwen3-14b": dict(n_layers=40, d_model=5120, n_heads=40, n_kv_heads=8,
                          head_dim=128, ffn_dim=17920, ffn_style="swiglu", weight_bits=16),
        "llama4-scout-17b": dict(n_layers=48, d_model=5120, n_heads=40, n_kv_heads=8,
                                 head_dim=128, ffn_dim=16384, ffn_style="swiglu", weight_bits=16),
        "gpt-oss-20b": dict(n_layers=40, d_model=5120, n_heads=40, n_kv_heads=8,
                            head_dim=128, ffn_dim=13824, ffn_style="swiglu", weight_bits=16),
        "qwen3.5-27b": dict(n_layers=62, d_model=7168, n_heads=56, n_kv_heads=8,
                            head_dim=128, ffn_dim=18944, ffn_style="swiglu", weight_bits=16),
    }
    HARDWARE_PRESETS = {
        "a100-80g":  dict(compute_tflops=312.0, memory_bw_gbps=2000.0, efficiency=0.70),
        "h100-sxm":  dict(compute_tflops=989.0, memory_bw_gbps=3350.0, efficiency=0.70),
        "gh200":     dict(compute_tflops=989.0, memory_bw_gbps=3350.0, efficiency=0.70),
        "a6000":     dict(compute_tflops=154.0, memory_bw_gbps=768.0,  efficiency=0.65),
        "rtx4090":   dict(compute_tflops=165.0, memory_bw_gbps=1008.0, efficiency=0.65),
    }
    QUANT_CONFIGS = {
        "fp16":              dict(bits=16, scale_bits=0,  group_size=1),
        "int8_ch":           dict(bits=8,  scale_bits=16, group_size=64),
        "int4_ch":           dict(bits=4,  scale_bits=16, group_size=64),
        "int3_ch":           dict(bits=3,  scale_bits=16, group_size=64),
        "int3_half_1357_ch": dict(bits=2,  scale_bits=16, group_size=64),
        "int2_ch":           dict(bits=2,  scale_bits=16, group_size=64),
    }

# Effective bits/weight for GGUF K-quants (same tensor shapes as FP16; only storage changes).
# Used for --main-gguf-quant / --draft-gguf-quant (adaptive draft = lighter GGUF, not fewer params).
GGUF_WEIGHT_QUANT_BPW = {
    "Q2_K":   3.1,
    "Q3_K_M": 4.0,
    "Q4_K_M": 4.9,
    "Q8_0":   8.5,
    "F16":    16.0,
}

# Roofline op names → logical bucket (mixed GGUF quants often differ attn vs MLP).
ROOFLINE_OP_TO_BUCKET = {
    "qkv_proj":    "attn",
    "out_proj":    "attn",
    "ffn_gate_up": "mlp",
    "ffn_down":    "mlp",
    "ffn_up":      "mlp",
}

WeightBpwSpec = Union[float, Dict[str, float]]


def _merge_weight_bpw_profile(scalar: float, path: Optional[str]) -> WeightBpwSpec:
    """Load JSON profile: keys attn, mlp, default, and/or per-op (qkv_proj, …). Adds default=scalar if missing."""
    if not path:
        return scalar
    p = os.path.expanduser(path)
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        out[str(k)] = float(v)
    if "default" not in out:
        out["default"] = float(scalar)
    return out


def _scalar_default_from_weight_spec(spec: WeightBpwSpec) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    return float(spec.get("default", 16.0))


def _resolve_weight_bpw(op_name: str, spec: WeightBpwSpec, *, fallback: float) -> float:
    """Effective bpw for one matmul op (bucket attn/mlp, per-op override, or scalar)."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if op_name in spec:
        return float(spec[op_name])
    bkey = ROOFLINE_OP_TO_BUCKET.get(op_name)
    if bkey and bkey in spec:
        return float(spec[bkey])
    if "default" in spec:
        return float(spec["default"])
    return float(fallback)


def _format_weight_bpw_spec(spec: WeightBpwSpec) -> str:
    if isinstance(spec, (int, float)):
        return f"{float(spec):.3f}"
    parts = [f"{k}={v:.3f}" for k, v in sorted(spec.items()) if not str(k).startswith("_")]
    return "{" + ", ".join(parts) + "}"


def _effective_draft_weight_spec(args, kw_base: dict, model: dict) -> WeightBpwSpec:
    """Adaptive draft path: optional draft profile; else --draft-weight-bpw / --draft-gguf-quant; else main spec."""
    main_bpw = kw_base.get("weight_bpw", float(model.get("weight_bits", 16)))
    dprof = getattr(args, "draft_weight_bpw_profile", None)
    if dprof:
        draft_default = (
            args.draft_weight_bpw if args.draft_weight_bpw is not None
            else _scalar_default_from_weight_spec(main_bpw))
        return _merge_weight_bpw_profile(float(draft_default), dprof)
    if args.draft_weight_bpw is not None:
        return float(args.draft_weight_bpw)
    return main_bpw


def _apply_gguf_weight_quant_args(args):
    """Set --weight-bpw / --draft-weight-bpw from GGUF quant tags when numeric not given."""
    if getattr(args, "main_gguf_quant", None):
        if args.weight_bpw is None:
            key = args.main_gguf_quant
            if key not in GGUF_WEIGHT_QUANT_BPW:
                raise ValueError(f"unknown --main-gguf-quant {key!r}")
            args.weight_bpw = GGUF_WEIGHT_QUANT_BPW[key]
    if getattr(args, "draft_gguf_quant", None):
        if args.draft_weight_bpw is None:
            key = args.draft_gguf_quant
            if key not in GGUF_WEIGHT_QUANT_BPW:
                raise ValueError(f"unknown --draft-gguf-quant {key!r}")
            args.draft_weight_bpw = GGUF_WEIGHT_QUANT_BPW[key]


# ── Core data structures ──────────────────────────────────────────────────────

@dataclass
class OpStats:
    name:        str
    flops:       float   # total FLOPs for this op across batch
    wt_bytes:    float   # weight bytes (shared across batch; no B multiplier)
    act_bytes:   float   # activation bytes (scales with B)
    intensity:   float   # FLOPs / total_bytes; inf if no memory access
    bound:       str     # "COMPUTE" or "memory"
    time_s:      float   # estimated time for this op (one layer)
    note:        str = ""
    kv_dependent: bool = False  # True for ops whose time scales with KV context length

    @property
    def total_bytes(self): return self.wt_bytes + self.act_bytes


def _kv_bytes_per_token_per_layer(kv_quant: str, n_kv_heads: int, head_dim: int,
                                   group_size: int = 64) -> float:
    """Bytes for K or V tensor, one token, one layer (K only; multiply by 2 for K+V)."""
    qcfg = QUANT_CONFIGS.get(kv_quant, QUANT_CONFIGS["fp16"])
    bits      = qcfg["bits"]
    scale_b   = qcfg["scale_bits"]
    gs        = group_size if group_size > 0 else qcfg.get("group_size", 64)
    data_b    = n_kv_heads * head_dim * (bits / 8)
    scale_oh  = n_kv_heads * head_dim * (scale_b / 8) / gs if scale_b > 0 and gs > 1 else 0.0
    return data_b + scale_oh   # per K or V tensor, one token, one layer


# ── Per-operation modelling ────────────────────────────────────────────────────

def ops_for_layer(stage: str,
                  n_q: int,           # query tokens this step (B already included)
                  n_ctx: int,         # KV context tokens PER SEQUENCE (not total)
                  model: dict,
                  compute_eff: float, # fraction of peak compute utilised
                  mem_eff: float,     # fraction of peak weight/act BW utilised
                  attn_eff: float,    # fraction of peak BW utilised for attention KV reads
                  compute_peak: float, # FLOPs/s
                  weight_bw: float,   # bytes/s for weight reads
                  attn_bw: float,     # bytes/s for attention KV reads (separate!)
                  act_bw: float,      # bytes/s for activation reads/writes
                  flash_attn: bool,
                  kv_quant: str,
                  kv_group_size: int,
                  padding_eff: float,  # 0.0–1.0; fraction of n_q that are real tokens
                  weight_bpw: WeightBpwSpec,  # scalar bpw or profile dict (attn/mlp/per-op)
                  weight_bpw_fallback: float = 16.0,
                  batch_size: int = 1,
                  act_bits: int = 16) -> list:
    """Return list of OpStats for one transformer layer at this stage.

    Memory model:
      - Weights: loaded once per step (shared across batch); BW = weight_bw
      - Activations (inputs/outputs): scale with n_q; BW = act_bw
      - Attention KV reads: scale with n_ctx × B; BW = attn_bw (affected by KV quant)

    Compute model:
      - FLOPs scale with n_q (or n_q × n_ctx for attention)
      - Effective peak = compute_peak × compute_eff
      - Weight matrix-vector ops: compute scales with n_q (benefits from large batch)
    """
    d     = model["d_model"]
    nh    = model["n_heads"]
    nkv   = model["n_kv_heads"]
    hd    = model["head_dim"]
    ffn   = model["ffn_dim"]
    style = model.get("ffn_style", "swiglu")
    abpe  = act_bits / 8       # activation bytes per element

    def _wbpe(op_name: str) -> float:
        return _resolve_weight_bpw(op_name, weight_bpw, fallback=weight_bpw_fallback) / 8.0

    # Effective compute and BW
    comp_peak  = compute_peak * compute_eff
    wt_bw_eff  = weight_bw  * mem_eff
    act_bw_eff = act_bw     * mem_eff
    kv_bw_eff  = attn_bw    * attn_eff

    # Real tokens in batch (after padding)
    n_real = n_q * padding_eff   # effective token count for compute/activations

    def _op(name, flops, wt_b, act_b, note="", attn_b=0.0, kv_dep=False):
        """Build one OpStats; time = max(compute, weight_bw, act_bw, attn_bw)."""
        total_b = wt_b + act_b + attn_b
        intens  = flops / total_b if total_b > 0 else float("inf")
        t_comp  = flops / comp_peak     if flops  > 0 else 0.0
        t_wt    = wt_b  / wt_bw_eff    if wt_b   > 0 else 0.0
        t_act   = act_b / act_bw_eff   if act_b  > 0 else 0.0
        t_attn  = attn_b / kv_bw_eff   if attn_b > 0 else 0.0
        t_total = max(t_comp, t_wt, t_act, t_attn)
        bottlenecks = []
        if t_total > 0:
            if t_comp  >= t_total * 0.95: bottlenecks.append("COMPUTE")
            if t_wt    >= t_total * 0.95: bottlenecks.append("wt-BW")
            if t_act   >= t_total * 0.95: bottlenecks.append("act-BW")
            if t_attn  >= t_total * 0.95: bottlenecks.append("attn-BW")
        bound = "+".join(bottlenecks) if bottlenecks else "idle"
        return OpStats(name=name, flops=flops, wt_bytes=wt_b, act_bytes=act_b + attn_b,
                       intensity=intens, bound=bound, time_s=t_total, note=note,
                       kv_dependent=kv_dep)

    ops = []

    # ── 1. RMSNorm (pre-attention) ────────────────────────────────────────────
    # ~5 FLOPs/element: square, mean, rsqrt, scale, add
    ln_flops = 5 * n_real * d
    ln_act   = 2 * n_q * d * abpe    # read input + write output
    ops.append(_op("rmsnorm_attn", ln_flops, 0, ln_act))

    # ── 2. QKV projection ─────────────────────────────────────────────────────
    # Q: d→nh×hd,  K: d→nkv×hd,  V: d→nkv×hd
    # With GQA the weight matrix is d × (nh+2·nkv)·hd
    qkv_out_dim = nh * hd + 2 * nkv * hd
    qkv_flops   = 2 * n_real * d * qkv_out_dim
    qkv_wt      = d * qkv_out_dim * _wbpe("qkv_proj")            # weight loaded once (shared)
    qkv_act     = n_q * d * abpe + n_q * qkv_out_dim * abpe   # in + out (scale with B)
    ops.append(_op("qkv_proj", qkv_flops, qkv_wt, qkv_act))

    # ── 3. RoPE ───────────────────────────────────────────────────────────────
    # ~6 FLOPs/element (cos/sin + rotate); applied to Q and K only
    rope_flops = 6 * n_real * (nh + nkv) * hd
    rope_act   = 2 * n_q * (nh + nkv) * hd * abpe
    ops.append(_op("rope", rope_flops, 0, rope_act))

    # ── 4. KV cache write (prefill / verify) ─────────────────────────────────
    if stage in ("prefill", "verify"):
        kv_bpt = _kv_bytes_per_token_per_layer(kv_quant, nkv, hd, kv_group_size)
        kv_write_act = n_q * 2 * kv_bpt    # write K and V (quantized)
        ops.append(_op("kv_write", 0, 0, kv_write_act,
                        note=f"write {kv_quant}", kv_dep=True))

    # ── 5. Attention ──────────────────────────────────────────────────────────
    # For causal prefill, average context per query ≈ n_ctx/2.
    # For decode, each query attends to all n_ctx tokens.
    # For verify, each query attends to all n_ctx tokens (full KV cache).
    if stage == "prefill":
        avg_attended = n_ctx / 2   # causal mask: position i sees positions 0..i
    else:
        avg_attended = n_ctx       # decode/verify: see all existing KV

    # FLOPs: QK scores + softmax×V; per head, per query: 4×n_ctx×hd FLOPs
    # (factor-2 for multiply-add, ×2 for QK and for V)
    attn_flops = 4 * n_real * nh * avg_attended * hd

    # Each sequence in the batch reads its own KV cache independently.
    B = batch_size

    if flash_attn:
        # Flash attention reads Q,K,V from HBM once (tiled, no O(n²) intermediate).
        # Q: n_q × nh × hd (activation BW)
        # K,V: B × n_ctx × nkv × hd per K and V (each seq reads its own KV)
        # Output: n_q × nh × hd (activation BW)
        kv_bpt     = _kv_bytes_per_token_per_layer(kv_quant, nkv, hd, kv_group_size)
        attn_q_b   = n_q * nh * hd * abpe         # Q in act BW
        attn_out_b = n_q * nh * hd * abpe         # output in act BW
        attn_kv_b  = B * n_ctx * 2 * kv_bpt       # K+V in attn BW (B sequences)
        note = f"flash,ctx={n_ctx}"
        ops.append(_op("attention_flash", attn_flops, 0,
                        attn_q_b + attn_out_b, note, attn_b=attn_kv_b,
                        kv_dep=True))
    else:
        # Standard attention: write + read O(n²) attention weight matrix
        attn_mat_b = n_q * nh * avg_attended * 4   # fp32 attn weights (write+read)
        kv_bpt     = _kv_bytes_per_token_per_layer(kv_quant, nkv, hd, kv_group_size)
        attn_q_b   = n_q * nh * hd * abpe
        attn_out_b = n_q * nh * hd * abpe
        attn_kv_b  = B * n_ctx * 2 * kv_bpt
        note = f"standard,ctx={n_ctx}"
        ops.append(_op("attention_std", attn_flops, 0,
                        attn_q_b + attn_out_b + attn_mat_b, note, attn_b=attn_kv_b,
                        kv_dep=True))

    # ── 6. Output projection ─────────────────────────────────────────────────
    out_flops = 2 * n_real * nh * hd * d
    out_wt    = nh * hd * d * _wbpe("out_proj")
    out_act   = n_q * nh * hd * abpe + n_q * d * abpe
    ops.append(_op("out_proj", out_flops, out_wt, out_act))

    # ── 7. RMSNorm (pre-FFN) ─────────────────────────────────────────────────
    ops.append(_op("rmsnorm_ffn", ln_flops, 0, ln_act))

    # ── 8. FFN ───────────────────────────────────────────────────────────────
    if style == "swiglu":
        # gate: d→ffn_dim,  up: d→ffn_dim,  act: SiLU(gate)⊙up,  down: ffn_dim→d
        ffn_flops_gu = 2 * 2 * n_real * d * ffn      # gate + up (2 matmuls)
        ffn_wt_gu    = 2 * d * ffn * _wbpe("ffn_gate_up")
        ffn_act_gu   = n_q * d * abpe + n_q * ffn * abpe
        ffn_flops_dn = 2 * n_real * ffn * d
        ffn_wt_dn    = ffn * d * _wbpe("ffn_down")
        ffn_act_dn   = n_q * ffn * abpe + n_q * d * abpe
        ops.append(_op("ffn_gate_up", ffn_flops_gu, ffn_wt_gu, ffn_act_gu))
        ops.append(_op("ffn_down",    ffn_flops_dn, ffn_wt_dn, ffn_act_dn))
    else:
        # Standard FFN: up + down
        ffn_flops_up = 2 * n_real * d * ffn
        ffn_wt_up    = d * ffn * _wbpe("ffn_up")
        ffn_act_up   = n_q * d * abpe + n_q * ffn * abpe
        ffn_flops_dn = 2 * n_real * ffn * d
        ffn_wt_dn    = ffn * d * _wbpe("ffn_down")
        ffn_act_dn   = n_q * ffn * abpe + n_q * d * abpe
        ops.append(_op("ffn_up",   ffn_flops_up, ffn_wt_up, ffn_act_up))
        ops.append(_op("ffn_down", ffn_flops_dn, ffn_wt_dn, ffn_act_dn))

    return ops


# ── Reporting ──────────────────────────────────────────────────────────────────

def _si(x, unit=""):
    """Format large numbers with SI suffixes."""
    for pfx, scale in [("T",1e12),("G",1e9),("M",1e6),("K",1e3)]:
        if abs(x) >= scale:
            return f"{x/scale:.2f}{pfx}{unit}"
    return f"{x:.2f}{unit}"


def print_stage_table(stage_label: str, ops_per_layer: list, n_layers: int,
                      n_q_total: int, stage: str, extra_info: str = ""):
    """Print per-op breakdown (summed over all layers) for one stage."""
    W = 105
    print(f"\n{'═'*W}")
    print(f"  {stage_label}  {extra_info}")
    print(f"{'─'*W}")
    hdr = (f"  {'op':<20} {'FLOPs':>9} {'wt_MB':>8} {'act_MB':>8} "
           f"{'tot_MB':>8} {'intens':>8} {'bound':<18} {'time_ms':>9} {'%':>5}")
    print(hdr)
    print(f"{'─'*W}")

    total_flops = 0.0
    total_wt    = 0.0
    total_act   = 0.0
    total_time  = 0.0

    op_rows = []
    for op in ops_per_layer:
        f   = op.flops    * n_layers
        wb  = op.wt_bytes * n_layers
        ab  = op.act_bytes * n_layers
        tb  = wb + ab
        t   = op.time_s   * n_layers
        total_flops += f
        total_wt    += wb
        total_act   += ab
        total_time  += t
        op_rows.append((op.name, f, wb, ab, tb, op.intensity, op.bound, t))

    for name, f, wb, ab, tb, intens, bound, t in op_rows:
        pct = 100 * t / total_time if total_time > 0 else 0.0
        i_str = f"{intens:.1f}" if not math.isinf(intens) else "∞"
        print(f"  {name:<20} {_si(f,'F'):>9} {wb/1e6:>8.2f} {ab/1e6:>8.2f} "
              f"{(wb+ab)/1e6:>8.2f} {i_str:>8} {bound:<18} {t*1e3:>9.3f} {pct:>5.1f}%")

    print(f"{'─'*W}")
    pct_cmp = sum(t for _,_,_,_,_,_,b,t in op_rows if "COMPUTE" in b) / total_time * 100 if total_time > 0 else 0
    pct_mem = 100 - pct_cmp
    print(f"  {'TOTAL':<20} {_si(total_flops,'F'):>9} {total_wt/1e6:>8.2f} "
          f"{total_act/1e6:>8.2f} {(total_wt+total_act)/1e6:>8.2f} "
          f"{'':>8} {'':18} {total_time*1e3:>9.3f} {'100%':>5}")
    print(f"  compute={pct_cmp:.0f}%  memory={pct_mem:.0f}%  "
          f"arithmetic_intensity={total_flops/(total_wt+total_act):.1f} FLOPs/B")
    return total_time


def _roofline_kv_quant_to_native(kv_quant: str) -> str:
    """Map research QUANT_CONFIGS name to benchmark_kv_timing native KV type string."""
    return {
        "fp16":              "f16",
        "int8_ch":           "q8_0",
        "int4_ch":           "q4_0",
        "int3_ch":           "q4_0",
        "int3_half_1357_ch": "q4_0",
        "int2_ch":           "q4_0",
    }.get(kv_quant, "f16")


def _calibration_hw_matches(cal_hw: Optional[str], args_hw: str) -> bool:
    if not cal_hw:
        return True
    c = str(cal_hw).lower().split("-")[0].split("_")[0]
    a = str(args_hw).lower().split("-")[0].split("_")[0]
    return c == a


def _infer_calibration_weight_tag(args) -> Optional[str]:
    if getattr(args, "calibration_weight_tag", None):
        return str(args.calibration_weight_tag).strip()
    if getattr(args, "main_gguf_quant", None):
        return args.main_gguf_quant
    return None


def _infer_calibration_kv_type(args) -> str:
    if getattr(args, "calibration_kv_type", None):
        return str(args.calibration_kv_type).strip().lower()
    return _roofline_kv_quant_to_native(args.kv_quant)


def _bootstrap_ms_lookup(cal: dict, weight_tag: str, kv_type: str,
                         batch: int, prompt: int, decode: int) -> Optional[float]:
    bq = cal.get("bootstrap_ms_quant")
    if not isinstance(bq, dict):
        return None
    key = f"{weight_tag} {kv_type} B{batch} p{prompt} d{decode}"
    v = bq.get(key)
    return float(v) if v is not None else None


def _roofline_avg_ctx_tokens(args) -> int:
    """Same midpoint as decode table: prompt + decode/2."""
    return int(args.n_prompt) + int(args.n_decode) // 2


def _find_kv_timing_row(
        cal: dict, args,
        weight_tag: str, kv_type: str) -> Tuple[Optional[dict], str]:
    """Return (row or None, reason string for logging)."""
    rows = cal.get("rows")
    if not isinstance(rows, list):
        return None, "no rows[]"
    relax = getattr(args, "calibration_relax_decode_len", False)

    def _row_ok_base(r: dict) -> bool:
        if r.get("model_preset") != args.model:
            return False
        if int(r.get("batch_size", -1)) != int(args.batch_size):
            return False
        if int(r.get("prompt_len", -1)) != int(args.n_prompt):
            return False
        if r.get("weight_tag") != weight_tag:
            return False
        if str(r.get("kv_type", "")).lower() != kv_type.lower():
            return False
        return True

    for r in rows:
        if not isinstance(r, dict):
            continue
        if not _row_ok_base(r):
            continue
        dl = int(r.get("decode_len", -1))
        if dl != int(args.n_decode):
            continue
        return r, "exact row match (decode_len match)"

    if relax:
        candidates = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            if not _row_ok_base(r):
                continue
            dl = int(r.get("decode_len", 0))
            if dl >= int(args.n_decode):
                candidates.append((dl, r))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            r = candidates[0][1]
            return r, (
                f"relaxed decode_len: using row decode_len={r.get('decode_len')} "
                f"(>= --n-decode={args.n_decode}); use decode_buckets for measured ms/tok")

    # Relaxed: same model, B, prompt, decode, weight — any kv
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("model_preset") != args.model:
            continue
        if int(r.get("batch_size", -1)) != int(args.batch_size):
            continue
        if int(r.get("prompt_len", -1)) != int(args.n_prompt):
            continue
        if int(r.get("decode_len", -1)) != int(args.n_decode):
            continue
        if r.get("weight_tag") != weight_tag:
            continue
        return r, "relaxed match (kv_type differed; using measured row)"
    return None, "no matching row"


def _measured_ms_from_kv_timing_row(row: dict, args) -> Tuple[float, str]:
    """Use row mean or decode_buckets[].ms_per_token closest to roofline avg_ctx."""
    policy = getattr(args, "calibration_bucket_policy", "match_ctx")
    mean_ms = float(row["measured_ms"])
    if policy == "mean":
        return mean_ms, "row measured_ms (mean over full timed decode)"

    buckets = row.get("decode_buckets")
    if not isinstance(buckets, list) or len(buckets) == 0:
        return mean_ms, "no decode_buckets in row; using row measured_ms"

    target = float(_roofline_avg_ctx_tokens(args))
    best = min(
        buckets,
        key=lambda b: abs(float(b.get("ctx_mid", 0)) - target))
    ctx = float(best.get("ctx_mid", 0))
    ms = float(best.get("ms_per_token", mean_ms))
    return ms, (
        f"decode_bucket ctx_mid={ctx:.0f} (target avg_ctx≈{target:.0f}); "
        f"decode[{best.get('decode_from')},{best.get('decode_to')}] "
        f"(same data as kv_timing_*.out decode buckets)")


def _print_calibration(model, args, kw, weight_bpw, results):
    """Scale decode times to measured cluster: T_scaled = T_roofline × scale.

    Supports:
      • Legacy JSON: decode_ms_per_token_fp16_baseline (or decode_ms_per_token) vs fp16 roofline ref.
      • Optional decode_scale_measured_over_roofline — applied to this run's roofline decode.
      • benchmark_kv_timing output: rows[] + bootstrap_ms_quant — measured_ms vs this run's roofline ms/tok.
      • Rows may include decode_buckets (same information as kv_timing_*.out “decode buckets”): with
        --calibration-bucket-policy match_ctx, use ms/tok from the bucket whose ctx_mid is closest to
        roofline avg_ctx (n_prompt + n_decode//2). Use --calibration-relax-decode-len to select a row
        whose decode_len is >= --n-decode when your timed run was shorter than the cluster sweep.
    """
    path = os.path.expanduser(args.calibration_json)
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)

    roof_ms_this = (
        (results["decode_total"] / args.n_decode) * 1000.0
        if "decode_total" in results and args.n_decode > 0 else 0.0)

    mode = "legacy_fp16_ref"
    measured_ms: Optional[float] = None
    scale: float = 1.0
    note = ""
    roof_ms_fp16: Optional[float] = None

    # 1) Explicit scale (from hand-tuned or precomputed summary JSON)
    if cal.get("decode_scale_measured_over_roofline") is not None:
        scale = float(cal["decode_scale_measured_over_roofline"])
        measured_ms = roof_ms_this * scale if roof_ms_this > 0 else None
        mode = "decode_scale_measured_over_roofline"
        note = "scale from JSON key decode_scale_measured_over_roofline"

    # 2) Full kv_timing profile (benchmark_kv_timing.py)
    elif isinstance(cal.get("rows"), list):
        wtag = _infer_calibration_weight_tag(args)
        kv_t = _infer_calibration_kv_type(args)
        if not wtag:
            wtag = "Q8_0"
            note = "calibration_weight_tag not set; defaulting weight_tag=Q8_0 for row lookup"
        row, match_reason = _find_kv_timing_row(cal, args, wtag, kv_t)
        if row is not None:
            measured_ms, bnote = _measured_ms_from_kv_timing_row(row, args)
            mode = "kv_timing_row"
            note = match_reason + "  |  " + bnote
        else:
            measured_ms = _bootstrap_ms_lookup(cal, wtag, kv_t, args.batch_size,
                                                 args.n_prompt, args.n_decode)
            if measured_ms is not None:
                mode = "kv_timing_bootstrap"
                note = f"bootstrap_ms_quant key ({wtag} {kv_t} B{args.batch_size} …)"
            else:
                raise ValueError(
                    f"kv_timing calibration: no row for model={args.model} B={args.batch_size} "
                    f"p={args.n_prompt} d={args.n_decode} weight_tag={wtag} kv_type={kv_t}; "
                    f"set --calibration-weight-tag / --calibration-kv-type or use legacy JSON. "
                    f"Lookup hint: {match_reason}")

        if roof_ms_this > 0 and measured_ms is not None:
            scale = measured_ms / roof_ms_this
        else:
            scale = 1.0

    # 3) Legacy flat calibration file
    else:
        ms = cal.get("decode_ms_per_token_fp16_baseline", cal.get("decode_ms_per_token"))
        if ms is None:
            raise ValueError(
                "calibration JSON needs decode_ms_per_token_fp16_baseline, decode_ms_per_token, "
                "decode_scale_measured_over_roofline, or a kv_timing file with rows[]")
        measured_ms = float(ms)
        T_ref = _decode_total_time(model, args, kw, 16.0, "fp16")
        roof_ms_fp16 = T_ref / args.n_decode * 1000.0
        # Prefer stored recomputed roofline when present (matches how scale was computed on cluster)
        if cal.get("roofline_decode_ms_per_token_fp16_recomputed") is not None:
            roof_ms_fp16 = float(cal["roofline_decode_ms_per_token_fp16_recomputed"])
        scale = measured_ms / roof_ms_fp16 if roof_ms_fp16 > 0 else 1.0
        mode = "legacy_ms_vs_fp16_roofline"
        note = "measured_ms vs fp16+fp16 roofline reference (see JSON _comment)"

    print(f"\n{'─'*105}")
    print(f"  CALIBRATION  ({path})  mode={mode}")
    if isinstance(cal.get("rows"), list) and not _calibration_hw_matches(cal.get("hardware"), args.hw):
        print(f"    WARNING: profile hardware={cal.get('hardware')!r} vs --hw={args.hw!r}")
    if note:
        print(f"    {note}")
    if (mode == "kv_timing_row"
            and getattr(args, "calibration_bucket_policy", "") == "match_ctx"):
        print(f"    Roofline avg_ctx (n_prompt + n_decode/2): {_roofline_avg_ctx_tokens(args)}")
    if measured_ms is not None:
        print(f"    Measured ms/tok (cluster / profile): {measured_ms:.6f}")
    if mode == "legacy_ms_vs_fp16_roofline" and roof_ms_fp16 is not None:
        print(f"    Roofline ref ms/tok (fp16 weights + fp16 KV, same n_prompt/n_decode): "
              f"{roof_ms_fp16:.6f}")
        print(f"    decode_scale = measured / ref_fp16 = {scale:.6f}")
    elif mode == "decode_scale_measured_over_roofline":
        print(f"    Roofline ms/tok (this run): {roof_ms_this:.6f}")
        print(f"    decode_scale = {scale:.6f}  (from JSON; × roofline_this → absolute tok/s)")
    else:
        print(f"    Roofline ms/tok (this run): {roof_ms_this:.6f}")
        print(f"    decode_scale = measured / roofline_this = {scale:.6f}")
    if "decode_total" in results:
        Td = results["decode_total"] * scale
        print(f"    Scaled decode (this run's kv/weight): {Td*1e3:.1f} ms  →  "
              f"{args.n_decode/Td:.1f} tok/s")
    if "prefill" in results and "decode_total" in results:
        tot = results["prefill"] + results["decode_total"] * scale
        print(f"    Scaled prefill+decode: {tot*1e3:.1f} ms  →  "
              f"{args.batch_size * args.n_decode/tot:.0f} tok/s sys (B={args.batch_size})")
    print(f"{'─'*105}")


def roofline_decode_total_seconds(model: dict, hw: dict, args) -> float:
    """Uncorrected analytic total decode time (seconds) for args.n_decode steps. No printing."""
    if args.n_decode <= 0:
        return 0.0
    compute_peak = hw["compute_tflops"] * 1e12
    base_bw      = hw["memory_bw_gbps"] * 1e9
    weight_bw    = base_bw * args.weight_bw_fraction
    act_bw       = base_bw * args.act_bw_fraction
    attn_bw      = args.attn_bw * 1e9 if args.attn_bw else base_bw * args.attn_bw_fraction
    B = args.batch_size
    base_scalar = (
        args.weight_bpw if args.weight_bpw is not None
        else float(model.get("weight_bits", 16)))
    weight_bpw = _merge_weight_bpw_profile(
        base_scalar, getattr(args, "weight_bpw_profile", None))
    kw = dict(
        model=model,
        compute_eff=args.compute_eff,
        mem_eff=args.mem_eff,
        attn_eff=args.attn_eff,
        compute_peak=compute_peak,
        weight_bw=weight_bw,
        attn_bw=attn_bw,
        act_bw=act_bw,
        flash_attn=args.flash_attn,
        kv_quant=args.kv_quant,
        kv_group_size=args.kv_group_size,
        padding_eff=args.padding_efficiency,
        weight_bpw=weight_bpw,
        weight_bpw_fallback=base_scalar,
        batch_size=B,
    )
    nl = model["n_layers"]
    avg_ctx = args.n_prompt + args.n_decode // 2
    n_q_dec = B * 1
    ops_dec = ops_for_layer("decode", n_q_dec, avg_ctx, **kw)
    T_fixed = sum(op.time_s * nl for op in ops_dec if not op.kv_dependent)
    T_kv = sum(op.time_s * nl for op in ops_dec if op.kv_dependent)
    return T_fixed * args.n_decode + T_kv * args.n_decode


def roofline_decode_ms_per_token(model: dict, hw: dict, args) -> float:
    """Roofline decode ms/token (uncorrected) matching the printed LATENCY SUMMARY decode line."""
    t = roofline_decode_total_seconds(model, hw, args)
    if args.n_decode <= 0 or t <= 0:
        return 0.0
    return t / float(args.n_decode) * 1000.0


def roofline_decode_ms_at_ctx(model: dict, hw: dict, args, ctx: int) -> float:
    """Single-step roofline ms/token at a specific KV context length.

    Unlike roofline_decode_ms_per_token (which uses avg_ctx = n_prompt + n_decode//2),
    this evaluates one decode step with n_ctx=ctx — useful for comparing to individual
    decode_buckets from benchmark_kv_timing where each bucket has its own ctx_mid.

    Returns ms per token (wall-clock per step / batch_size).
    """
    if ctx <= 0:
        return 0.0
    compute_peak = hw["compute_tflops"] * 1e12
    base_bw      = hw["memory_bw_gbps"] * 1e9
    weight_bw    = base_bw * args.weight_bw_fraction
    act_bw       = base_bw * args.act_bw_fraction
    attn_bw      = args.attn_bw * 1e9 if args.attn_bw else base_bw * args.attn_bw_fraction
    B = args.batch_size
    base_scalar = (
        args.weight_bpw if args.weight_bpw is not None
        else float(model.get("weight_bits", 16)))
    weight_bpw = _merge_weight_bpw_profile(
        base_scalar, getattr(args, "weight_bpw_profile", None))
    kw = dict(
        model=model,
        compute_eff=args.compute_eff,
        mem_eff=args.mem_eff,
        attn_eff=args.attn_eff,
        compute_peak=compute_peak,
        weight_bw=weight_bw,
        attn_bw=attn_bw,
        act_bw=act_bw,
        flash_attn=args.flash_attn,
        kv_quant=args.kv_quant,
        kv_group_size=args.kv_group_size,
        padding_eff=args.padding_efficiency,
        weight_bpw=weight_bpw,
        weight_bpw_fallback=base_scalar,
        batch_size=B,
    )
    nl = model["n_layers"]
    ops = ops_for_layer("decode", B * 1, ctx, **kw)
    T_step = sum(op.time_s * nl for op in ops)   # seconds for one decode step
    return T_step / B * 1000.0   # ms per token


def analyze_and_print(model, hw, args):
    """Run full analysis for all three stages and print summary."""
    # Build effective bandwidths
    compute_peak = hw["compute_tflops"] * 1e12
    base_bw      = hw["memory_bw_gbps"] * 1e9
    weight_bw    = base_bw * args.weight_bw_fraction
    act_bw       = base_bw * args.act_bw_fraction
    attn_bw      = args.attn_bw * 1e9 if args.attn_bw else base_bw * args.attn_bw_fraction
    ridge        = compute_peak * args.compute_eff / (base_bw * args.mem_eff)

    nl = model["n_layers"]
    B  = args.batch_size

    # Shared kwargs for ops_for_layer (weight_bpw may be scalar or attn/mlp/per-op profile)
    base_scalar = (
        args.weight_bpw if args.weight_bpw is not None
        else float(model.get("weight_bits", 16)))
    weight_bpw = _merge_weight_bpw_profile(
        base_scalar, getattr(args, "weight_bpw_profile", None))
    kw = dict(
        model                = model,
        compute_eff          = args.compute_eff,
        mem_eff              = args.mem_eff,
        attn_eff             = args.attn_eff,
        compute_peak         = compute_peak,
        weight_bw            = weight_bw,
        attn_bw              = attn_bw,
        act_bw               = act_bw,
        flash_attn           = args.flash_attn,
        kv_quant             = args.kv_quant,
        kv_group_size        = args.kv_group_size,
        padding_eff          = args.padding_efficiency,
        weight_bpw           = weight_bpw,
        weight_bpw_fallback  = base_scalar,
        batch_size           = B,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*105}")
    print(f"  Layer-level roofline model  |  model={args.model}  hw={args.hw}  "
          f"batch={B}  kv_quant={args.kv_quant}  weight_bpw={_format_weight_bpw_spec(weight_bpw)}")
    attn_mode = "flash" if args.flash_attn else "standard"
    attn_bw_label = f"{attn_bw/1e9:.0f} GB/s" + (" (separate)" if args.attn_bw else "")
    print(f"  weight_bw={weight_bw/1e9:.0f} GB/s  attn_bw={attn_bw_label}  "
          f"act_bw={act_bw/1e9:.0f} GB/s  "
          f"compute_eff={args.compute_eff:.0%}  mem_eff={args.mem_eff:.0%}  "
          f"attn_eff={args.attn_eff:.0%}  pad={args.padding_efficiency:.0%}")
    print(f"  attn={attn_mode}  ridge={ridge:.0f} FLOPs/B")

    results = {}

    # ── Prefill ───────────────────────────────────────────────────────────────
    if args.n_prompt > 0:
        n_q   = B * args.n_prompt
        n_ctx = args.n_prompt
        ops   = ops_for_layer("prefill", n_q, n_ctx, **kw)
        info  = (f"B={B}  n_tokens={args.n_prompt}  "
                 f"causal_avg_ctx≈{n_ctx//2}")
        T_pre = print_stage_table(f"PREFILL", ops, nl, n_q, "prefill", info)
        tps   = args.n_prompt / T_pre if T_pre > 0 else float("inf")
        print(f"  → Prefill latency: {T_pre*1e3:.1f} ms  |  "
              f"throughput: {tps:.0f} tok/s (prefill prompt)")
        results["prefill"] = T_pre

    # ── Decode: one token at avg context ──────────────────────────────────────
    if args.n_decode > 0:
        avg_ctx  = args.n_prompt + args.n_decode // 2
        n_q_dec  = B * 1              # one new token per sequence per step
        ops_dec  = ops_for_layer("decode", n_q_dec, avg_ctx, **kw)
        info     = (f"B={B}  n_q=1/seq  avg_ctx={avg_ctx}  "
                    f"(prompt={args.n_prompt}+decode/2={args.n_decode//2})")
        T_one    = print_stage_table("DECODE (one step, avg ctx)", ops_dec, nl,
                                      n_q_dec, "decode", info)

        # Total decode over n_decode steps: KV cache grows each step
        # T_total = Σ_{i=0}^{n_decode-1} T_step(i); weight ops are constant,
        # KV/attn ops scale with (n_prompt + i).
        # Approximation: split into weight-dominated part (constant) and
        # KV-dominated part (linear in context).
        T_fixed_per_step = sum(op.time_s * nl for op in ops_dec if not op.kv_dependent)
        T_kv_per_step    = sum(op.time_s * nl for op in ops_dec if op.kv_dependent)
        # KV-dependent ops scale linearly with context length.  The per-step
        # time was computed at avg_ctx; summing over n_decode steps where
        # context grows from n_prompt to n_prompt+n_decode-1 gives the same
        # total as n_decode × T_avg (by construction: avg_ctx = n_prompt + N/2).
        T_decode_total  = T_fixed_per_step * args.n_decode + T_kv_per_step * args.n_decode
        T_total_e2e     = results.get("prefill", 0.0) + T_decode_total
        seq_tps         = args.n_decode / T_decode_total if T_decode_total > 0 else float("inf")
        sys_tps         = B * args.n_decode / T_total_e2e if T_total_e2e > 0 else float("inf")
        print(f"  → Decode latency (×{args.n_decode} steps): {T_decode_total*1e3:.1f} ms  "
              f"({args.n_decode/T_decode_total:.1f} tok/s per seq)")
        print(f"  → End-to-end  (prefill + decode):  {T_total_e2e*1e3:.1f} ms  "
              f"sys_throughput={sys_tps:.0f} tok/s  (B={B})")
        results["decode_total"] = T_decode_total
        results["decode_per_step_avg"] = T_one

    # ── Verification window ───────────────────────────────────────────────────
    if args.verify_window and args.verify_window > 0:
        W_v    = args.verify_window
        # KV cache at verification point: n_prompt + first_fail_pos ≈ n_prompt + n_decode×first_fail_frac
        v_ctx  = args.n_prompt + int(args.n_decode * args.verify_ctx_frac)
        n_q_v  = B * W_v
        ops_v  = ops_for_layer("verify", n_q_v, v_ctx + W_v, **kw)
        info   = (f"B={B}  W={W_v} tokens  n_ctx={v_ctx+W_v}  "
                  f"(existing KV={v_ctx} + verify window)")
        T_ver  = print_stage_table(f"VERIFY WINDOW (W={W_v})", ops_v, nl,
                                    n_q_v, "verify", info)
        ver_tps = W_v / T_ver if T_ver > 0 else float("inf")
        overhead_vs_decode = T_ver / results.get("decode_per_step_avg", T_ver)
        print(f"  → Verify latency (W={W_v}): {T_ver*1e3:.1f} ms  "
              f"|  {T_ver / (T_ver + T_ver):.0%} overhead if sequential  "
              f"|  {W_v/T_ver:.0f} tok/s")
        print(f"  → Cost = {overhead_vs_decode:.1f}× single decode step")
        results["verify"] = T_ver

    # ── Adaptive decode analysis ────────────────────────────────────────────
    if args.adaptive and args.n_decode > 0:
        _print_adaptive_analysis(model, hw, args, kw, results)

    if getattr(args, "calibration_json", None) and args.n_decode > 0:
        _print_calibration(model, args, kw, weight_bpw, results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*105}")
    print(f"  LATENCY SUMMARY")
    print(f"{'─'*105}")
    if "prefill" in results:
        print(f"  Prefill  ({args.n_prompt:>6} tok):  {results['prefill']*1e3:>8.1f} ms")
    if "decode_total" in results:
        print(f"  Decode   ({args.n_decode:>6} tok):  {results['decode_total']*1e3:>8.1f} ms  "
              f"({args.n_decode / results['decode_total']:.1f} tok/s/seq)")
    if "verify" in results:
        print(f"  Verify 1× (W={args.verify_window:>4}):  "
              f"{results['verify']*1e3:>8.1f} ms")
    if "prefill" in results and "decode_total" in results:
        tot = results["prefill"] + results["decode_total"]
        sys = B * args.n_decode / tot
        print(f"  Total (prefill+decode):         {tot*1e3:>8.1f} ms  "
              f"sys_throughput={sys:.0f} tok/s (B={B})")
    print(f"{'═'*105}\n")


# ── Adaptive decode model ─────────────────────────────────────────────────────

def _decode_step_time(n_q, n_ctx, kv_quant, model, kw_base, weight_bpw=None):
    """Compute one decode step time (all layers) for a given KV quant."""
    kw = dict(kw_base, kv_quant=kv_quant)
    if weight_bpw is not None:
        kw = dict(kw, weight_bpw=weight_bpw)
    ops = ops_for_layer("decode", n_q, n_ctx, **kw)
    nl = model["n_layers"]
    T_fixed = sum(op.time_s * nl for op in ops if not op.kv_dependent)
    T_kv    = sum(op.time_s * nl for op in ops if op.kv_dependent)
    return T_fixed, T_kv


def _verify_window_time(B, W, n_ctx, verify_quant, model, kw_base, weight_bpw=None):
    """Cost of re-running W tokens at verifier precision (like a short prefill).
    batch_size is already in kw_base from the caller."""
    kw = dict(kw_base, kv_quant=verify_quant)
    if weight_bpw is not None:
        kw = dict(kw, weight_bpw=weight_bpw)
    n_q_v = B * W
    ops = ops_for_layer("verify", n_q_v, n_ctx + W, **kw)
    nl = model["n_layers"]
    return sum(op.time_s * nl for op in ops)


def _decode_total_time(model, args, kw, weight_bpw: WeightBpwSpec, kv_quant: str) -> float:
    """Total multi-step decode time using avg_ctx approximation (matches main decode block)."""
    avg_ctx = args.n_prompt + args.n_decode // 2
    n_q_dec = args.batch_size * 1
    kw2 = dict(kw, kv_quant=kv_quant, weight_bpw=weight_bpw)
    ops_dec = ops_for_layer("decode", n_q_dec, avg_ctx, **kw2)
    nl = model["n_layers"]
    T_fixed_per_step = sum(op.time_s * nl for op in ops_dec if not op.kv_dependent)
    T_kv_per_step    = sum(op.time_s * nl for op in ops_dec if op.kv_dependent)
    return T_fixed_per_step * args.n_decode + T_kv_per_step * args.n_decode


def _adaptive_decode_acceptance(
        W: int, n_dec: int, T_draft_step: float, T_verify_total: float,
        T_fp16_step: float, accept_rate: float) -> float:
    """Expected decode time with per-window acceptance (reject → fp16 fallback for that window)."""
    if W <= 0 or n_dec <= 0:
        return 0.0

    def _win(wtok: int) -> float:
        return (wtok * T_draft_step + T_verify_total
                + (1.0 - accept_rate) * wtok * T_fp16_step)

    n_full = n_dec // W
    n_rem  = n_dec % W
    T = n_full * _win(W)
    if n_rem > 0:
        T += _win(n_rem)
    return T


def _print_adaptive_analysis(model, hw, args, kw_base, results):
    """Print adaptive decode: optional measured acceptance, else fail-fraction sweep."""
    nl = model["n_layers"]
    B  = args.batch_size
    W  = args.adaptive_window
    n_dec = args.n_decode
    avg_ctx = args.n_prompt + n_dec // 2

    draft_q  = args.draft_kv_quant
    verify_q = args.verify_kv_quant

    n_q_dec = B * 1

    main_bpw = kw_base.get("weight_bpw", float(model.get("weight_bits", 16)))
    draft_bpw = _effective_draft_weight_spec(args, kw_base, model)

    # Per-step times at avg context (draft uses lighter weights if --draft-weight-bpw / profile)
    T_draft_fixed, T_draft_kv = _decode_step_time(
        n_q_dec, avg_ctx, draft_q, model, kw_base, weight_bpw=draft_bpw)
    T_verify_fixed, T_verify_kv = _decode_step_time(
        n_q_dec, avg_ctx, verify_q, model, kw_base, weight_bpw=main_bpw)
    T_fp16_fixed, T_fp16_kv = _decode_step_time(
        n_q_dec, avg_ctx, "fp16", model, kw_base, weight_bpw=16.0)

    T_draft_step  = T_draft_fixed  + T_draft_kv
    T_verify_step = T_verify_fixed + T_verify_kv
    T_fp16_step   = T_fp16_fixed   + T_fp16_kv

    # Verification window cost (re-run W tokens at verifier precision)
    T_verify_win = _verify_window_time(B, W, avg_ctx, verify_q, model, kw_base,
                                       weight_bpw=main_bpw)

    # KV write-back cost after verification (write base+delta for W tokens)
    kv_bpt_verify = _kv_bytes_per_token_per_layer(verify_q, model["n_kv_heads"],
                                                   model["head_dim"], args.kv_group_size)
    T_writeback = W * 2 * kv_bpt_verify * nl / (hw["memory_bw_gbps"] * 1e9 * args.mem_eff)

    T_verify_total = T_verify_win + T_writeback

    # Amortized verify overhead per draft token
    T_verify_amort = T_verify_total / W if W > 0 else 0.0

    print(f"\n{'═'*105}")
    print(f"  ADAPTIVE DRAFT + VERIFIER  (same arch as --model; draft = GGUF quant / bpw)")
    print(f"  draft KV={draft_q}  verify KV={verify_q}  W={W}")
    print(f"  main weight_bpw={_format_weight_bpw_spec(main_bpw)}  "
          f"draft weight_bpw={_format_weight_bpw_spec(draft_bpw)}")
    print(f"{'─'*105}")
    print(f"  Per-step decode times (avg_ctx={avg_ctx}):")
    print(f"    Draft  ({draft_q:>20}):  {T_draft_step*1e3:>8.4f} ms/step  "
          f"({1/T_draft_step:.1f} tok/s)")
    print(f"    Verify ({verify_q:>20}):  {T_verify_step*1e3:>8.4f} ms/step  "
          f"({1/T_verify_step:.1f} tok/s)")
    print(f"    FP16   ({'fp16':>20}):  {T_fp16_step*1e3:>8.4f} ms/step  "
          f"({1/T_fp16_step:.1f} tok/s)")
    print(f"  Verification window (W={W} tokens):")
    print(f"    Re-run W tokens:   {T_verify_win*1e3:>8.3f} ms")
    print(f"    KV write-back:     {T_writeback*1e3:>8.3f} ms")
    print(f"    Total per window:  {T_verify_total*1e3:>8.3f} ms")
    print(f"    Amortized/token:   {T_verify_amort*1e3:>8.4f} ms  "
          f"({T_verify_amort/T_draft_step*100:.1f}% of draft step)")

    # Measured acceptance (from cluster JSON): expected window cost model
    if args.acceptance_rate is not None:
        a = float(args.acceptance_rate)
        if not 0.0 <= a <= 1.0:
            raise ValueError("--acceptance-rate must be in [0, 1]")
        T_acc = _adaptive_decode_acceptance(
            W, n_dec, T_draft_step, T_verify_total, T_fp16_step, a)
        tps_a = n_dec / T_acc if T_acc > 0 else float("inf")
        T_fp16_decode = T_fp16_step * n_dec
        T_verify_decode = T_verify_step * n_dec
        print(f"\n{'─'*105}")
        print(f"  ACCEPTANCE-RATE MODEL  accept_rate={a:.4f}  (from cluster / JSON)")
        print(f"    E[T_decode] ≈ Σ_windows( W·T_draft + T_verify_win + (1-a)·W·T_fp16 )")
        print(f"    Decode-only total: {T_acc*1e3:>10.2f} ms  →  {tps_a:.1f} tok/s")
        print(f"    vs fp16 decode-only:  {T_fp16_decode/T_acc:.2f}×")
        print(f"    vs verify decode-only: {T_verify_decode/T_acc:.2f}×")
        print(f"{'═'*105}")

    # Sweep over first-fail fractions (sensitivity; skipped when only acceptance needed)
    if args.acceptance_rate is not None:
        return

    print(f"\n{'─'*105}")
    print(f"  Sweep: first_fail_frac = fraction of decode completed before fallback to verifier")
    print(f"  first_fail=1.0 → draft never fails (best case);  =0.0 → fails immediately (worst case)")
    print(f"{'─'*105}")
    hdr = (f"  {'fail_frac':>9} {'n_draft':>7} {'n_tail':>7} "
           f"{'T_draft':>9} {'T_verify':>9} {'T_tail':>9} {'T_total':>9} "
           f"{'tok/s':>7} {'vs_fp16':>8} {'vs_verif':>9}")
    print(hdr)
    print(f"  {'─'*100}")

    # Reference: non-adaptive decode at verifier and fp16
    T_verify_decode = T_verify_step * n_dec
    T_fp16_decode   = T_fp16_step * n_dec

    fail_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ff in fail_fracs:
        first_fail_pos = int(ff * n_dec)
        n_draft_phase  = first_fail_pos
        n_tail         = n_dec - first_fail_pos

        # Draft phase: generate at draft BW + periodic verification every W tokens
        n_windows = n_draft_phase // W if W > 0 else 0
        T_draft_gen    = T_draft_step * n_draft_phase
        T_draft_verify = T_verify_total * n_windows
        T_phase1       = T_draft_gen + T_draft_verify

        # Tail phase: generate at verifier BW (full base+delta)
        T_phase2 = T_verify_step * n_tail

        T_total = T_phase1 + T_phase2
        tps = n_dec / T_total if T_total > 0 else float("inf")
        vs_fp16  = T_fp16_decode / T_total if T_total > 0 else float("inf")
        vs_verif = T_verify_decode / T_total if T_total > 0 else float("inf")

        label = ""
        if ff == 0.0: label = " ← worst (always verifier)"
        if ff == 1.0: label = " ← best (draft never fails)"

        print(f"  {ff:>9.1f} {n_draft_phase:>7} {n_tail:>7} "
              f"{T_phase1*1e3:>8.1f}m {T_draft_verify*1e3:>8.1f}m {T_phase2*1e3:>8.1f}m "
              f"{T_total*1e3:>8.1f}m {tps:>7.1f} {vs_fp16:>7.2f}× {vs_verif:>8.2f}×{label}")

    # Reference rows
    print(f"  {'─'*100}")
    ref_fp16_tps  = n_dec / T_fp16_decode if T_fp16_decode > 0 else float("inf")
    ref_verif_tps = n_dec / T_verify_decode if T_verify_decode > 0 else float("inf")
    print(f"  {'[fp16]':>9} {'':>7} {'':>7} "
          f"{'':>9} {'':>9} {'':>9} {T_fp16_decode*1e3:>8.1f}m {ref_fp16_tps:>7.1f} "
          f"{'1.00':>7}× {'':>9}")
    print(f"  {'['+verify_q+']':>9} {'':>7} {'':>7} "
          f"{'':>9} {'':>9} {'':>9} {T_verify_decode*1e3:>8.1f}m {ref_verif_tps:>7.1f} "
          f"{T_fp16_decode/T_verify_decode:>7.2f}× {'1.00':>8}×")
    print(f"{'═'*105}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model + Hardware
    p.add_argument("--model", default="qwen3-8b", choices=list(MODEL_PRESETS))
    p.add_argument("--hw",    default="a100-80g",  choices=list(HARDWARE_PRESETS))
    p.add_argument("--n-layers",   type=int, default=None)
    p.add_argument("--d-model",    type=int, default=None)
    p.add_argument("--n-heads",    type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--head-dim",   type=int, default=None)
    p.add_argument("--ffn-dim",    type=int, default=None)
    p.add_argument("--weight-bits", type=int, default=None,
                   help="Integer weight width fallback; use --weight-bpw for effective bpw (e.g. Q8_0).")
    p.add_argument("--weight-bpw", type=float, default=None,
                   help="Effective bits per weight element for matmul traffic (overrides weight_bits). "
                        "Example: 8.5 for Q8_0 average bpw.")

    # Workload
    p.add_argument("--n-prompt",  type=int, default=1024)
    p.add_argument("--n-decode",  type=int, default=512)
    p.add_argument("--batch-size",type=int, default=1,
                   help="Sequences in batch. Weights shared; KV and activations scale with B.")

    # KV quantization
    p.add_argument("--kv-quant",      default="fp16",
                   choices=list(QUANT_CONFIGS),
                   help="KV cache quantization (affects attention KV read BW and kv_write)")
    p.add_argument("--kv-group-size", type=int, default=64)

    # Attention
    p.add_argument("--flash-attn",    action="store_true", default=True,
                   help="Use flash attention memory model (default: on)")
    p.add_argument("--no-flash-attn", dest="flash_attn", action="store_false",
                   help="Use standard attention (O(n²) intermediate buffer)")

    # Bandwidth and efficiency
    p.add_argument("--compute-eff",         type=float, default=0.70,
                   help="Compute utilisation fraction (default 0.70)")
    p.add_argument("--mem-eff",             type=float, default=0.85,
                   help="Weight/activation memory BW utilisation (default 0.85)")
    p.add_argument("--attn-eff",            type=float, default=0.85,
                   help="Attention KV memory BW utilisation (default 0.85)")
    p.add_argument("--attn-bw",             type=float, default=None,
                   help="Override attention KV BW (GB/s). Useful to model HBM vs DRAM "
                        "or the effect of KV quantisation on effective bandwidth.")
    p.add_argument("--weight-bw-fraction",  type=float, default=1.0,
                   help="Weight BW as fraction of peak hw BW (default 1.0)")
    p.add_argument("--act-bw-fraction",     type=float, default=1.0,
                   help="Activation BW as fraction of peak hw BW (default 1.0)")
    p.add_argument("--attn-bw-fraction",    type=float, default=1.0,
                   help="Attention KV BW as fraction of peak hw BW (overridden by --attn-bw)")

    # Padding
    p.add_argument("--padding-efficiency", type=float, default=1.0,
                   help="Fraction of tokens that are real (1.0 = no padding). "
                        "Reduces compute FLOPs and activation bytes; weights unchanged.")

    # Verification window (single-shot)
    p.add_argument("--verify-window",    type=int,   default=0,
                   help="Verification window size W (0 = skip). "
                        "Models cost of re-running W tokens at verifier quant.")
    p.add_argument("--verify-ctx-frac",  type=float, default=0.5,
                   help="KV cache fill fraction at verification point "
                        "(first_fail_pos / n_decode). Default 0.5.")

    # Adaptive base+delta decode
    p.add_argument("--adaptive",         action="store_true", default=False,
                   help="Enable adaptive base+delta decode analysis. "
                        "Sweeps over fail rates and shows tok/s for each.")
    p.add_argument("--draft-kv-quant",   default="int3_half_1357_ch",
                   choices=list(QUANT_CONFIGS),
                   help="Draft (base-only) KV quant for adaptive mode (default: int3_half_1357_ch)")
    p.add_argument("--verify-kv-quant",  default="int4_ch",
                   choices=list(QUANT_CONFIGS),
                   help="Verifier (base+delta) KV quant for adaptive mode (default: int4_ch)")
    p.add_argument("--adaptive-window",  type=int, default=32,
                   help="Verification window size W for adaptive mode (default: 32)")
    p.add_argument("--draft-weight-bpw", type=float, default=None,
                   help="Draft path effective bits/weight (matmul bytes). Overrides --draft-gguf-quant.")
    p.add_argument("--draft-gguf-quant", default=None,
                   choices=list(GGUF_WEIGHT_QUANT_BPW),
                   help="Draft GGUF family (same shapes as --model): Q2_K, Q3_K_M, Q4_K_M, Q8_0, F16. "
                        "Sets effective bpw; use --draft-weight-bpw to override numerically.")
    p.add_argument("--main-gguf-quant", default=None,
                   choices=list(GGUF_WEIGHT_QUANT_BPW),
                   help="Verifier/main GGUF weight quant (default: use --weight-bpw or model weight_bits). "
                        "Same architecture; only bpw changes.")
    p.add_argument("--weight-bpw-profile", default=None, metavar="PATH",
                   help="JSON with attn/mlp/default and/or per-op bpw (qkv_proj, out_proj, ffn_gate_up, "
                        "ffn_down). Overrides uniform --weight-bpw for matmul weight bytes. "
                        "Generate from a GGUF via research/scripts/gguf_roofline_weight_profile.py.")
    p.add_argument("--draft-weight-bpw-profile", default=None, metavar="PATH",
                   help="Same schema as --weight-bpw-profile for adaptive draft path only.")
    p.add_argument("--acceptance-rate", type=float, default=None,
                   help="Measured window acceptance in [0,1] from cluster JSON; "
                        "with --adaptive, prints acceptance model instead of fail-fraction sweep.")
    p.add_argument("--calibration-json", type=str, default=None, metavar="PATH",
                   help="Calibration: (1) legacy summary JSON with decode_ms_per_token_fp16_baseline "
                        "or decode_scale_measured_over_roofline; (2) benchmark_kv_timing.py output "
                        "(rows[] + bootstrap_ms_quant) — matches row by model/B/prompt/decode/"
                        "weight_tag/kv_type; scale = measured_ms / roofline_this_run.")
    p.add_argument("--calibration-weight-tag", type=str, default=None, metavar="TAG",
                   help="GGUF weight tag for kv_timing row match (e.g. Q8_0, Q2_K). "
                        "Default: --main-gguf-quant or Q8_0.")
    p.add_argument("--calibration-kv-type", type=str, default=None, metavar="TYPE",
                   help="Native KV type for kv_timing row match: f16, q8_0, q4_0. "
                        "Default: map from --kv-quant (fp16→f16, int4_ch→q4_0, …).")
    p.add_argument(
        "--calibration-bucket-policy", choices=("mean", "match_ctx"), default="match_ctx",
        help="For kv_timing rows with decode_buckets[]: mean=row measured_ms; "
             "match_ctx=ms/tok from bucket with ctx_mid closest to n_prompt+n_decode/2 (like .out file).")
    p.add_argument(
        "--calibration-relax-decode-len", action="store_true",
        help="If no row matches decode_len exactly, use a row with same model/B/prompt/weight/kv "
             "and decode_len >= --n-decode (then decode_buckets apply). For short --n-decode vs "
             "long cluster benchmark JSON.")

    args = p.parse_args()
    _apply_gguf_weight_quant_args(args)

    # Build configs
    model = dict(MODEL_PRESETS[args.model])
    for attr, key in [("n_layers","n_layers"),("d_model","d_model"),("n_heads","n_heads"),
                      ("ffn_dim","ffn_dim"),("head_dim","head_dim"),("weight_bits","weight_bits")]:
        val = getattr(args, attr.replace("-","_"), None)
        if val is not None: model[key] = val
    if args.n_kv_heads is not None: model["n_kv_heads"] = args.n_kv_heads

    hw = dict(HARDWARE_PRESETS[args.hw])

    analyze_and_print(model, hw, args)


if __name__ == "__main__":
    main()
