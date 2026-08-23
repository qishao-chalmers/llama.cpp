#!/usr/bin/env python3
"""
adaptive_gen_speedup.py  —  Performance model for adaptive speculative generation.

Estimates tok/s speedup for the scheme:
  1. Draft model (quantized) generates W tokens
  2. Teacher (Q8_0) verifies all W in one parallel pass  ≈  npl=W wall time
  3. With prob α = p^W (window accept): commit W draft tokens
     With prob 1-α (reject):           teacher generates M correction tokens

Per-op draft timing is estimated by scaling the Q8_0 measured baseline by the
bpw ratio for weight-bound ops.  Non-weight ops (RMSNorm, RoPE, FlashAttn,
KV write) are assumed fixed unless explicitly modified by KV quantization.

KV cache quantization (--kv-quant int8|int4):
  - Changes FlashAttn time: fewer KV bytes to load, but dequant overhead per tile.
    At short ctx (npp≤512): net overhead ≈ +8% (dequant > BW savings).
    At long ctx (npp=4096+): net speedup ≈ -10% to -20% (BW savings win).
  - Lowers per-token acceptance rate: KV accumulation error degrades draft quality.
    Estimated penalty: int8 ≈ 0.2%, int4 ≈ 1% per token (from PPL delta data).

Hardware precision (--draft nvfp4|fp2):
  - nvfp4: 4 bpw native GEMM (Blackwell MXFP4/NVFP4) — uniform 4-bit, no dequant overhead.
  - fp2:   2 bpw hypothetical native GEMM — extreme compression, very low p_accept expected.
  These are modelled the same as INT quants (memory-bandwidth-bound at bs=1), just
  with lower effective bpw and uniform tensor assignment.

Data source: research/results/decode_sweep_Qwen3-{8B,14B}.txt

Usage:
    # Qwen3-8B baseline
    python3 research/scripts/adaptive_gen_speedup.py \\
        research/results/decode_sweep_Qwen3-8B.txt \\
        --npp 512 --window 4 8 16 --correction 8 \\
        --draft q4_k_m q3_k_m q2_k

    # With int4 KV cache on draft model
    python3 research/scripts/adaptive_gen_speedup.py \\
        research/results/decode_sweep_Qwen3-8B.txt \\
        --draft q4_k_m q3_k_m --kv-quant int4 --npp 2048

    # Include hypothetical native FP4 / FP2 hardware
    python3 research/scripts/adaptive_gen_speedup.py \\
        research/results/decode_sweep_Qwen3-8B.txt \\
        --draft q4_k_m q3_k_m q2_k nvfp4 fp2 --window 8 16

    # Qwen3-14B with plot
    python3 research/scripts/adaptive_gen_speedup.py \\
        research/results/decode_sweep_Qwen3-14B.txt \\
        --model 14b --npp 1024 --plot
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# ─── Weight quantization config ───────────────────────────────────────────────
#
# Per-tensor bpw for each projection type, from llama-quant.cpp analysis.
# Source: research/context/gguf_quant_sizes.md
#
# Block bpw:  Q8_0=8.50  Q6_K=6.56  Q5_K=5.50  Q4_K=4.50  Q3_K=3.44  Q2_K=2.63
#
# Q4_K_M recipe (Qwen3-8B):
#   attn_q,k → Q4_K;  attn_v → 50% Q4_K + 50% Q6_K (use_more_bits, ~5.53 avg)
#   attn_o   → Q4_K;  ffn_gate,up → Q4_K;  ffn_down → 50% Q4_K + 50% Q6_K
# Q3_K_M recipe:
#   attn_q,k → Q3_K;  attn_v,o → Q4_K;  ffn_gate,up → Q3_K;  ffn_down → Q4_K
# Q2_K recipe:
#   attn_q,k,v,o → Q3_K (upgraded for stability);  ffn_gate,up,down → Q2_K
# nvfp4 (Blackwell MXFP4/NVFP4):
#   All projection tensors at uniform 4.0 bpw — native hardware GEMM, no software dequant.
#   Models the same as Q4_K_M in memory-bandwidth terms, but uniform (no Q5_K upgrades).
# fp2 (hypothetical native 2-bit):
#   All tensors at 2.0 bpw.  Extreme compression; very low acceptance rate expected.

QUANT_PROJ_BPW: dict[str, dict[str, float]] = {
    "q8_0": {
        "attn_q": 8.50, "attn_k": 8.50, "attn_v": 8.50, "attn_o": 8.50,
        "ffn_gate": 8.50, "ffn_up": 8.50, "ffn_down": 8.50,
        "attn_avg": 8.50, "ffn_avg": 8.50,
    },
    "q4_k_m": {
        "attn_q": 4.50, "attn_k": 4.50, "attn_v": 5.53, "attn_o": 4.50,
        "ffn_gate": 4.50, "ffn_up": 4.50, "ffn_down": 5.53,
        "attn_avg": 4.76,  # (4.50+4.50+5.53+4.50)/4
        "ffn_avg":  4.84,  # (4.50+4.50+5.53)/3
    },
    "q3_k_m": {
        "attn_q": 3.44, "attn_k": 3.44, "attn_v": 4.50, "attn_o": 4.50,
        "ffn_gate": 3.44, "ffn_up": 3.44, "ffn_down": 4.50,
        "attn_avg": 3.97, "ffn_avg": 3.79,
    },
    "q2_k": {
        "attn_q": 3.44, "attn_k": 3.44, "attn_v": 3.44, "attn_o": 3.44,
        "ffn_gate": 2.63, "ffn_up": 2.63, "ffn_down": 2.63,
        "attn_avg": 3.44, "ffn_avg": 2.63,
    },
    "nvfp4": {
        # Blackwell MXFP4/NVFP4: uniform 4 bpw, native GEMM (no software dequant overhead).
        # At bs=1, memory-bandwidth-bound → same scaling as Q4_K_M but uniform across tensors.
        "attn_q": 4.00, "attn_k": 4.00, "attn_v": 4.00, "attn_o": 4.00,
        "ffn_gate": 4.00, "ffn_up": 4.00, "ffn_down": 4.00,
        "attn_avg": 4.00, "ffn_avg": 4.00,
    },
    "fp2": {
        # Hypothetical 2-bit native hardware GEMM (e.g., future Blackwell or research chip).
        # Very aggressive; model quality degradation is severe — treat p_accept as empirical.
        "attn_q": 2.00, "attn_k": 2.00, "attn_v": 2.00, "attn_o": 2.00,
        "ffn_gate": 2.00, "ffn_up": 2.00, "ffn_down": 2.00,
        "attn_avg": 2.00, "ffn_avg": 2.00,
    },
}

QUANT_LABEL: dict[str, str] = {
    "q8_0": "Q8_0", "q4_k_m": "Q4_K_M", "q3_k_m": "Q3_K_M", "q2_k": "Q2_K",
    "nvfp4": "NVFP4", "fp2": "FP2",
}

# ─── KV cache quantization config ─────────────────────────────────────────────
#
# Applied to the DRAFT model's KV cache only (teacher always uses fp16 KV).
# Two effects:
#   1. FlashAttn time changes: fewer KV bytes loaded, but dequant overhead per tile.
#      At short npp (≤512): dequant overhead dominates → net SLOWER (+8%).
#      At long npp (≥4096): bandwidth savings dominate → net FASTER (-10 to -20%).
#      Source: research/context/roofline_calibration.md, KV quant overhead section.
#   2. Per-token acceptance rate is reduced by KV accumulation error.
#      Penalty estimates from PPL degradation in server_results:
#        int8_ch: Δppl ≈ +0.01 → ~0.2% token degradation
#        int4_ch: Δppl ≈ +0.05 → ~1.0% token degradation

KV_QUANT_CONFIG: dict[str, dict] = {
    "fp16": {
        # No KV quantization; FlashAttn unchanged; no acceptance penalty.
        "label":         "fp16 KV",
        "attn_factor":   {512: 1.00, 1024: 1.00, 2048: 1.00, 4096: 1.00},
        "p_penalty":     1.000,   # multiplicative factor on per-token p_accept
    },
    "int8": {
        # Native int8 KV (e.g., flash_attn Q8_0 KV cache).
        # +8% overhead at short ctx; ~0% at npp=2048; ~-10% at npp=4096+.
        "label":         "int8 KV",
        "attn_factor":   {512: 1.08, 1024: 1.04, 2048: 1.00, 4096: 0.90},
        "p_penalty":     0.998,
    },
    "int4": {
        # Native int4 KV (e.g., flash_attn Q4_0 KV cache).
        # Same per-tile dequant overhead as int8 but 2× fewer bytes.
        # Net: +8% overhead at short ctx; -15 to -20% at long ctx.
        "label":         "int4 KV",
        "attn_factor":   {512: 1.08, 1024: 1.02, 2048: 0.93, 4096: 0.82},
        "p_penalty":     0.990,
    },
}

# ─── Per-op baseline (Q8_0 measured) ─────────────────────────────────────────
#
# Qwen3-8B  npl=1, npp=512 (decode_sweep_Qwen3-8B.txt, ### CUDA PERF DECODE row)
# avg_ms is the per-step total across all 36 layers.
# bpw_key: key into QUANT_PROJ_BPW to scale this op; None = fixed cost.

OPS_8B: list[tuple[str, str, float, str | None]] = [
    # (op_id,   display_name,      ms_q8,  bpw_key)
    ("q_proj",   "Q proj",          0.710,  "attn_q"),
    ("k_proj",   "K proj",          0.401,  "attn_k"),
    ("v_proj",   "V proj",          0.407,  "attn_v"),
    ("o_proj",   "O proj",          0.728,  "attn_o"),
    ("ffn_gate", "FFN gate+up",     2.811,  "ffn_gate"),  # gate+up fused at npl=1
    ("ffn_down", "FFN down",        1.619,  "ffn_down"),
    ("norm",     "RMSNorm",         0.994,  None),        # activation compute, fixed
    ("rope",     "RoPE",            0.392,  None),        # activation compute, fixed
    ("attn",     "FlashAttn",       0.475,  None),        # KV bandwidth, fixed
    ("kv_write", "KV write",        0.191,  None),        # activation BW, fixed
    ("lm_head",  "LM head (Q6_K)", 0.425,  None),        # Q6_K for all recipes
    ("other",    "Other",           0.023,  None),
]

# Qwen3-14B  npl=1, npp=1024 (decode_sweep_Qwen3-14B.txt DURATION table)
# Categories match the pre-aggregated summary columns.
OPS_14B: list[tuple[str, str, float, str | None]] = [
    ("qkvo",  "QKV+O proj",       2.02,  "attn_avg"),   # weight-bound, Q+K+V+O combined
    ("ffn",   "FFN gate+up+dn",   8.01,  "ffn_avg"),    # weight-bound, gate+up+down combined
    ("norm",  "RMSNorm",          1.18,  None),
    ("rope",  "RoPE",             0.43,  None),
    ("attn",  "FlashAttn",        1.25,  None),
    ("other", "Other",            1.64,  None),          # KV write + LM head + misc
]

# CPU-GPU pipeline overhead (wall − kernel) at npl=1; from roofline_calibration.md.
# At npl≥2 this stabilizes at ~1.94ms (CUDA stream launch overhead grows slightly).
T_FLOOR_MS = 1.42


# ─── File parsers ─────────────────────────────────────────────────────────────

def parse_8b_per_op(path: Path) -> dict[tuple[int, int], dict]:
    """Parse per-op timing + T_floor for all (npp, npl) from the 8B decode sweep.

    Returns {(npp, npl): {"ops": [(op_id, label, ms_q8, bpw_key), ...], "t_floor": float}}.

    At npl=1 ffn_gate:MUL_MAT is fused with up (one kernel); at npl≥2 gate and up
    appear as separate entries.
    """
    OP_MAP: dict[str, tuple[str, str, str | None]] = {
        "Qcur:MUL_MAT":             ("q_proj",   "Q proj",    "attn_q"),
        "Kcur:MUL_MAT":             ("k_proj",   "K proj",    "attn_k"),
        "Vcur:MUL_MAT":             ("v_proj",   "V proj",    "attn_v"),
        "node_*:MUL_MAT":           ("o_proj",   "O proj",    "attn_o"),
        "ffn_gate:MUL_MAT":         ("ffn_gate", "FFN gate",  "ffn_gate"),
        "ffn_up:MUL_MAT":           ("ffn_up",   "FFN up",    "ffn_up"),
        "ffn_out:MUL_MAT":          ("ffn_down", "FFN down",  "ffn_down"),
        "__fattn__:FLASH_ATTN_EXT": ("attn",     "FlashAttn", None),
        "result_output:MUL_MAT":    ("lm_head",  "LM head",   None),
        "norm:RMS_NORM":            ("norm",     "RMSNorm",   None),
    }
    ROPE_OPS     = {"Qcur:ROPE", "Kcur:ROPE"}
    KV_WRITE_OPS = {"cache_k:SET_ROWS", "cache_v:SET_ROWS"}

    text      = path.read_text()
    result:   dict[tuple[int, int], dict] = {}
    header_re = re.compile(r"### CUDA PERF DECODE row\s+npp=(\d+)\s+npl=(\d+)")
    timing_re = re.compile(r"kernel=([\d.]+)\s+ms\s+wall=([\d.]+)\s+ms")
    op_re     = re.compile(r"^\s{2}(\S+:\S+)\s+([\d.]+)\s+")

    for hdr in header_re.finditer(text):
        npp, npl = int(hdr.group(1)), int(hdr.group(2))
        key = (npp, npl)
        if key in result:
            continue

        block = text[hdr.end(): hdr.end() + 3000]
        lines = block.splitlines()

        t_floor = T_FLOOR_MS
        for line in lines[:3]:
            m = timing_re.search(line)
            if m:
                t_floor = float(m.group(2)) - float(m.group(1))
                break

        by_id: dict[str, float] = {}
        rope_ms = kv_write_ms = other_ms = 0.0
        sep_count = 0
        for line in lines:
            if line.strip().startswith("----"):
                sep_count += 1
                if sep_count == 2:
                    break
                continue
            if sep_count < 1:
                continue
            m = op_re.match(line)
            if not m:
                continue
            op_name, ms = m.group(1), float(m.group(2))
            if op_name in ROPE_OPS:
                rope_ms += ms
            elif op_name in KV_WRITE_OPS:
                kv_write_ms += ms
            elif op_name in OP_MAP:
                oid = OP_MAP[op_name][0]
                by_id[oid] = by_id.get(oid, 0.0) + ms
            else:
                other_ms += ms

        # Canonical order; at npl=1 gate+up are fused into ffn_gate
        canonical: list[tuple[str, str, str | None]] = [
            ("q_proj",  "Q proj",  "attn_q"),
            ("k_proj",  "K proj",  "attn_k"),
            ("v_proj",  "V proj",  "attn_v"),
            ("o_proj",  "O proj",  "attn_o"),
        ]
        if "ffn_up" in by_id:
            canonical += [("ffn_gate", "FFN gate", "ffn_gate"),
                          ("ffn_up",   "FFN up",   "ffn_up")]
        else:
            canonical.append(("ffn_gate", "FFN gate+up", "ffn_gate"))
        canonical += [
            ("ffn_down", "FFN down", "ffn_down"),
            ("norm",     "RMSNorm",  None),
        ]

        ops: list[tuple] = []
        for oid, lbl, bpw_key in canonical:
            if oid in by_id:
                ops.append((oid, lbl, by_id[oid], bpw_key))
        if rope_ms     > 0: ops.append(("rope",     "RoPE",     rope_ms,     None))
        if "attn"      in by_id: ops.append(("attn",  "FlashAttn", by_id["attn"], None))
        if kv_write_ms > 0: ops.append(("kv_write", "KV write", kv_write_ms, None))
        if "lm_head"   in by_id: ops.append(("lm_head", "LM head", by_id["lm_head"], None))
        if other_ms    > 0: ops.append(("other",    "Other",    other_ms,    None))

        result[key] = {"ops": ops, "t_floor": t_floor}

    return result


def parse_8b_wall_times(path: Path) -> dict[tuple[int, int], float]:
    """Parse Qwen3-8B decode sweep file → {(npp, npl): wall_ms}."""
    text = path.read_text()
    result: dict[tuple[int, int], float] = {}
    pattern = re.compile(
        r"npp=(\d+),\s*npl=(\d+).*?OK:\s*wall=([\d.]+)ms",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        key = (int(m.group(1)), int(m.group(2)))
        if key not in result:
            result[key] = float(m.group(3))
    return result


def parse_14b_wall_times(path: Path) -> dict[tuple[int, int], float]:
    """Parse Qwen3-14B decode sweep summary table → {(npp, npl): wall_ms}."""
    result: dict[tuple[int, int], float] = {}
    in_duration = header_done = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if "DECODE DURATION" in s:
            in_duration, header_done = True, False
        elif "DECODE PERCENTAGE" in s:
            in_duration = False
        elif in_duration:
            if "npp" in s and "Wall" in s:
                header_done = True
            elif header_done and s and not s.startswith("---"):
                parts = s.split()
                try:
                    result[(int(parts[0]), int(parts[1]))] = float(parts[8])
                except (ValueError, IndexError):
                    pass
    return result


def _interpolate_wall(wall_table: dict[tuple[int, int], float], npp: int, target_npl: int) -> float:
    """Interpolate wall time at (npp, target_npl); clamp to table range."""
    if (npp, target_npl) in wall_table:
        return wall_table[(npp, target_npl)]
    candidates = sorted(
        [(npl, t) for (n, npl), t in wall_table.items() if n == npp],
        key=lambda x: x[0],
    )
    if not candidates:
        raise ValueError(f"No wall-time data for npp={npp}")
    lo = [(npl, t) for npl, t in candidates if npl <= target_npl]
    hi = [(npl, t) for npl, t in candidates if npl >= target_npl]
    if lo and hi:
        lo_npl, lo_t = lo[-1]
        hi_npl, hi_t = hi[0]
        if lo_npl == hi_npl:
            return lo_t
        frac = (target_npl - lo_npl) / (hi_npl - lo_npl)
        return lo_t + frac * (hi_t - lo_t)
    return candidates[-1][1] if not hi else candidates[0][1]


def get_t_verify(
    wall_table: dict[tuple[int, int], float],
    npp: int,
    W: int,
    npl: int = 1,
) -> float:
    """T_verify: teacher checks W tokens for npl sequences in one step.

    The teacher processes npl*W tokens simultaneously, approximated as
    wall time at (npp, npl*W) in the decode-step table.
    """
    return _interpolate_wall(wall_table, npp, npl * W)


# ─── Performance model ────────────────────────────────────────────────────────

def _kv_attn_factor(kv_cfg: dict, npp: int) -> float:
    """Interpolate the FlashAttn time multiplier for a given npp."""
    table = kv_cfg["attn_factor"]
    keys = sorted(table.keys())
    if npp <= keys[0]:
        return table[keys[0]]
    if npp >= keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= npp <= hi:
            frac = (npp - lo) / (hi - lo)
            return table[lo] + frac * (table[hi] - table[lo])
    return 1.0


def scale_ops(
    ops: list,
    weight_quant: str,
    kv_quant: str = "fp16",
    npp: int = 512,
) -> list[tuple[str, str, float, float]]:
    """Return (op_id, label, ms_q8, ms_draft) scaling each op by the appropriate factor.

    Weight-bound ops scale by bpw ratio.
    FlashAttn scales by KV quant attn_factor (BW savings vs dequant overhead).
    """
    bpw_d   = QUANT_PROJ_BPW[weight_quant]
    bpw_r   = QUANT_PROJ_BPW["q8_0"]
    kv_cfg  = KV_QUANT_CONFIG[kv_quant]
    attn_f  = _kv_attn_factor(kv_cfg, npp)
    out = []
    for op_id, label, ms_q8, bpw_key in ops:
        if bpw_key:
            scale = bpw_d[bpw_key] / bpw_r[bpw_key]
        elif op_id == "attn":
            scale = attn_f   # FlashAttn: KV quant effect
        else:
            scale = 1.0      # RMSNorm, RoPE, KV write, LM head, other: fixed
        out.append((op_id, label, ms_q8, ms_q8 * scale))
    return out


def draft_wall(
    ops: list,
    weight_quant: str,
    kv_quant: str = "fp16",
    npp: int = 512,
    t_floor: float = T_FLOOR_MS,
) -> float:
    """Estimated wall time (ms/step) for draft model."""
    return sum(ms_d for _, _, _, ms_d in scale_ops(ops, weight_quant, kv_quant, npp)) + t_floor


def effective_p(p_base: float, kv_quant: str) -> float:
    """Apply KV quant acceptance penalty to per-token p_accept."""
    return p_base * KV_QUANT_CONFIG[kv_quant]["p_penalty"]


def adaptive_tok_s(
    T_draft: float,
    T_teacher: float,
    T_verify: float,
    W: int,
    p: float,
    kv_quant: str = "fp16",
) -> dict:
    """Compute adaptive gen (= standard speculative decoding) performance.

    The teacher verify pass over W draft tokens also produces the corrected /
    bonus token at no extra cost.  So no separate M-token correction window.

      p_eff  = p × kv_penalty          (KV quant degrades per-token agreement)
      E[draft accepted] = p_eff·(1 - p_eff^W) / (1 - p_eff)   (geometric sum)
      E[tokens]         = E[draft accepted] + 1                 (free from verify)
      E[time]           = W·T_draft + T_verify(W)               (no extra teacher step)
      tok/s             = 1000·E[tokens] / E[time]
      speedup           = tok/s × T_teacher / 1000
    """
    p_eff = effective_p(p, kv_quant)
    if abs(1.0 - p_eff) < 1e-9:
        e_draft = float(W)
    else:
        e_draft = p_eff * (1.0 - p_eff ** W) / (1.0 - p_eff)
    e_tok = e_draft + 1.0
    e_ms  = W * T_draft + T_verify
    tok_s = 1000.0 * e_tok / e_ms
    return {"tok_s": tok_s, "e_tok": e_tok, "p_eff": p_eff,
            "speedup": tok_s * T_teacher / 1000.0}


def breakeven_p(
    T_draft: float,
    T_teacher: float,
    T_verify: float,
    W: int,
    kv_quant: str = "fp16",
) -> float:
    """Minimum base p_accept for adaptive gen to beat pure teacher throughput.

    Solve: E[tokens](p_base) = (W·T_draft + T_verify) / T_teacher
    where  E[tokens] = p_eff·(1-p_eff^W)/(1-p_eff) + 1,  p_eff = p_base × penalty.
    Returns nan if p_base=1 still cannot match teacher.
    """
    e_time   = W * T_draft + T_verify
    target   = e_time / T_teacher          # E[tokens] needed to match teacher
    penalty  = KV_QUANT_CONFIG[kv_quant]["p_penalty"]

    def e_tok(p_base: float) -> float:
        p_eff = p_base * penalty
        if abs(1.0 - p_eff) < 1e-9:
            return float(W) + 1.0
        return p_eff * (1.0 - p_eff ** W) / (1.0 - p_eff) + 1.0

    if e_tok(1.0) < target:
        return math.nan          # even p=1 can't match teacher
    if e_tok(0.0) >= target:
        return 0.0

    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        (lo if e_tok(mid) < target else hi).__class__  # dummy
        if e_tok(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ceiling_tok_s(T_draft: float, T_verify: float, W: int) -> float:
    """Best-case tok/s (p→1): (W+1) tokens in W·T_draft+T_verify ms."""
    return 1000.0 * (W + 1) / (W * T_draft + T_verify)


# ─── Output ───────────────────────────────────────────────────────────────────

def print_op_table(
    ops: list,
    drafts: list[str],
    kv_quant: str = "fp16",
    npp: int = 512,
    t_floor: float = T_FLOOR_MS,
) -> None:
    labels = [QUANT_LABEL[q] for q in drafts]
    scaled = {q: scale_ops(ops, q, kv_quant, npp) for q in drafts}
    kv_label = KV_QUANT_CONFIG[kv_quant]["label"]

    W_NAME, W_BOUND = 18, 14
    col = 9
    header = f"  {'Op':<{W_NAME}} {'Bound by':<{W_BOUND}} {'Q8_0':>{col}}"
    for lbl in labels:
        header += f"  {lbl:>{col}}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, (op_id, label, ms_q8, bpw_key) in enumerate(ops):
        if bpw_key:
            bound = "weight BW"
        elif op_id == "attn":
            bound = f"KV BW ({kv_label})"
        else:
            bound = "fixed"
        line = f"  {label:<{W_NAME}} {bound:<{W_BOUND}} {ms_q8:{col}.3f}"
        for q in drafts:
            line += f"  {scaled[q][i][3]:{col}.3f}"
        print(line)

    print("  " + "─" * (len(header) - 2))

    q8_kernel = sum(ms for _, _, ms, _ in ops)
    print(f"  {'Kernel total':<{W_NAME}} {'':>{W_BOUND}} {q8_kernel:{col}.3f}", end="")
    for q in drafts:
        print(f"  {sum(ms_d for _, _, _, ms_d in scaled[q]):{col}.3f}", end="")
    print()

    print(f"  {'+ T_floor':<{W_NAME}} {'CPU-GPU latency':>{W_BOUND}} {t_floor:{col}.2f}", end="")
    for _ in drafts:
        print(f"  {t_floor:{col}.2f}", end="")
    print()

    q8_wall = q8_kernel + t_floor
    print(f"  {'Wall (ms/step)':<{W_NAME}} {'':>{W_BOUND}} {q8_wall:{col}.2f}", end="")
    for q in drafts:
        w = sum(ms_d for _, _, _, ms_d in scaled[q]) + t_floor
        print(f"  {w:{col}.2f}", end="")
    print()

    print(f"  {'Tok/s (draft)':<{W_NAME}} {'':>{W_BOUND}} {1000/q8_wall:{col}.1f}", end="")
    for q in drafts:
        w = sum(ms_d for _, _, _, ms_d in scaled[q]) + t_floor
        print(f"  {1000/w:{col}.1f}", end="")
    print()

    # Show KV penalty row if not fp16
    if kv_quant != "fp16":
        penalty = KV_QUANT_CONFIG[kv_quant]["p_penalty"]
        print(f"\n  KV quant p_accept penalty: ×{penalty:.3f} per token  "
              f"(e.g., base p=0.95 → effective p={0.95*penalty:.4f})")


def print_adaptive_table(
    ops: list,
    wall_table: dict,
    drafts: list[str],
    npp: int,
    windows: list[int],
    p_values: list[float],
    kv_quant: str = "fp16",
    npl: int = 1,
    t_floor: float = T_FLOOR_MS,
) -> None:
    # Teacher uses Q8_0 at the requested npl
    T_teacher = _interpolate_wall(wall_table, npp, npl)

    T_drafts = {q: draft_wall(ops, q, kv_quant, npp, t_floor) for q in drafts}
    labels   = [QUANT_LABEL[q] for q in drafts]
    kv_label = KV_QUANT_CONFIG[kv_quant]["label"]
    penalty  = KV_QUANT_CONFIG[kv_quant]["p_penalty"]

    col = 10

    for W in windows:
        T_verify = get_t_verify(wall_table, npp, W, npl)

        ceiling_teacher = 1000.0 / T_teacher
        print(f"\n  W={W}  |  T_verify={T_verify:.2f}ms (≈npl={npl*W}@npp={npp})"
              f"  |  T_teacher={T_teacher:.2f}ms ({ceiling_teacher:.1f} tok/s)  |  draft KV={kv_label}")
        print(f"  Correction: 1 token free from verify pass  "
              f"[E[tok]= p_eff(1-p_eff^W)/(1-p_eff) + 1,  E[time]= W·T_draft + T_verify]")

        ceiling = {q: ceiling_tok_s(T_drafts[q], T_verify, W) for q in drafts}
        pure    = {q: 1000.0 / T_drafts[q] for q in drafts}
        print(f"  Pure draft (no verify): "
              + "  ".join(f"{QUANT_LABEL[q]}={pure[q]:.1f}" for q in drafts))
        print(f"  Ceiling (p→1, W+1 tok/window): "
              + "  ".join(f"{QUANT_LABEL[q]}={ceiling[q]:.1f}" for q in drafts))
        if kv_quant != "fp16":
            print(f"  KV p_penalty: ×{penalty:.3f}/tok  "
                  f"→ p_effective = p_base × {penalty:.3f}")

        # Header
        hdr = f"\n  {'p_base':>7}  {'p_eff':>6}  {'E[tok]':>7}  {'Teacher':>{col}}"
        for lbl in labels:
            hdr += f"  {lbl:>{col}}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for p in p_values:
            p_eff = effective_p(p, kv_quant)
            # E[tok] for display (using Q8_0 bpw just to show the count)
            e_tok_disp = (p_eff * (1 - p_eff**W) / (1 - p_eff) + 1) if abs(1-p_eff) > 1e-9 else W + 1
            line = f"  {p:7.3f}  {p_eff:6.4f}  {e_tok_disp:7.2f}  {1000/T_teacher:{col}.1f}"
            for q in drafts:
                r = adaptive_tok_s(T_drafts[q], T_teacher, T_verify, W, p, kv_quant)
                s = r["tok_s"]
                delta = s - ceiling_teacher
                line += f"  {s:{col-6}.1f}({delta:+.1f})"
            print(line)

        # Break-even
        print(f"\n  Break-even p_accept (adaptive = teacher {ceiling_teacher:.1f} tok/s):")
        for q in drafts:
            p_be = breakeven_p(T_drafts[q], T_teacher, T_verify, W, kv_quant)
            if math.isnan(p_be):
                ceil = ceiling_tok_s(T_drafts[q], T_verify, W)
                print(f"    {QUANT_LABEL[q]:<8}: impossible  "
                      f"(ceiling {ceil:.1f} < teacher {ceiling_teacher:.1f} tok/s)")
            else:
                r_be = adaptive_tok_s(T_drafts[q], T_teacher, T_verify, W, p_be, kv_quant)
                print(f"    {QUANT_LABEL[q]:<8}: p_base > {p_be:.4f}  "
                      f"(E[tok]={r_be['e_tok']:.2f},  speedup=1.00×)")


# ─── Plot ─────────────────────────────────────────────────────────────────────

def make_plot(
    ops: list,
    wall_table: dict,
    drafts: list[str],
    npp: int,
    windows: list[int],
    model_tag: str,
    kv_quant: str,
    out_path: Path,
    npl: int = 1,
    t_floor: float = T_FLOOR_MS,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    T_teacher = _interpolate_wall(wall_table, npp, npl)
    T_drafts  = {q: draft_wall(ops, q, kv_quant, npp, t_floor) for q in drafts}
    kv_label  = KV_QUANT_CONFIG[kv_quant]["label"]

    default_colors = ["C0", "C1", "C2", "C3", "C4"]
    color_map = {q: default_colors[i % len(default_colors)] for i, q in enumerate(drafts)}
    p_range = np.linspace(0.70, 1.00, 200)

    fig, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5), sharey=True)
    if len(windows) == 1:
        axes = [axes]

    for ax, W in zip(axes, windows):
        T_verify = get_t_verify(wall_table, npp, W, npl)
        ax.axhline(1000 / T_teacher, color="k", linestyle="--", lw=1.5,
                   label=f"Teacher Q8_0 ({1000/T_teacher:.1f})")
        for q in drafts:
            tok_s = [adaptive_tok_s(T_drafts[q], T_teacher, T_verify, W, float(p), kv_quant)["tok_s"]
                     for p in p_range]
            ax.plot(p_range, tok_s, label=QUANT_LABEL[q], color=color_map[q])
            p_be = breakeven_p(T_drafts[q], T_teacher, T_verify, W, kv_quant)
            if not math.isnan(p_be) and 0.70 <= p_be <= 1.00:
                ax.axvline(p_be, color=color_map[q], linestyle=":", alpha=0.6)

        top = max(ceiling_tok_s(T_drafts[q], T_verify, W) for q in drafts) * 1.05
        ax.fill_betweenx([1000 / T_teacher, top], 0.70, 1.00,
                         alpha=0.04, color="green")
        ax.set_title(f"W={W}  T_verify={T_verify:.1f}ms  npl={npl}\n"
                     f"draft KV={kv_label}", fontsize=9)
        ax.set_xlabel("Base per-token acceptance rate  p")
        if ax is axes[0]:
            ax.set_ylabel("Tok/s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Adaptive gen speedup — Qwen3-{model_tag}  (npp={npp},  npl={npl},  draft KV={kv_label})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


# ─── npl sweep ────────────────────────────────────────────────────────────────

def print_npl_sweep(
    per_op_db: dict,
    wall_table: dict,
    npp: int,
    drafts: list[str],
    kv_quant: str,
) -> None:
    """Show how FlashAttn fraction and KV-quant benefit grow with npl."""
    kv_cfg = KV_QUANT_CONFIG[kv_quant]
    attn_f = _kv_attn_factor(kv_cfg, npp)

    npl_values = sorted(set(b for (n, b) in per_op_db if n == npp))
    if not npl_values:
        return

    print(f"\n{'─'*72}")
    print(f"  npl sweep (npp={npp}) — how batch size shifts the weight/KV balance")
    print(f"  KV draft: {kv_cfg['label']}  (attn_factor={attn_f:.2f}×)")
    print(f"{'─'*72}")
    hdr = f"  {'npl':>5}  {'Wall(ms)':>9}  {'T_floor':>8}  {'Attn_ms':>8}  {'Attn%':>6}  {'Weight_ms':>10}  {'Weight%':>7}"
    for q in drafts:
        hdr += f"  {'dW(' + QUANT_LABEL[q] + ')':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for b in npl_values:
        entry = per_op_db[(npp, b)]
        ops   = entry["ops"]
        t_fl  = entry["t_floor"]
        kernel_total = sum(ms for _, _, ms, _ in ops)
        wall_ms      = kernel_total + t_fl

        attn_ms   = next((ms for oid, _, ms, _ in ops if oid == "attn"), 0.0)
        weight_ms = sum(ms for _, _, ms, bk in ops if bk is not None)
        attn_pct  = 100.0 * attn_ms / wall_ms
        wt_pct    = 100.0 * weight_ms / wall_ms

        row = (f"  {b:>5}  {wall_ms:>9.2f}  {t_fl:>8.2f}  "
               f"{attn_ms:>8.3f}  {attn_pct:>5.1f}%  "
               f"{weight_ms:>10.3f}  {wt_pct:>6.1f}%")
        for q in drafts:
            w = draft_wall(ops, q, kv_quant, npp, t_fl)
            row += f"  {w:>12.2f}"
        print(row)

    # Show int4-KV attn time at each npl (to quantify the growing benefit)
    if kv_quant == "fp16":
        print(f"\n  Tip: run --kv-quant int4 to see how KV quant savings grow with npl.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Adaptive gen speedup model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("sweep_file", type=Path,
                    help="decode_sweep_Qwen3-8B.txt or decode_sweep_Qwen3-14B.txt")
    ap.add_argument("--model", choices=["8b", "14b"], default="8b",
                    help="Which model's per-op baseline to use")
    ap.add_argument("--npp", type=int, default=512,
                    help="KV prompt length for timing lookup (use 1024 for 14B)")
    ap.add_argument("--npl", type=int, default=1,
                    help="Number of parallel sequences (batch size) for the draft model. "
                         "At npl=B each draft step processes B sequences; teacher verifies "
                         "B×W tokens; KV quant benefit grows. Only supported for --model 8b.")
    ap.add_argument("--npl-sweep", action="store_true",
                    help="Print a table of attn fraction vs npl to show how KV matters more "
                         "at larger batch. Only for --model 8b.")
    ap.add_argument("--window", "-W", nargs="+", type=int, default=[4, 8, 16],
                    help="Draft window size(s) W")
    ap.add_argument("--draft", nargs="+",
                    default=["q4_k_m", "q3_k_m", "q2_k"],
                    choices=list(QUANT_PROJ_BPW.keys()),
                    help="Draft weight quantization type(s).  "
                         "nvfp4 = Blackwell MXFP4 (uniform 4-bit native).  "
                         "fp2 = hypothetical 2-bit native hardware.")
    ap.add_argument("--kv-quant", choices=list(KV_QUANT_CONFIG.keys()), default="fp16",
                    help="KV cache quantization on the draft model.  "
                         "int8/int4 affect FlashAttn speed and acceptance rate.  "
                         "Teacher always uses fp16 KV.")
    ap.add_argument("--p", nargs="+", type=float,
                    default=[0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00],
                    help="Base per-token acceptance rates (before KV penalty) to tabulate")
    ap.add_argument("--plot", action="store_true",
                    help="Save tok/s vs p curves (requires matplotlib)")
    args = ap.parse_args()

    # Parse timing data
    wall_table = (parse_8b_wall_times if args.model == "8b" else parse_14b_wall_times)(args.sweep_file)
    if not wall_table:
        sys.exit("ERROR: no wall-time data found in file")

    # For 8B: parse per-op data for the requested npl (and npl-sweep if requested)
    per_op_db: dict = {}
    ops = OPS_8B if args.model == "8b" else OPS_14B
    t_floor = T_FLOOR_MS

    if args.model == "8b" and (args.npl != 1 or args.npl_sweep):
        per_op_db = parse_8b_per_op(args.sweep_file)

    if args.model == "8b" and args.npl != 1:
        entry = per_op_db.get((args.npp, args.npl))
        if entry is None:
            available = sorted(b for (n, b) in per_op_db if n == args.npp)
            sys.exit(f"ERROR: no per-op data for npp={args.npp}, npl={args.npl}. "
                     f"Available npl values: {available}")
        ops     = entry["ops"]
        t_floor = entry["t_floor"]

    model_tag = args.model.upper()
    sep = "=" * 72

    kv_cfg   = KV_QUANT_CONFIG[args.kv_quant]
    kv_label = kv_cfg["label"]

    print(f"\n{sep}")
    print(f"  Adaptive speculative generation — performance model")
    print(f"  Model: Qwen3-{model_tag} (Q8_0 teacher)  |  npp={args.npp}  |  npl={args.npl}")
    print(f"  Draft KV cache: {kv_label}  "
          f"(p_penalty={kv_cfg['p_penalty']:.3f}/tok, "
          f"attn_factor@npp={args.npp}: "
          f"{_kv_attn_factor(kv_cfg, args.npp):.2f}×)")
    print(sep)

    # ── 0. npl sweep (optional) ──────────────────────────────────────────────
    if args.npl_sweep and args.model == "8b":
        if not per_op_db:
            per_op_db = parse_8b_per_op(args.sweep_file)
        print_npl_sweep(per_op_db, wall_table, args.npp, args.draft, args.kv_quant)

    # ── 1. Per-op breakdown ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  Per-op timing breakdown (ms/decode-step, npl={args.npl}, npp={args.npp})")
    print(f"  Weight-bound ops scale by bpw ratio.  "
          f"FlashAttn scales by KV quant factor.")
    print(f"{'─'*72}")
    print_op_table(ops, args.draft, args.kv_quant, args.npp, t_floor)

    # ── 2. Adaptive gen tables ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  Adaptive gen tok/s  (teacher=Q8_0 npl={args.npl}, draft=quantized + {kv_label})")
    print(f"  Correction: 1 token free from verify pass.  No separate teacher correction step.")
    print(f"  T_verify uses effective npl={args.npl}×W (teacher processes npl×W tokens).")
    print(f"{'─'*72}")
    print_adaptive_table(
        ops, wall_table, args.draft,
        args.npp, args.window, args.p, args.kv_quant,
        npl=args.npl, t_floor=t_floor,
    )

    # ── 3. Optional plot ─────────────────────────────────────────────────────
    if args.plot:
        try:
            suffix = f"_{args.kv_quant}kv" if args.kv_quant != "fp16" else ""
            npl_sfx = f"_npl{args.npl}" if args.npl != 1 else ""
            out = args.sweep_file.parent / (
                f"adaptive_gen_speedup_qwen3{args.model}_{args.npp}{npl_sfx}{suffix}.png"
            )
            make_plot(ops, wall_table, args.draft, args.npp,
                      args.window, model_tag, args.kv_quant, out,
                      npl=args.npl, t_floor=t_floor)
        except ImportError:
            print("\n[--plot] requires matplotlib — skipping.")


if __name__ == "__main__":
    main()
