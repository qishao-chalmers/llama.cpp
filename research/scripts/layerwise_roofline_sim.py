#!/usr/bin/env python3
"""Layer-wise roofline decode-step simulator (MVP).

Loads model structure from model_structures.json (--preset) or --structure-json,
then walks one decode step layer-by-layer: per-op max(T_comp, T_mem).

**Weight quantization:** default is a **single** effective bpw from preset / ``weight_tag`` /
``--weight-bits`` (same for all GEMMs). For **mixed per-tensor** sizes (e.g. different
Q/K/V/FFN quants), pass ``--gguf PATH`` to a ``.gguf`` file: stored tensor sizes drive
roofline bytes per op (``blk.L.*`` names from llama.cpp); missing tensors fall back to the
same synthetic ``n_params × bpe`` as above. RMSNorm γ uses ``--norm-weight-bits`` when
not overridden by GGUF norms.

**KV cache quantization:** ``--kv-type`` maps to ``perf_model.QUANT_CONFIGS`` (f16, q8_0,
q4_0, int3_ch, …). Optional ``--kv-group-size`` / ``--kv-asymmetric`` match
``kv_bytes_per_token`` semantics.

**η fit:** ``--eta-json`` from ``fit_layerwise_eta.py``.

**Linear calibration (cluster match):** ``--calibration-json`` from
``fit_layerwise_calibration.py`` applies
``ms/tok = t_floor_ms/B + scale ×`` raw layerwise ms/tok.

Does not modify perf_model.py; imports kv_bytes_per_token / QUANT_CONFIGS only.

**Attention:** ``--attn-impl simple`` (default) is one ``attn_core`` roofline per layer.
``--attn-impl flash`` models a **decode** FlashAttention-style fused kernel: total time is
``max(T_comp, T_mem)`` over softmax-augmented FLOPs and KV+Q HBM bytes, then splits time
across ``flash_qk``, ``flash_softmax``, ``flash_pv`` (all still ``family=attn_core`` for
aggregation). ``--fa-bc`` only sets ``kv_tiles_along_seq`` in JSON for now.

**KV bytes in ``attn_core`` (native q8/q4):** storage bytes under-predict latency because
CUDA dequant dominates BW savings (see ``benchmark_kv_timing.KV_DEQUANT_OVERHEAD``).
Default ``--kv-attn-byte-mode fp16_equiv_dequant`` uses **fp16 KV bytes × overhead** for
the attention **read** path for ``int8_ch`` / ``int4_ch``; use ``storage`` for the legacy
bytes-only model.

**Overlap / serial-sum error:** optional ``--attn-time-scale`` and
``--attn-time-scale-inv-batch`` multiply **only** ``attn_core`` time by
``attn_time_scale + attn_time_scale_inv_batch / B`` (β_attn-style correction).

Example::

    python3 layerwise_roofline_sim.py --preset qwen3-8b-q8_0 -B 8 --ctx-len 4096 --hw h100-sxm \\
        --kv-type f16 \\
        --eta-json ../results/qwen3-8b/profile/layerwise_eta_h100.json \\
        --calibration-json ../results/qwen3-8b/profile/layerwise_calibration_h100.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_STRUCT = os.path.join(_SCRIPT_DIR, "model_structures.json")

# Read-only use of perf_model KV accounting
sys.path.insert(0, _SCRIPT_DIR)
from perf_model import QUANT_CONFIGS, kv_bytes_per_token, prefill_time  # noqa: E402
from gguf_layerwise_weights import (  # noqa: E402
    load_gguf_tensor_n_bytes,
    resolve_layer_weight_bytes_from_gguf,
)


def _fmt_bytes(n: float) -> str:
    n = float(n)
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"

# Duplicated hardware table (keep independent of edits to perf_model presets)
HARDWARE_PRESETS: dict[str, dict[str, float]] = {
    "a100-80g": dict(
        compute_tflops=312.0,
        memory_bw_gbps=2000.0,
        efficiency=0.70,
    ),
    "h100-sxm": dict(
        compute_tflops=989.0,
        memory_bw_gbps=3350.0,
        efficiency=0.70,
    ),
    "a6000": dict(
        compute_tflops=154.0,
        memory_bw_gbps=768.0,
        efficiency=0.65,
    ),
    "rtx4090": dict(
        compute_tflops=165.0,
        memory_bw_gbps=1008.0,
        efficiency=0.65,
    ),
    "mn5-h100": dict(
        compute_tflops=989.0,
        memory_bw_gbps=3350.0,
        efficiency=0.70,
    ),
    "gh200": dict(
        compute_tflops=989.0,
        memory_bw_gbps=3350.0,
        efficiency=0.70,
    ),
}

# llama.cpp / CLI names -> perf_model.kv_bytes_per_token key (must exist in QUANT_CONFIGS)
KV_QUANT_ALIASES: dict[str, str] = {
    "f16": "fp16",
    "fp16": "fp16",
    "bf16": "bf16",
    "q8_0": "int8_ch",
    "q8": "int8_ch",
    "int8": "int8_ch",
    "int8_ch": "int8_ch",
    "q4_0": "int4_ch",
    "q4": "int4_ch",
    "int4": "int4_ch",
    "int4_ch": "int4_ch",
    "int4_tok": "int4_tok",
    "q3": "int3_ch",
    "int3_ch": "int3_ch",
    "int3_half": "int3_half",
    "int3_half_1357": "int3_half_1357",
    "int3_half_1357_ch": "int3_half_1357_ch",
    "q2": "int2_ch",
    "int2_ch": "int2_ch",
    "int2": "int2",
}


def resolve_kv_quant_key(cli: str) -> str:
    """Map user --kv-type to a perf_model QUANT_CONFIGS key."""

    raw = cli.strip()
    k = raw.lower()
    if k in KV_QUANT_ALIASES:
        return KV_QUANT_ALIASES[k]
    if raw in QUANT_CONFIGS:
        return raw
    for qk in QUANT_CONFIGS:
        if qk.lower() == k:
            return qk
    raise ValueError(
        f"Unknown kv-type {cli!r}. Try: {sorted(KV_QUANT_ALIASES.keys())} "
        f"or: {list(QUANT_CONFIGS.keys())}"
    )


# Native GPU KV (q8_0 / q4_0): attn read path effective bytes = fp16 bytes × mult
# (aligned with research/scripts/benchmark_kv_timing.KV_DEQUANT_OVERHEAD).
NATIVE_KV_ATTN_FP16_MULT: dict[str, float] = {
    "int8_ch": 1.08,
    "int4_ch": 1.08,
}


def kv_attn_read_bpt_layer(
    kv_quant_key: str,
    model: dict[str, Any],
    n_layers: int,
    *,
    kv_group_size: Optional[int],
    kv_asym: bool,
    mode: str,
) -> float:
    """Bytes per layer for **attention KV read** roofline (one token row per layer).

    mode:
        ``storage`` — use ``kv_bytes_per_token(kv_quant_key) / n_layers`` (legacy).
        ``fp16_equiv_dequant`` — for ``int8_ch`` / ``int4_ch``, use
        ``kv_bytes_per_token('fp16')/n_layers × NATIVE_KV_ATTN_FP16_MULT``; else storage.
    """

    nl = float(n_layers)
    kv_total = kv_bytes_per_token(
        kv_quant_key,
        dict(model),
        kv_group_size,
        kv_asym,
    )
    kv_layer = kv_total / nl
    m = str(mode).strip().lower()
    if m == "storage":
        return float(kv_layer)
    if m == "fp16_equiv_dequant":
        if kv_quant_key not in NATIVE_KV_ATTN_FP16_MULT:
            return float(kv_layer)
        fp16_total = kv_bytes_per_token(
            "fp16",
            dict(model),
            kv_group_size,
            kv_asym,
        )
        return float(fp16_total) / nl * NATIVE_KV_ATTN_FP16_MULT[kv_quant_key]
    raise ValueError(
        f"Unknown kv_attn_byte_mode {mode!r}; use 'storage' or 'fp16_equiv_dequant'"
    )


def effective_attn_scale(
    batch_size: int,
    kv_quant_key: str,
    attn_time_scale: float,
    attn_time_scale_inv_batch: float,
    attn_scale_by_batch: Optional[dict[int, float]] = None,
    attn_scale_by_batch_and_kv: Optional[dict[str, dict[int, float]]] = None,
) -> float:
    """Return attention time multiplier.

    Priority:
      1) If ``attn_scale_by_batch_and_kv`` has an entry for (kv_quant_key, B), use it.
      2) Else if ``attn_scale_by_batch`` has an entry for B, use it.
      3) Else use ``attn_time_scale + attn_time_scale_inv_batch / B``.
    """

    B = float(batch_size)
    if B < 1.0:
        raise ValueError("batch_size must be >= 1")
    if attn_scale_by_batch_and_kv:
        by_kv = attn_scale_by_batch_and_kv.get(str(kv_quant_key))
        if by_kv:
            v = by_kv.get(int(batch_size))
            if v is not None:
                return float(v)
    if attn_scale_by_batch:
        v = attn_scale_by_batch.get(int(batch_size))
        if v is not None:
            return float(v)
    return float(attn_time_scale) + float(attn_time_scale_inv_batch) / B


def load_sim_physics_json(path: Optional[str]) -> dict[str, Any]:
    """Optional JSON with keys:

    - kv_attn_byte_mode
    - attn_time_scale, attn_time_scale_inv_batch, attn_scale_by_batch, attn_scale_by_batch_and_kv
    - weight_time_scale_by_tag: map weight_tag -> multiplier (scales GEMM family only)
    """

    if not path:
        return {
            "kv_attn_byte_mode": "fp16_equiv_dequant",
            "attn_time_scale": 1.0,
            "attn_time_scale_inv_batch": 0.0,
            "attn_scale_by_batch": {},
            "attn_scale_by_batch_and_kv": {},
            "weight_time_scale_by_tag": {},
        }
    data = _load_json(os.path.expanduser(path))
    base = load_sim_physics_json(None)
    for k in base:
        if k in data and data[k] is not None:
            base[k] = data[k]
    return base


def resolve_sim_physics(
    sim_physics_json: Optional[str],
    kv_attn_byte_mode: Optional[str] = None,
    attn_time_scale: Optional[float] = None,
    attn_time_scale_inv_batch: Optional[float] = None,
) -> dict[str, Any]:
    """Merge ``load_sim_physics_json`` with optional CLI overrides."""

    phys = load_sim_physics_json(sim_physics_json)
    if kv_attn_byte_mode is not None:
        phys["kv_attn_byte_mode"] = str(kv_attn_byte_mode)
    if attn_time_scale is not None:
        phys["attn_time_scale"] = float(attn_time_scale)
    if attn_time_scale_inv_batch is not None:
        phys["attn_time_scale_inv_batch"] = float(attn_time_scale_inv_batch)
    if "attn_scale_by_batch" not in phys or phys["attn_scale_by_batch"] is None:
        phys["attn_scale_by_batch"] = {}
    # Normalize keys to int, values to float if provided as JSON.
    if isinstance(phys["attn_scale_by_batch"], dict):
        norm: dict[int, float] = {}
        for k, v in phys["attn_scale_by_batch"].items():
            try:
                kk = int(k)
                vv = float(v)
            except Exception:
                continue
            norm[kk] = vv
        phys["attn_scale_by_batch"] = norm

    if "attn_scale_by_batch_and_kv" not in phys or phys["attn_scale_by_batch_and_kv"] is None:
        phys["attn_scale_by_batch_and_kv"] = {}
    if isinstance(phys["attn_scale_by_batch_and_kv"], dict):
        norm2: dict[str, dict[int, float]] = {}
        for kv_k, by_b in phys["attn_scale_by_batch_and_kv"].items():
            if not isinstance(by_b, dict):
                continue
            try:
                kv_key = resolve_kv_quant_key(str(kv_k))
            except Exception:
                kv_key = str(kv_k)
            inner: dict[int, float] = {}
            for bk, bv in by_b.items():
                try:
                    b_int = int(bk)
                    s = float(bv)
                except Exception:
                    continue
                inner[b_int] = s
            if inner:
                norm2[str(kv_key)] = inner
        phys["attn_scale_by_batch_and_kv"] = norm2

    if "weight_time_scale_by_tag" not in phys or phys["weight_time_scale_by_tag"] is None:
        phys["weight_time_scale_by_tag"] = {}
    if isinstance(phys["weight_time_scale_by_tag"], dict):
        wnorm: dict[str, float] = {}
        for k, v in phys["weight_time_scale_by_tag"].items():
            try:
                kk = str(k)
                vv = float(v)
            except Exception:
                continue
            wnorm[kk] = vv
        phys["weight_time_scale_by_tag"] = wnorm
    return phys


def tag_to_weight_bits(tag: str) -> float:
    """Effective bits/weight for roofline-style byte counts (aligned with benchmark_kv_timing)."""

    t = tag.upper().replace("-", "_")
    if "Q2_K" in t:
        return 3.1
    if "Q3_K" in t:
        return 4.0
    if "Q4_K" in t:
        return 4.9
    if "Q8_0" in t or t == "Q8_0":
        return 8.5
    if "F16" in t or "FP16" in t:
        return 16.0
    return 8.5


def resolve_weight_bits(
    model: dict[str, Any],
    weight_tag: Optional[str],
    weight_bits_override: Optional[float],
) -> tuple[float, Optional[str]]:
    """Return (effective_bits per weight for roofline, tag or None).

    **Single effective bpw** for all weight tensors (same convention as ``perf_model`` /
    ``benchmark_kv_timing``). Per-layer mixed K-quants are **not** split in the MVP.

    Priority: ``--weight-bits`` > ``--weight-tag`` > ``model["weight_bits"]`` (if set) >
    ``model["weight_tag"]`` → ``tag_to_weight_bits`` > default 16.
    """

    if weight_bits_override is not None:
        return float(weight_bits_override), weight_tag
    if weight_tag:
        return tag_to_weight_bits(str(weight_tag)), str(weight_tag)
    if model.get("weight_bits") is not None:
        wb = float(model["weight_bits"])
        wt = model.get("weight_tag")
        return wb, (str(wt) if wt is not None else None)
    if model.get("weight_tag"):
        t = str(model["weight_tag"])
        return tag_to_weight_bits(t), t
    return 16.0, None


ETA_FIELD_NAMES: tuple[str, ...] = (
    "gemm_comp",
    "gemm_bw",
    "attn_comp",
    "attn_bw",
    "elem_comp",
    "elem_bw",
    "kv_bw",
)


@dataclass
class Eta:
    """Per-op-family efficiencies (0–1] applied to hardware peaks."""

    gemm_comp: float = 0.40
    gemm_bw: float = 0.45
    attn_comp: float = 0.35
    attn_bw: float = 0.32
    elem_comp: float = 0.50
    elem_bw: float = 0.40
    kv_bw: float = 0.35


def eta_to_dict(e: Eta) -> dict[str, float]:
    return {k: float(getattr(e, k)) for k in ETA_FIELD_NAMES}


def eta_from_dict(d: dict[str, Any]) -> Eta:
    e = Eta()
    for k in ETA_FIELD_NAMES:
        if k in d and d[k] is not None:
            setattr(e, k, float(d[k]))
    return e


def load_eta_json(path: str) -> Eta:
    """Load JSON with top-level ``eta`` object or flat eta keys."""

    data = _load_json(path)
    if "eta" in data and isinstance(data["eta"], dict):
        return eta_from_dict(data["eta"])
    return eta_from_dict(data)


def load_calibration_json(path: str) -> dict[str, Any]:
    """Load JSON from fit_layerwise_calibration.py (``t_floor_ms`` + ``scale``)."""

    data = _load_json(os.path.expanduser(path))
    if "t_floor_ms" not in data or "scale" not in data:
        raise ValueError(f"{path}: expected t_floor_ms and scale")
    return data


def calibrated_ms_per_token(
    raw_ms_per_token: float,
    batch_size: int,
    t_floor_ms: float,
    scale: float,
) -> float:
    """``measured ≈ t_floor_ms / B + scale × layerwise_ms/tok`` (cluster fit)."""

    return float(t_floor_ms) / float(batch_size) + float(scale) * float(raw_ms_per_token)


@dataclass
class SimEvent:
    layer: int
    name: str
    family: str
    flops: float
    bytes_moved: float
    seconds: float
    batch_scaling: str = "linear_B"


def _load_json(path: str) -> dict[str, Any]:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return json.load(f)


def load_structure_catalog(path: str) -> dict[str, Any]:
    data = _load_json(path)
    if "schema_version" not in data or "presets" not in data:
        raise ValueError(f"{path}: expected schema_version and presets")
    return data


def resolve_model(
    *,
    preset: Optional[str],
    structure_json: Optional[str],
    catalog_path: str,
) -> tuple[str, dict[str, Any]]:
    """Return (preset_id, structure dict)."""

    if structure_json:
        data = _load_json(structure_json)
        if "presets" in data and preset:
            if preset not in data["presets"]:
                raise KeyError(f"preset {preset!r} not in {structure_json}")
            return preset, dict(data["presets"][preset])
        if "presets" in data and len(data["presets"]) == 1:
            pid, spec = next(iter(data["presets"].items()))
            return pid, dict(spec)
        # Flat single-model file
        spec = {k: v for k, v in data.items() if not str(k).startswith("_")}
        if "n_layers" not in spec:
            raise ValueError("structure JSON must contain presets map or flat n_layers, ...")
        return structure_json, spec

    if not preset:
        raise ValueError("Provide --preset or --structure-json")

    cat = load_structure_catalog(catalog_path)
    if preset not in cat["presets"]:
        raise KeyError(f"Unknown preset {preset!r}. Known: {list(cat['presets'].keys())}")
    return preset, dict(cat["presets"][preset])


def validate_structure(m: dict[str, Any]) -> None:
    req = (
        "n_layers",
        "d_model",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "ffn_dim",
        "ffn_style",
    )
    for k in req:
        if k not in m:
            raise ValueError(f"structure missing {k!r}")
    d, nh, hdim = m["d_model"], m["n_heads"], m["head_dim"]
    if nh * hdim != d:
        raise ValueError(f"d_model ({d}) != n_heads*head_dim ({nh}*{hdim})")
    if m["n_kv_heads"] > nh:
        raise ValueError("n_kv_heads must be <= n_heads")


def bytes_fp16_activations(n: float) -> float:
    return n * 2.0


def roofline_seconds(
    flops: float,
    bytes_moved: float,
    hw: dict[str, float],
    eta_comp: float,
    eta_bw: float,
) -> float:
    peak_flops = hw["compute_tflops"] * 1e12 * hw["efficiency"] * eta_comp
    peak_bw = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"] * eta_bw
    if peak_flops <= 0 or peak_bw <= 0:
        return float("nan")
    t_c = flops / peak_flops if flops > 0 else 0.0
    t_m = bytes_moved / peak_bw if bytes_moved > 0 else 0.0
    return max(t_c, t_m)


def split_fused_roofline_parts(
    parts: list[tuple[str, float, float]],
    hw: dict[str, float],
    eta_comp: float,
    eta_bw: float,
) -> list[tuple[str, float]]:
    """Split a fused roofline max(T_comp, T_mem) across named parts.

    Each part is (name, flops, bytes_moved). Non-overlapping bytes across parts
    should sum to the total HBM model for the fused kernel.

    Uses a **blend** so phases that are compute-heavy (softmax) still get time when
    the fused kernel is memory-bound, and vice versa::

        blend_i = t_comp * (flops_i / total_flops) + t_mem * (bytes_i / total_bytes)
        t_i = t_fused * blend_i / (t_comp + t_mem)

    When one of t_comp, t_mem is zero, this reduces to pure flops or pure bytes split.
    """

    if not parts:
        return []
    total_flops = sum(p[1] for p in parts)
    total_bytes = sum(p[2] for p in parts)
    peak_flops = hw["compute_tflops"] * 1e12 * hw["efficiency"] * eta_comp
    peak_bw = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"] * eta_bw
    t_comp = total_flops / peak_flops if total_flops > 0 and peak_flops > 0 else 0.0
    t_mem = total_bytes / peak_bw if total_bytes > 0 and peak_bw > 0 else 0.0
    t_fused = max(t_comp, t_mem)
    if t_fused <= 0.0:
        return [(n, 0.0) for n, _, _ in parts]

    denom = t_comp + t_mem
    if denom <= 0.0:
        n = len(parts)
        return [(name, t_fused / n) for name, _, _ in parts]

    out: list[tuple[str, float]] = []
    for name, fl, b in parts:
        c_part = t_comp * (fl / total_flops) if total_flops > 0 else 0.0
        m_part = t_mem * (b / total_bytes) if total_bytes > 0 else 0.0
        blend_i = c_part + m_part
        out.append((name, t_fused * (blend_i / denom)))
    return out


def simulate_decode_step(
    m: dict[str, Any],
    batch_size: int,
    ctx_len: int,
    hw: dict[str, float],
    eta: Eta,
    *,
    weight_bpe: float,
    norm_bpe: float,
    kv_quant_key: str,
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    gguf_tensor_bytes: Optional[dict[str, int]] = None,
    attn_impl: str = "simple",
    fa_bc: int = 128,
    attn_naive_spill: bool = False,
    kv_attn_byte_mode: str = "fp16_equiv_dequant",
    attn_time_scale: float = 1.0,
    attn_time_scale_inv_batch: float = 0.0,
    attn_scale_by_batch: Optional[dict[int, float]] = None,
    attn_scale_by_batch_and_kv: Optional[dict[str, dict[int, float]]] = None,
    weight_tag: Optional[str] = None,
    weight_time_scale_by_tag: Optional[dict[str, float]] = None,
) -> tuple[float, list[SimEvent]]:
    """One token step for B parallel sequences; return (total_seconds, events).

    If ``gguf_tensor_bytes`` is set, GEMM and RMSNorm weight bytes come from the GGUF
    tensor map (per tensor); fused ``attn_qkv`` is split by parameter count. Otherwise
    all GEMM weights use ``weight_bpe`` and norms use ``norm_bpe``.

    **Attention:** ``attn_impl``:

    - ``simple`` — single ``attn_core`` roofline (legacy): QK^T + P·V FLOPs, KV HBM
      read for full context; optional ``attn_naive_spill`` adds read/write traffic for
      logits/probs if an unfused implementation materializes them in HBM.
    - ``flash`` — FlashAttention-style **decode** (one query position per head): fused
      roofline with explicit softmax FLOPs, bytes = full KV read + Q activations (no
      O(L²) tile — decode is O(L)); emits ``flash_qk``, ``flash_softmax``, ``flash_pv``
      (same ``family`` = ``attn_core`` so aggregates match). ``fa_bc`` is used only to
      report ``ceil(ctx_len / fa_bc)`` KV tiles (metadata for scripts / JSON).

    **kv_attn_byte_mode:** ``storage`` (legacy) or ``fp16_equiv_dequant`` (default) —
    see ``kv_attn_read_bpt_layer``.

    **attn_time_scale / attn_time_scale_inv_batch / attn_scale_by_batch(/_and_kv):**
    multiply only ``attn_core`` time by an effective scale (see ``effective_attn_scale``).
    """

    B = int(batch_size)
    ctx = int(ctx_len)
    if B < 1 or ctx < 1:
        raise ValueError("batch_size and ctx_len must be >= 1")

    impl = str(attn_impl).strip().lower()
    if impl not in ("simple", "flash"):
        raise ValueError("attn_impl must be 'simple' or 'flash'")
    fa_bc_i = int(fa_bc)
    if fa_bc_i < 1:
        raise ValueError("fa_bc must be >= 1")

    nl = int(m["n_layers"])
    d = int(m["d_model"])
    nh = int(m["n_heads"])
    nkv = int(m["n_kv_heads"])
    hd = int(m["head_dim"])
    ffn = int(m["ffn_dim"])
    style = str(m.get("ffn_style", "swiglu"))
    ffn_mats = 3 if style == "swiglu" else 2
    bpe = float(weight_bpe)
    bpe_norm = float(norm_bpe)
    tb = gguf_tensor_bytes

    model_for_kv = dict(m)
    kv_bpt_total = kv_bytes_per_token(
        kv_quant_key,
        model_for_kv,
        kv_group_size,
        kv_asym,
    )
    kv_bpt_layer = kv_bpt_total / float(nl)
    kv_attn_bpt_layer = kv_attn_read_bpt_layer(
        kv_quant_key,
        m,
        nl,
        kv_group_size=kv_group_size,
        kv_asym=kv_asym,
        mode=kv_attn_byte_mode,
    )
    attn_scale_eff = effective_attn_scale(
        B,
        kv_quant_key,
        attn_time_scale,
        attn_time_scale_inv_batch,
        attn_scale_by_batch,
        attn_scale_by_batch_and_kv,
    )

    wt_scale = 1.0
    if weight_tag and weight_time_scale_by_tag and isinstance(weight_time_scale_by_tag, dict):
        wt_scale = float(weight_time_scale_by_tag.get(str(weight_tag), 1.0))

    events: list[SimEvent] = []
    total_s = 0.0

    def emit(
        layer: int,
        name: str,
        family: str,
        flops: float,
        bmov: float,
        scaling: str,
        eta_c: float,
        eta_b: float,
        *,
        time_scale: float = 1.0,
    ) -> None:
        nonlocal total_s
        t = roofline_seconds(flops, bmov, hw, eta_c, eta_b) * float(time_scale)
        events.append(
            SimEvent(
                layer=layer,
                name=name,
                family=family,
                flops=flops,
                bytes_moved=bmov,
                seconds=t,
                batch_scaling=scaling,
            )
        )
        total_s += t

    def append_event_seconds(
        layer_i: int,
        name: str,
        family: str,
        flops: float,
        bmov: float,
        scaling: str,
        seconds: float,
    ) -> None:
        nonlocal total_s
        events.append(
            SimEvent(
                layer=layer_i,
                name=name,
                family=family,
                flops=flops,
                bytes_moved=bmov,
                seconds=seconds,
                batch_scaling=scaling,
            )
        )
        total_s += seconds

    for layer in range(nl):
        wb_map: Optional[dict[str, float]] = None
        if tb is not None:
            wb_map = resolve_layer_weight_bytes_from_gguf(layer, m, tb, bpe, bpe_norm)

        # --- Pre-attn RMSNorm (one gamma vector; usually fp16 even when matrices quantized) ---
        w_rms = wb_map["attn_norm"] if wb_map is not None else d * bpe_norm
        act_rms = bytes_fp16_activations(B * d * 2)
        emit(
            layer,
            "rms_norm_pre_attn",
            "elementwise",
            B * d * 8.0,
            w_rms + act_rms,
            "linear_B",
            eta.elem_comp,
            eta.elem_bw,
        )

        # Q, K, V GEMMs (weights read once; activations scale with B)
        q_w = wb_map["q"] if wb_map is not None else d * nh * hd * bpe
        q_flops = 2.0 * B * d * nh * hd
        q_act = bytes_fp16_activations(B * d + B * nh * hd)
        emit(layer, "q_proj", "gemm", q_flops, q_w + q_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)

        k_w = wb_map["k"] if wb_map is not None else d * nkv * hd * bpe
        k_flops = 2.0 * B * d * nkv * hd
        k_act = bytes_fp16_activations(B * d + B * nkv * hd)
        emit(layer, "k_proj", "gemm", k_flops, k_w + k_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)

        v_w = wb_map["v"] if wb_map is not None else d * nkv * hd * bpe
        v_flops = 2.0 * B * d * nkv * hd
        v_act = bytes_fp16_activations(B * d + B * nkv * hd)
        emit(layer, "v_proj", "gemm", v_flops, v_w + v_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)

        # KV write: one new token row (K+V) for this layer, B sequences
        kv_write_b = kv_bpt_layer * B
        emit(
            layer,
            "kv_cache_write",
            "kv_rw",
            0.0,
            kv_write_b,
            "linear_B",
            eta.elem_comp,
            eta.kv_bw,
        )

        # --- Attention (simple = legacy single roofline; flash = fused + split sub-events) ---
        # kv_attn_bpt_layer: may use fp16-equiv dequant model for native int8/int4 KV (attn read only)
        kv_read_bytes = kv_attn_bpt_layer * float(ctx) * float(B)
        flops_qk = 2.0 * B * nh * ctx * hd
        flops_pv = 2.0 * B * nh * ctx * hd
        # Softmax over L positions per head (exp / sum / normalize): ~5 FLOPs per logit (tunable)
        flops_soft = 5.0 * B * nh * ctx

        if impl == "simple":
            attn_flops = flops_qk + flops_pv
            kv_bytes_simple = kv_read_bytes
            if attn_naive_spill:
                # Unfused path: logits + probs rows (fp16) read/written to HBM (order-of-magnitude)
                kv_bytes_simple += 4.0 * B * nh * ctx * 2.0
            emit(
                layer,
                "attn_core",
                "attn_core",
                attn_flops,
                kv_bytes_simple,
                "linear_B",
                eta.attn_comp,
                eta.attn_bw,
                time_scale=attn_scale_eff,
            )
        else:
            # FlashAttention-style decode: fused kernel keeps softmax on-chip; count Q read + KV read.
            kv_total = kv_read_bytes
            k_bytes = 0.5 * kv_total
            v_bytes = 0.5 * kv_total
            q_bytes = bytes_fp16_activations(B * nh * hd)
            soft_bytes = 1.0  # epsilon so memory-bound split does not collapse softmax to 0

            part_defs = [
                ("flash_qk", flops_qk, k_bytes + q_bytes),
                ("flash_softmax", flops_soft, soft_bytes),
                ("flash_pv", flops_pv, v_bytes),
            ]
            split = split_fused_roofline_parts(
                part_defs,
                hw,
                eta.attn_comp,
                eta.attn_bw,
            )
            for (name, fl, bmov), (_n2, sec) in zip(part_defs, split):
                append_event_seconds(
                    layer,
                    name,
                    "attn_core",
                    fl,
                    bmov,
                    "linear_B",
                    sec * attn_scale_eff,
                )

        o_w = wb_map["o"] if wb_map is not None else nh * hd * d * bpe
        o_flops = 2.0 * B * nh * hd * d
        o_act = bytes_fp16_activations(B * nh * hd + B * d)
        emit(layer, "o_proj", "gemm", o_flops, o_w + o_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)

        emit(
            layer,
            "residual_post_attn",
            "elementwise",
            B * d * 2.0,
            bytes_fp16_activations(2 * B * d),
            "linear_B",
            eta.elem_comp,
            eta.elem_bw,
        )

        w_rms2 = wb_map["ffn_norm"] if wb_map is not None else d * bpe_norm
        emit(
            layer,
            "rms_norm_pre_ffn",
            "elementwise",
            B * d * 8.0,
            w_rms2 + bytes_fp16_activations(B * d * 2),
            "linear_B",
            eta.elem_comp,
            eta.elem_bw,
        )

        if ffn_mats == 3:
            gate_w = wb_map["gate"] if wb_map is not None else d * ffn * bpe
            up_w = wb_map["up"] if wb_map is not None else d * ffn * bpe
            down_w = wb_map["down"] if wb_map is not None else ffn * d * bpe
            g_flops = 2.0 * B * d * ffn
            u_flops = 2.0 * B * d * ffn
            d_flops = 2.0 * B * ffn * d
            g_act = bytes_fp16_activations(B * d + B * ffn)
            u_act = bytes_fp16_activations(B * d + B * ffn)
            d_act = bytes_fp16_activations(B * ffn + B * d)
            emit(layer, "ffn_gate", "gemm", g_flops, gate_w + g_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)
            emit(layer, "ffn_up", "gemm", u_flops, up_w + u_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)
            emit(layer, "ffn_down", "gemm", d_flops, down_w + d_act, "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)
        else:
            up_w = wb_map["up"] if wb_map is not None else d * ffn * bpe
            down_w = wb_map["down"] if wb_map is not None else ffn * d * bpe
            u_flops = 2.0 * B * d * ffn
            d_flops = 2.0 * B * ffn * d
            emit(layer, "ffn_up", "gemm", u_flops, up_w + bytes_fp16_activations(B * d + B * ffn), "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)
            emit(layer, "ffn_down", "gemm", d_flops, down_w + bytes_fp16_activations(B * ffn + B * d), "linear_B", eta.gemm_comp, eta.gemm_bw, time_scale=wt_scale)

        emit(
            layer,
            "residual_post_ffn",
            "elementwise",
            B * d * 2.0,
            bytes_fp16_activations(2 * B * d),
            "linear_B",
            eta.elem_comp,
            eta.elem_bw,
        )

    return total_s, events


def predict_ms_per_token(
    model: dict[str, Any],
    *,
    batch_size: int,
    ctx_len: int,
    hw_name: str,
    eta: Eta,
    weight_bits: float,
    norm_weight_bits: float = 16.0,
    kv_quant_key: str,
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    gguf_tensor_bytes: Optional[dict[str, int]] = None,
    attn_impl: str = "simple",
    fa_bc: int = 128,
    attn_naive_spill: bool = False,
    kv_attn_byte_mode: str = "fp16_equiv_dequant",
    attn_time_scale: float = 1.0,
    attn_time_scale_inv_batch: float = 0.0,
    attn_scale_by_batch: Optional[dict[int, float]] = None,
    attn_scale_by_batch_and_kv: Optional[dict[str, dict[int, float]]] = None,
    weight_tag: Optional[str] = None,
    weight_time_scale_by_tag: Optional[dict[str, float]] = None,
) -> float:
    """Layerwise decode-step prediction: wall ms divided by batch (ms/tok)."""

    hw = dict(HARDWARE_PRESETS[hw_name])
    wbpe = float(weight_bits) / 8.0
    norm_bpe = float(norm_weight_bits) / 8.0
    total_s, _ = simulate_decode_step(
        model,
        batch_size=batch_size,
        ctx_len=ctx_len,
        hw=hw,
        eta=eta,
        weight_bpe=wbpe,
        norm_bpe=norm_bpe,
        kv_quant_key=kv_quant_key,
        kv_group_size=kv_group_size,
        kv_asym=kv_asym,
        gguf_tensor_bytes=gguf_tensor_bytes,
        attn_impl=attn_impl,
        fa_bc=fa_bc,
        attn_naive_spill=attn_naive_spill,
        kv_attn_byte_mode=kv_attn_byte_mode,
        attn_time_scale=attn_time_scale,
        attn_time_scale_inv_batch=attn_time_scale_inv_batch,
        attn_scale_by_batch=attn_scale_by_batch,
        attn_scale_by_batch_and_kv=attn_scale_by_batch_and_kv,
        weight_tag=weight_tag,
        weight_time_scale_by_tag=weight_time_scale_by_tag,
    )
    return total_s * 1000.0 / float(batch_size)


def aggregate_by_family(events: list[SimEvent]) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in events:
        out[e.family] = out.get(e.family, 0.0) + e.seconds
    return out


# ─── Pipeline (whole-step BW) model ──────────────────────────────────────────
#
# Physical motivation: the GPU's HBM bus services all reads in one continuous
# stream while SMs compute in parallel.  The serial per-op roofline
# (sum_i max(T_comp_i, T_mem_i)) overestimates because it treats each op's
# memory access as an independent bottleneck, ignoring the fact that the memory
# controller can prefetch the next op's data while SMs are computing the current
# op.
#
# Whole-step model:
#   step_ms = T_floor/B  +  wt_bytes / eff_BW_wt  +  kv_bytes / eff_BW_kv
#
# where wt_bytes = sum over all layers of all weight tensor bytes (GGUF-exact
# or synthetic), kv_bytes = B × ctx × kv_bytes_per_token (total all layers),
# and eff_BW_{wt,kv} = peak_BW × hw_efficiency × η_{wt,kv}.
#
# η_wt and η_kv are fitted from measurements (see fit_pipeline_eta.py).
# Two separate η because weight GEMV/GEMM and attention KV-read use different
# CUDA kernel paths with different BW efficiency.
#
# Activation bytes (B × d × bpe per proj) are <1% of weight bytes at B≤32
# and are absorbed into the η_wt fit.

def compute_step_bytes(
    m: dict[str, Any],
    batch_size: int,
    ctx_len: int,
    kv_quant_key: str,
    weight_bpe: float,
    norm_bpe: float,
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    gguf_tensor_bytes: Optional[dict[str, int]] = None,
) -> tuple[float, float]:
    """Return (wt_bytes, kv_bytes) for one decode step.

    wt_bytes: total HBM bytes for all weight tensors across all layers.
        Uses GGUF stored sizes if ``gguf_tensor_bytes`` is set; otherwise
        synthetic n_params × bpe per tensor.
    kv_bytes: total HBM bytes for KV cache reads across all layers.
        = batch_size × ctx_len × kv_bytes_per_token(all_layers).
    """
    B = int(batch_size)
    ctx = int(ctx_len)
    nl = int(m["n_layers"])
    d = int(m["d_model"])
    nh = int(m["n_heads"])
    nkv = int(m["n_kv_heads"])
    hd = int(m["head_dim"])
    ffn = int(m["ffn_dim"])
    style = str(m.get("ffn_style", "swiglu"))
    bpe = float(weight_bpe)
    bpe_norm = float(norm_bpe)

    wt_bytes = 0.0
    for layer in range(nl):
        if gguf_tensor_bytes is not None:
            wb = resolve_layer_weight_bytes_from_gguf(layer, m, gguf_tensor_bytes, bpe, bpe_norm)
            wt_bytes += sum(wb.values())
        else:
            # Synthetic: n_params × bpe per tensor
            wt_bytes += (
                d * nh * hd * bpe          # q_proj
                + d * nkv * hd * bpe       # k_proj
                + d * nkv * hd * bpe       # v_proj
                + nh * hd * d * bpe        # o_proj
                + d * bpe_norm             # attn_norm
                + d * bpe_norm             # ffn_norm
            )
            if style == "swiglu":
                wt_bytes += 3 * d * ffn * bpe   # gate + up + down
            else:
                wt_bytes += 2 * d * ffn * bpe   # up + down

    kv_bpt_total = kv_bytes_per_token(kv_quant_key, m, kv_group_size, kv_asym)
    kv_bytes = float(B) * float(ctx) * float(kv_bpt_total)

    return wt_bytes, kv_bytes


def predict_ms_per_token_pipeline(
    model: dict[str, Any],
    *,
    batch_size: int,
    ctx_len: int,
    hw_name: str,
    eta_wt: float,
    eta_kv: float,
    t_floor_ms: float,
    weight_bits: float,
    norm_weight_bits: float = 16.0,
    kv_quant_key: str,
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    gguf_tensor_bytes: Optional[dict[str, int]] = None,
) -> float:
    """Whole-step pipeline prediction: ms per token.

    step_ms  = T_floor/B  +  wt_bytes / eff_BW_wt  +  kv_bytes / eff_BW_kv
    ms/tok   = step_ms / B

    Args:
        eta_wt: BW efficiency for weight reads (0–1].
        eta_kv: BW efficiency for KV cache reads (0–1].
        t_floor_ms: fixed per-step overhead (kernel launch, sync). Divided by B
            in the ms/tok formula, i.e. contributes T_floor_ms / B ms/tok.
    """
    hw = dict(HARDWARE_PRESETS[hw_name])
    wbpe = float(weight_bits) / 8.0
    nbpe = float(norm_weight_bits) / 8.0
    peak_bw = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"]   # bytes/s

    wt_bytes, kv_bytes = compute_step_bytes(
        model, batch_size, ctx_len, kv_quant_key,
        wbpe, nbpe, kv_group_size, kv_asym, gguf_tensor_bytes,
    )

    eff_bw_wt = peak_bw * float(eta_wt)
    eff_bw_kv = peak_bw * float(eta_kv)

    step_ms = (
        float(t_floor_ms)
        + wt_bytes / eff_bw_wt * 1000.0
        + kv_bytes / eff_bw_kv * 1000.0
    )
    return step_ms / float(batch_size)


def predict_verify_step_ms(
    model: dict[str, Any],
    *,
    n_draft_tokens: int,
    ctx_len: int,
    hw_name: str,
    eta_wt: float,
    eta_kv: float,
    t_floor_ms: float,
    weight_bits: float,
    norm_weight_bits: float = 16.0,
    kv_quant_key: str,
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    gguf_tensor_bytes: Optional[dict[str, int]] = None,
) -> float:
    """Total wall-clock ms to verify n_draft_tokens in one parallel forward pass.

    In speculative decoding the verifier processes all K draft tokens in a single
    batched forward pass (K tokens attend to the shared KV cache in parallel).
    The roofline analysis is identical to a batch_size=K decode step:

        verify_step_ms = T_floor + wt_bytes/eff_BW_wt + K×ctx×kv_bpt/eff_BW_kv

    Weight bytes are read **once** regardless of K (amortized across all tokens),
    while KV attention bytes scale linearly with K.  The compute ceiling
    (weight GEMMs flip from memory-bound to compute-bound) is reached only at
    K ≳ peak_TFLOPS / peak_BW × bpe / 2; for Q8_0 on H100-SXM this is ~140 tokens,
    well above typical speculative-decoding draft lengths of 4–16.

    Returns the **total step time in ms** (NOT ms/tok).  The effective throughput
    rate for the verifier is verify_step_ms / n_accepted_tokens, where
    n_accepted_tokens ≤ n_draft_tokens + 1 (including the bonus token).

    For the bootstrap cost model, compare:
        draft_k_ms   = k × draft_decode_ms        (K sequential single-token steps)
        verify_ms    = predict_verify_step_ms(K)   (one parallel verify pass)
    Both produce up to K+1 accepted tokens; verify_ms is cheaper because the
    verifier's weight cost is shared across all K tokens.
    """
    ms_per_tok = predict_ms_per_token_pipeline(
        model,
        batch_size=n_draft_tokens,
        ctx_len=ctx_len,
        hw_name=hw_name,
        eta_wt=eta_wt,
        eta_kv=eta_kv,
        t_floor_ms=t_floor_ms,
        weight_bits=weight_bits,
        norm_weight_bits=norm_weight_bits,
        kv_quant_key=kv_quant_key,
        kv_group_size=kv_group_size,
        kv_asym=kv_asym,
        gguf_tensor_bytes=gguf_tensor_bytes,
    )
    return ms_per_tok * float(n_draft_tokens)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default=None, help="Key in model_structures.json")
    p.add_argument(
        "--structure-json",
        default=None,
        help="JSON file with presets map or single flat model",
    )
    p.add_argument(
        "--catalog",
        default=_DEFAULT_STRUCT,
        help=f"Path to model_structures.json (default: {_DEFAULT_STRUCT})",
    )
    p.add_argument("--batch-size", "-B", type=int, default=1)
    p.add_argument("--ctx-len", type=int, default=4096, help="KV context length for decode step")
    p.add_argument(
        "--sample-ctx",
        type=int,
        default=0,
        metavar="N",
        help="If N>0, override the context length used for the decode-step sim (takes precedence over --ctx-len and derived mid_ctx).",
    )
    p.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        metavar="N",
        help="If N>0 and --decode-len>0, derive mid_ctx = prompt_len + decode_len//2 for decode sim",
    )
    p.add_argument(
        "--decode-len",
        type=int,
        default=0,
        metavar="N",
        help="If N>0 and --prompt-len>0, derive mid_ctx for decode sim and print end-to-end tok/s",
    )
    p.add_argument(
        "--prefill-len",
        type=int,
        default=0,
        metavar="N",
        help="Override prefill length for the prefill estimate block. If 0, uses --prompt-len (if set).",
    )
    p.add_argument(
        "--attn-impl",
        default="simple",
        choices=["simple", "flash"],
        help="Attention model: simple=legacy single attn_core; flash=fused roofline + flash_qk/"
        "flash_softmax/flash_pv (explicit softmax FLOPs; KV+Q bytes)",
    )
    p.add_argument(
        "--fa-bc",
        type=int,
        default=128,
        metavar="N",
        help="FlashAttention KV tile size along sequence (reported as ceil(ctx_len/N) tiles; "
        "does not change roofline unless extended later)",
    )
    p.add_argument(
        "--attn-naive-spill",
        action="store_true",
        help="With --attn-impl simple: add HBM traffic for unfused logits/probs (fp16)",
    )
    p.add_argument("--hw", default="h100-sxm", choices=list(HARDWARE_PRESETS.keys()))
    p.add_argument(
        "--weight-tag",
        default=None,
        metavar="TAG",
        help="Weight quant tag for effective bpw (Q8_0, Q4_K_M, Q2_K, …); overrides structure weight_bits",
    )
    p.add_argument(
        "--weight-bits",
        type=float,
        default=None,
        metavar="F",
        help="Explicit effective bits per weight (overrides --weight-tag and structure)",
    )
    p.add_argument(
        "--norm-weight-bits",
        type=float,
        default=16.0,
        metavar="F",
        help="Bits per RMSNorm gamma (default 16 = fp16; norms often stay fp16 when weights quantized)",
    )
    p.add_argument(
        "--gguf",
        default=None,
        metavar="PATH",
        help="GGUF model file: use per-tensor stored sizes for GEMM/norm weight bytes "
        "(mixed quants); missing tensors use --weight-bits / structure fallback",
    )
    p.add_argument(
        "--kv-type",
        default="f16",
        help="KV cache quant: f16, bf16, q8_0, q4_0, int3_ch, int2_ch, … (see --list-kv-types)",
    )
    p.add_argument(
        "--kv-group-size",
        type=int,
        default=None,
        metavar="N",
        help="Override group_size for kv_bytes_per_token (default: from QUANT_CONFIGS)",
    )
    p.add_argument(
        "--kv-asymmetric",
        action="store_true",
        help="Asymmetric KV quant (doubles scale storage in perf_model)",
    )
    p.add_argument(
        "--list-kv-types",
        action="store_true",
        help="Print KV quant aliases and perf_model keys, then exit",
    )
    p.add_argument(
        "--eta-json",
        default=None,
        metavar="PATH",
        help="JSON from fit_layerwise_eta.py (or any file with an ``eta`` object); base for η",
    )
    p.add_argument("--eta-gemm-comp", type=float, default=None)
    p.add_argument("--eta-gemm-bw", type=float, default=None)
    p.add_argument("--eta-attn-comp", type=float, default=None)
    p.add_argument("--eta-attn-bw", type=float, default=None)
    p.add_argument("--eta-elem-comp", type=float, default=None)
    p.add_argument("--eta-elem-bw", type=float, default=None)
    p.add_argument("--eta-kv-bw", type=float, default=None)
    p.add_argument(
        "--calibration-json",
        default=None,
        metavar="PATH",
        help="From fit_layerwise_calibration.py: "
          "ms/tok = t_floor_ms/B + scale × raw layerwise ms/tok",
    )
    p.add_argument(
        "--sim-physics-json",
        default=None,
        metavar="PATH",
        help="Optional JSON with kv_attn_byte_mode, attn_time_scale, attn_time_scale_inv_batch",
    )
    p.add_argument(
        "--kv-attn-byte-mode",
        choices=["fp16_equiv_dequant", "storage"],
        default=None,
        help="Attention KV read bytes model (default: fp16_equiv_dequant; storage = legacy)",
    )
    p.add_argument(
        "--attn-time-scale",
        type=float,
        default=None,
        metavar="F",
        help="Multiply attn_core time by (F + attn_time_scale_inv_batch / B); default 1",
    )
    p.add_argument(
        "--attn-time-scale-inv-batch",
        type=float,
        default=None,
        metavar="F",
        help="Added to attn scale as F/B (overlap / large-B correction); default 0",
    )
    p.add_argument("--json-out", default=None, help="Write full result JSON to path")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-event lines")
    p.add_argument(
        "--measured-json",
        default=None,
        metavar="PATH",
        help="Optional benchmark_kv_timing JSON (e.g. kv_timing_h100.json). "
        "If provided, prints measured vs predicted ms/tok for the matching slice.",
    )
    args = p.parse_args()

    if args.list_kv_types:
        print("KV --kv-type aliases → perf_model key:")
        for a in sorted(KV_QUANT_ALIASES.keys()):
            print(f"  {a:16s} → {KV_QUANT_ALIASES[a]}")
        print("QUANT_CONFIGS keys (also accepted as --kv-type):")
        print(" ", ", ".join(QUANT_CONFIGS.keys()))
        sys.exit(0)

    pid, model = resolve_model(
        preset=args.preset,
        structure_json=args.structure_json,
        catalog_path=args.catalog,
    )
    validate_structure(model)

    w_bits, w_tag_resolved = resolve_weight_bits(
        model,
        weight_tag=args.weight_tag,
        weight_bits_override=args.weight_bits,
    )
    wbpe = w_bits / 8.0
    norm_bpe = float(args.norm_weight_bits) / 8.0
    kv_key = resolve_kv_quant_key(args.kv_type)

    gguf_tb: Optional[dict[str, int]] = None
    gguf_path: Optional[str] = None
    if args.gguf:
        gguf_path = os.path.abspath(os.path.expanduser(args.gguf))
        gguf_tb = load_gguf_tensor_n_bytes(gguf_path)

    hw = dict(HARDWARE_PRESETS[args.hw])
    eta_base = load_eta_json(args.eta_json) if args.eta_json else Eta()
    eta = Eta(
        gemm_comp=args.eta_gemm_comp if args.eta_gemm_comp is not None else eta_base.gemm_comp,
        gemm_bw=args.eta_gemm_bw if args.eta_gemm_bw is not None else eta_base.gemm_bw,
        attn_comp=args.eta_attn_comp if args.eta_attn_comp is not None else eta_base.attn_comp,
        attn_bw=args.eta_attn_bw if args.eta_attn_bw is not None else eta_base.attn_bw,
        elem_comp=args.eta_elem_comp if args.eta_elem_comp is not None else eta_base.elem_comp,
        elem_bw=args.eta_elem_bw if args.eta_elem_bw is not None else eta_base.elem_bw,
        kv_bw=args.eta_kv_bw if args.eta_kv_bw is not None else eta_base.kv_bw,
    )

    phys = resolve_sim_physics(
        args.sim_physics_json,
        kv_attn_byte_mode=args.kv_attn_byte_mode,
        attn_time_scale=args.attn_time_scale,
        attn_time_scale_inv_batch=args.attn_time_scale_inv_batch,
    )

    prompt_len = int(args.prompt_len)
    decode_len = int(args.decode_len)
    mid_ctx = None
    if prompt_len > 0 and decode_len > 0:
        mid_ctx = prompt_len + decode_len // 2

    ctx_len_used = int(args.ctx_len)
    if mid_ctx is not None:
        ctx_len_used = int(mid_ctx)
    if int(args.sample_ctx) > 0:
        ctx_len_used = int(args.sample_ctx)

    # ─── Footprint summary (model weights + KV cache) ─────────────────────────
    kv_bpt_total = kv_bytes_per_token(
        kv_key,
        dict(model),
        args.kv_group_size,
        bool(args.kv_asymmetric),
    )
    if gguf_tb is not None:
        model_bytes = float(sum(int(v) for v in gguf_tb.values()))
        model_bytes_note = "GGUF sum(all tensors)"
    else:
        # Same synthetic accounting used by compute_step_bytes (weights touched in one decode step).
        wt_step_bytes, _ = compute_step_bytes(
            model,
            batch_size=int(args.batch_size),
            ctx_len=int(ctx_len_used),
            kv_quant_key=kv_key,
            weight_bpe=float(wbpe),
            norm_bpe=float(norm_bpe),
            kv_group_size=args.kv_group_size,
            kv_asym=bool(args.kv_asymmetric),
            gguf_tensor_bytes=None,
        )
        model_bytes = float(wt_step_bytes)
        model_bytes_note = "synthetic decode-step weights"

    print("  Footprint summary:", flush=True)
    print(f"    model:     {_fmt_bytes(model_bytes)}  ({model_bytes_note})", flush=True)
    print(f"    kv/token:  {_fmt_bytes(kv_bpt_total)}  (kv_type={args.kv_type!r})", flush=True)
    if prompt_len > 0 and decode_len > 0:
        kv_start = float(args.batch_size) * float(prompt_len) * float(kv_bpt_total)
        kv_end = float(args.batch_size) * float(prompt_len + decode_len) * float(kv_bpt_total)
        print(f"    kv start:  {_fmt_bytes(kv_start)}  (prompt_len={prompt_len})", flush=True)
        print(f"    kv end:    {_fmt_bytes(kv_end)}  (prompt+decode={prompt_len + decode_len})", flush=True)
    else:
        kv_at_ctx = float(args.batch_size) * float(ctx_len_used) * float(kv_bpt_total)
        print(
            f"    kv @ctx:   {_fmt_bytes(kv_at_ctx)}  (ctx_len_used={ctx_len_used}; "
            "pass --prompt-len and --decode-len for start/end)",
            flush=True,
        )

    total_s, events = simulate_decode_step(
        model,
        batch_size=args.batch_size,
        ctx_len=ctx_len_used,
        hw=hw,
        eta=eta,
        weight_bpe=wbpe,
        norm_bpe=norm_bpe,
        kv_quant_key=kv_key,
        kv_group_size=args.kv_group_size,
        kv_asym=args.kv_asymmetric,
        gguf_tensor_bytes=gguf_tb,
        attn_impl=args.attn_impl,
        fa_bc=args.fa_bc,
        attn_naive_spill=args.attn_naive_spill,
        kv_attn_byte_mode=str(phys["kv_attn_byte_mode"]),
        attn_time_scale=float(phys["attn_time_scale"]),
        attn_time_scale_inv_batch=float(phys["attn_time_scale_inv_batch"]),
        attn_scale_by_batch=phys.get("attn_scale_by_batch"),
        attn_scale_by_batch_and_kv=phys.get("attn_scale_by_batch_and_kv"),
        weight_tag=w_tag_resolved,
        weight_time_scale_by_tag=phys.get("weight_time_scale_by_tag"),
    )

    ms_step = total_s * 1000.0
    ms_per_tok_raw = ms_step / float(args.batch_size)
    cal: Optional[dict[str, Any]] = None
    ms_per_tok = ms_per_tok_raw
    if args.calibration_json:
        cal = load_calibration_json(args.calibration_json)
        ms_per_tok = calibrated_ms_per_token(
            ms_per_tok_raw,
            args.batch_size,
            float(cal["t_floor_ms"]),
            float(cal["scale"]),
        )
    tps = 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0

    by_fam = aggregate_by_family(events)

    print(
        f"preset={pid!r}  hw={args.hw!r}  B={args.batch_size}  ctx_len={args.ctx_len}  "
        f"kv={args.kv_type!r}→{kv_key!r}"
    )
    if mid_ctx is not None:
        print(f"  prompt_len={prompt_len}  decode_len={decode_len}  mid_ctx={mid_ctx}")
    if int(args.sample_ctx) > 0:
        print(f"  sample_ctx={int(args.sample_ctx)}")
    if ctx_len_used != int(args.ctx_len):
        print(f"  ctx_len_used={ctx_len_used}")
    wtag_line = f"weight_bits={w_bits:.2f}"
    if w_tag_resolved:
        wtag_line += f"  (tag={w_tag_resolved!r})"
    elif args.weight_bits is not None:
        wtag_line += "  (explicit --weight-bits)"
    else:
        wtag_line += "  (from structure)"
    if gguf_tb is not None:
        wtag_line += f"  gguf={gguf_path!r} (per-tensor bytes; fallback bpe as above)"
    print(wtag_line + f"  norm_bits={args.norm_weight_bits:g}")
    print(f"layers={model['n_layers']}  d_model={model['d_model']}  ffn_dim={model['ffn_dim']}")
    if cal is not None:
        print(f"  calibration: {args.calibration_json!r}  (t_floor_ms={cal['t_floor_ms']}, scale={cal['scale']})")
    psc = effective_attn_scale(
        int(args.batch_size),
        str(kv_key),
        float(phys["attn_time_scale"]),
        float(phys["attn_time_scale_inv_batch"]),
        phys.get("attn_scale_by_batch"),
        phys.get("attn_scale_by_batch_and_kv"),
    )
    print(
        f"  sim_physics: kv_attn_byte_mode={phys['kv_attn_byte_mode']!r}  "
        f"attn_scale={phys['attn_time_scale']!r}+inv_B*{phys['attn_time_scale_inv_batch']!r} "
        f"→ effective={psc:.4f} @ B={args.batch_size}"
    )
    print()
    print(f"  total_time (one decode step, all layers): {ms_step:.4f} ms")
    print(f"  ms/token (layerwise raw, wall / B):        {ms_per_tok_raw:.4f} ms/tok")
    if cal is not None:
        print(f"  ms/token (calibrated):                    {ms_per_tok:.4f} ms/tok")
    else:
        print(f"  ms/token (wall / B):                       {ms_per_tok:.4f} ms/tok")
    print(f"  tok/s (reported):                          {tps:.2f}")
    # ─── Compare to measured (if provided) ────────────────────────────────────
    if args.measured_json:
        mpath = os.path.abspath(os.path.expanduser(args.measured_json))
        try:
            with open(mpath, encoding="utf-8") as f:
                mdata = json.load(f)
        except Exception as e:
            print(f"  [measured] failed to read {mpath!r}: {e}", file=sys.stderr)
            mdata = None
        rows = []
        if isinstance(mdata, list):
            rows = mdata
        elif isinstance(mdata, dict):
            rows = mdata.get("rows", []) if isinstance(mdata.get("rows", []), list) else []
        target_mid = int(mid_ctx) if mid_ctx is not None else int(ctx_len_used)
        want_kv = str(args.kv_type)
        want_wtag = str(args.weight_tag or "")
        best = None
        best_key = None
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("model_preset", "")) != str(pid):
                continue
            if int(r.get("batch_size", -1)) != int(args.batch_size):
                continue
            if str(r.get("kv_type", "")) != want_kv:
                continue
            if want_wtag and str(r.get("weight_tag", "")) != want_wtag:
                continue
            pl = int(r.get("prompt_len", 0))
            dl = int(r.get("decode_len", 0))
            r_mid = int(r.get("mid_ctx", pl + dl // 2))
            dmid = abs(r_mid - target_mid)
            key = (dmid, pl, dl)
            if best is None or key < best_key:
                best = r
                best_key = key
        if best is None:
            print(
                f"  [measured] no matching row found in {mpath!r} for "
                f"preset={pid!r} kv_type={want_kv!r} weight_tag={want_wtag!r} "
                f"B={args.batch_size} mid_ctx≈{target_mid}",
                flush=True,
            )
        else:
            meas = float(best.get("measured_ms", float("nan")))
            pl = int(best.get("prompt_len", 0))
            dl = int(best.get("decode_len", 0))
            r_mid = int(best.get("mid_ctx", pl + dl // 2))
            pred = float(ms_per_tok)
            if not math.isnan(meas) and meas > 0:
                ratio = pred / meas
                err = pred - meas
                print(
                    f"  [measured] measured={meas:.4f} ms/tok  predicted={pred:.4f} ms/tok  "
                    f"err={err:+.4f}  pred/meas={ratio:.3f}  (row mid_ctx={r_mid}, prompt_len={pl}, decode_len={dl})",
                    flush=True,
                )
            else:
                print(
                    f"  [measured] row found but measured_ms invalid: {best.get('measured_ms')!r}  "
                    f"(row mid_ctx={r_mid}, prompt_len={pl}, decode_len={dl})",
                    flush=True,
                )
    print()
    print("  Time by op family (s, fraction of total):")
    for fam in sorted(by_fam.keys(), key=lambda k: -by_fam[k]):
        s = by_fam[fam]
        print(f"    {fam:12s}  {s*1000.0:10.4f} ms  ({100.0*s/total_s:5.1f}%)")

    if args.verbose:
        print()
        print("  Per-event (layer, name, ms):")
        for e in events:
            print(f"    L{e.layer:3d}  {e.name:20s}  {e.seconds*1000.0:10.6f} ms  [{e.family}]")

    # ─── Prefill estimate (perf_model) ────────────────────────────────────────
    # prompt_len == prefill_len in meaning; keep --prefill-len as an override.
    prefill_n = int(args.prefill_len) if int(args.prefill_len) > 0 else int(prompt_len)
    prefill_s_seq: Optional[float] = None
    prefill_s_step: Optional[float] = None
    prefill_tok_s: Optional[float] = None
    if prefill_n > 0:
        prefill_s_seq = float(prefill_time(prefill_n, model, hw))
        # Approximation: prefill flops scale with number of sequences in the batch.
        prefill_s_step = prefill_s_seq * float(args.batch_size)
        prefill_tok_s = (float(prefill_n) * float(args.batch_size)) / prefill_s_step if prefill_s_step > 0 else None
        print()
        print("  Prefill estimate (compute-bound, perf_model):")
        print(f"    prompt_len:                     {prefill_n}")
        print(f"    prefill_time per sequence:      {prefill_s_seq*1000.0:.3f} ms")
        print(f"    prefill_time (B sequences):     {prefill_s_step*1000.0:.3f} ms")
        if prefill_tok_s is not None:
            print(f"    prefill throughput (tok/s):     {prefill_tok_s:.1f}")

    # ─── End-to-end throughput for (prompt_len, decode_len) ───────────────────
    # Uses:
    #   - prefill_time() (compute-bound) for prompt_len tokens
    #   - layerwise decode step time at mid_ctx (wall per step = ms_step)
    # Total tokens processed = (prompt_len + decode_len) * B
    # Total wall time ≈ prefill_time(prompt_len)*B + decode_len * ms_step
    if prompt_len > 0 and decode_len > 0:
        t_pref_seq = float(prefill_time(prompt_len, model, hw))
        t_pref = t_pref_seq * float(args.batch_size)
        t_dec = float(decode_len) * (ms_step / 1000.0)
        t_tot = t_pref + t_dec
        toks = float(prompt_len + decode_len) * float(args.batch_size)
        sys_tps = toks / t_tot if t_tot > 0 else float("nan")
        print()
        print("  End-to-end estimate (prefill + decode):")
        print(f"    prompt_len:                     {prompt_len}")
        print(f"    decode_len:                      {decode_len}")
        print(f"    prefill_time (B sequences):     {t_pref*1000.0:.3f} ms")
        print(f"    decode_time (B sequences):      {t_dec*1000.0:.3f} ms")
        print(f"    total_time (B sequences):       {t_tot*1000.0:.3f} ms")
        print(f"    throughput (tok/s, prompt+dec): {sys_tps:.1f}")

    out: dict[str, Any] = {
        "preset": pid,
        "hardware": args.hw,
        "batch_size": args.batch_size,
        "ctx_len": int(args.ctx_len),
        "prompt_len": int(prompt_len),
        "decode_len": int(decode_len),
        "mid_ctx": int(mid_ctx) if mid_ctx is not None else None,
        "sample_ctx": int(args.sample_ctx) if int(args.sample_ctx) > 0 else None,
        "ctx_len_used": int(ctx_len_used),
        "kv_type_cli": args.kv_type,
        "kv_quant_key": kv_key,
        "kv_group_size": args.kv_group_size,
        "kv_asymmetric": bool(args.kv_asymmetric),
        "weight_bits_effective": round(w_bits, 6),
        "weight_tag": w_tag_resolved,
        "weight_source": "gguf" if gguf_tb is not None else "uniform_bpe",
        "gguf_path": gguf_path,
        "norm_weight_bits": float(args.norm_weight_bits),
        "model": {k: model[k] for k in model if not str(k).startswith("_")},
        "attn_impl": args.attn_impl,
        "fa_bc": int(args.fa_bc),
        "attn_naive_spill": bool(args.attn_naive_spill),
        "sim_physics": dict(phys),
        "attn_time_scale_effective": round(psc, 6),
        "kv_tiles_along_seq": (int(args.ctx_len) + int(args.fa_bc) - 1) // int(args.fa_bc),
        "total_ms_per_step": round(ms_step, 6),
        "ms_per_token_raw_layerwise": round(ms_per_tok_raw, 6),
        "ms_per_token": round(ms_per_tok, 6),
        "tok_per_s": round(tps, 4),
        "prefill_len": int(prefill_n),
        "prefill_time_s_per_seq": round(prefill_s_seq, 9) if prefill_s_seq is not None else None,
        "prefill_time_s_for_batch": round(prefill_s_step, 9) if prefill_s_step is not None else None,
        "prefill_tok_per_s": round(prefill_tok_s, 6) if prefill_tok_s is not None else None,
        "system_tok_per_s_prompt_plus_decode": round(sys_tps, 6) if (prompt_len > 0 and decode_len > 0 and not math.isnan(sys_tps)) else None,
        "seconds_by_family": {k: round(v, 9) for k, v in by_fam.items()},
        "events": [
            {
                "layer": e.layer,
                "name": e.name,
                "family": e.family,
                "flops": round(e.flops, 2),
                "bytes": round(e.bytes_moved, 2),
                "seconds": round(e.seconds, 9),
                "batch_scaling": e.batch_scaling,
            }
            for e in events
        ],
    }
    if cal is not None:
        out["calibration_file"] = args.calibration_json
        out["calibration"] = {
            "t_floor_ms": cal["t_floor_ms"],
            "scale": cal["scale"],
        }

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
