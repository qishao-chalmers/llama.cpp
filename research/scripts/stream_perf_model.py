#!/usr/bin/env python3
"""Decode-only resource-stream performance model (experimental).

This is intentionally separate from layerwise_roofline_sim.py.

Core predictor (seconds per decode step for batch B):

    T_step = T_fixed_s(B)
           + max(T_compute_s, T_weight_s, T_kv_s)
           + alpha_wm * min(T_compute_s, T_weight_s)
           + alpha_mk * min(max(T_compute_s, T_weight_s), T_kv_s)
           + T_tail_s

    ms_per_tok = 1000 * T_step / B

The two alpha terms are optional (default 0). They approximate partial overlap /
non-fused tails without returning to a full serial sum across all ops.

Where:
  T_weight_s  = weight_bytes / (peak_mem_Bps * eta_weight)
  T_kv_s      = kv_attn_read_bytes / (peak_mem_Bps * eta_kv[kv])
  T_compute_s = flops / (peak_flops * eta_compute)

Features:
  - weight_bytes: GGUF-exact per-layer tensor sizes when gguf_tensor_bytes is provided
  - kv_attn_read_bytes: kv_attn_read_bpt_layer × n_layers × B × ctx_len
    (matches layerwise attention KV read: bytes scale with context length)
  - flops: summed from layerwise_roofline_sim.simulate_decode_step events (family totals)

Calibration JSON is produced by fit_stream_perf_tables.py.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import layerwise_roofline_sim as sim  # noqa: E402
from gguf_layerwise_weights import resolve_layer_weight_bytes_from_gguf  # noqa: E402


def _load_json(path: str) -> dict[str, Any]:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return json.load(f)


def load_stream_calib(path: str) -> dict[str, Any]:
    return _load_json(path)


def default_calib(hw: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": "stream_decode_v1",
        "hw": hw,
        "kv_attn_byte_mode": "fp16_equiv_dequant",
        "fixed_overhead_ms_by_batch": {},
        "tail_ms": 0.0,
        "eta_weight_bw": 0.40,
        "eta_kv_bw_by_kv": {"fp16": 0.30, "int8_ch": 0.28, "int4_ch": 0.28},
        "eta_compute": 0.35,
        "alpha_wm": 0.0,
        "alpha_mk": 0.0,
    }


def stream_calib_report_warnings(
    cal: dict[str, Any],
    *,
    hw_name: str,
    attn_impl: str,
    fa_bc: int,
    attn_naive_spill: bool,
    kv_attn_byte_mode_cli: Optional[str],
) -> list[str]:
    """Return warnings when calibration metadata disagrees with report/fit CLI options."""

    out: list[str] = []
    cal_hw = cal.get("hw")
    if cal_hw is not None and str(cal_hw) != str(hw_name):
        out.append(f"calibration hw={cal_hw!r} differs from requested hw={hw_name!r}")

    cal_kv_mode = cal.get("kv_attn_byte_mode")
    if kv_attn_byte_mode_cli is not None and cal_kv_mode is not None:
        if str(cal_kv_mode) != str(kv_attn_byte_mode_cli):
            out.append(
                f"calibration kv_attn_byte_mode={cal_kv_mode!r} differs from "
                f"CLI override {kv_attn_byte_mode_cli!r} (predictions use override bytes model)"
            )

    fit = cal.get("fit")
    if not isinstance(fit, dict):
        return out

    if fit.get("attn_impl") is not None and str(fit.get("attn_impl")) != str(attn_impl):
        out.append(
            f"calibration fit.attn_impl={fit.get('attn_impl')!r} differs from "
            f"requested {attn_impl!r}"
        )
    if fit.get("fa_bc") is not None and int(fit.get("fa_bc")) != int(fa_bc):
        out.append(
            f"calibration fit.fa_bc={fit.get('fa_bc')!r} differs from requested {fa_bc}"
        )
    if fit.get("attn_naive_spill") is not None and bool(fit.get("attn_naive_spill")) != bool(
        attn_naive_spill
    ):
        out.append(
            f"calibration fit.attn_naive_spill={fit.get('attn_naive_spill')!r} differs from "
            f"requested {attn_naive_spill!r}"
        )
    return out


def mid_ctx(r: dict[str, Any]) -> int:
    pl = int(r.get("prompt_len", 0) or 0)
    dl = int(r.get("decode_len", 0) or 0)
    return int(r.get("mid_ctx", pl + dl // 2))


def weight_bytes_per_decode_step_gguf(
    model: dict[str, Any],
    gguf_tb: dict[str, int],
    *,
    weight_bpe: float,
    norm_bpe: float,
) -> float:
    nl = int(model["n_layers"])
    tot = 0.0
    for layer in range(nl):
        mp = resolve_layer_weight_bytes_from_gguf(layer, model, gguf_tb, weight_bpe, norm_bpe)
        tot += sum(float(v) for v in mp.values())
    return float(tot)


def weight_bytes_per_decode_step_uniform(model: dict[str, Any], weight_bpe: float, norm_bpe: float) -> float:
    # Use GGUF mapping path with an empty table → synthetic bytes from bpw (same as layerwise fallback).
    return weight_bytes_per_decode_step_gguf(model, {}, weight_bpe=weight_bpe, norm_bpe=norm_bpe)


def kv_attn_read_bytes_per_decode_step(
    model: dict[str, Any],
    *,
    kv_quant_key: str,
    batch_size: int,
    ctx_len: int,
    kv_attn_byte_mode: str,
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
) -> float:
    nl = int(model["n_layers"])
    bpt_layer = sim.kv_attn_read_bpt_layer(
        kv_quant_key,
        model,
        nl,
        kv_group_size=kv_group_size,
        kv_asym=kv_asym,
        mode=str(kv_attn_byte_mode),
    )
    # Same scaling as layerwise_roofline_sim attention: kv_attn_bpt_layer * ctx * B per layer.
    return float(bpt_layer) * float(nl) * float(batch_size) * float(ctx_len)


def flops_by_family_from_sim(
    model: dict[str, Any],
    *,
    batch_size: int,
    ctx_len: int,
    hw_name: str,
    kv_quant_key: str,
    weight_bpe: float,
    norm_bpe: float,
    gguf_tensor_bytes: Optional[dict[str, int]],
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    attn_impl: str = "simple",
    fa_bc: int = 128,
    attn_naive_spill: bool = False,
    kv_attn_byte_mode: str = "fp16_equiv_dequant",
) -> dict[str, float]:
    """Return summed FLOPs by event family for one decode step.

    Uses simulate_decode_step with neutral attention scaling; FLOPs are independent of η,
    but we pass a fixed Eta() for completeness.
    """

    hw = dict(sim.HARDWARE_PRESETS[hw_name])
    eta0 = sim.Eta()
    total_s, events = sim.simulate_decode_step(
        model,
        batch_size=int(batch_size),
        ctx_len=int(ctx_len),
        hw=hw,
        eta=eta0,
        weight_bpe=float(weight_bpe),
        norm_bpe=float(norm_bpe),
        kv_quant_key=str(kv_quant_key),
        kv_group_size=kv_group_size,
        kv_asym=bool(kv_asym),
        gguf_tensor_bytes=gguf_tensor_bytes,
        attn_impl=str(attn_impl),
        fa_bc=int(fa_bc),
        attn_naive_spill=bool(attn_naive_spill),
        kv_attn_byte_mode=str(kv_attn_byte_mode),
        attn_time_scale=1.0,
        attn_time_scale_inv_batch=0.0,
        attn_scale_by_batch=None,
        attn_scale_by_batch_and_kv=None,
        weight_tag=None,
        weight_time_scale_by_tag=None,
    )
    _ = total_s
    out: dict[str, float] = {}
    for e in events:
        out[e.family] = out.get(e.family, 0.0) + float(e.flops)
    return out


def total_compute_flops(flops_by_family: dict[str, float]) -> float:
    # For decode, treat all non-I/O-ish compute as part of the compute stream.
    # kv_rw is usually tiny in flops; include it to avoid negative surprises.
    fams = ("gemm", "attn_core", "elementwise", "kv_rw")
    return float(sum(float(flops_by_family.get(f, 0.0)) for f in fams))


def extract_decode_features(
    model: dict[str, Any],
    *,
    batch_size: int,
    ctx_len: int,
    hw_name: str,
    kv_quant_key: str,
    weight_bpe: float,
    norm_bpe: float,
    gguf_tensor_bytes: Optional[dict[str, int]],
    kv_group_size: Optional[int] = None,
    kv_asym: bool = False,
    attn_impl: str = "simple",
    fa_bc: int = 128,
    attn_naive_spill: bool = False,
    kv_attn_byte_mode: str = "fp16_equiv_dequant",
) -> dict[str, Any]:
    if gguf_tensor_bytes is not None:
        wbytes = weight_bytes_per_decode_step_gguf(
            model, gguf_tensor_bytes, weight_bpe=float(weight_bpe), norm_bpe=float(norm_bpe)
        )
        wsrc = "gguf"
    else:
        wbytes = weight_bytes_per_decode_step_uniform(model, float(weight_bpe), float(norm_bpe))
        wsrc = "uniform_bpw"

    kv_bytes = kv_attn_read_bytes_per_decode_step(
        model,
        kv_quant_key=str(kv_quant_key),
        batch_size=int(batch_size),
        ctx_len=int(ctx_len),
        kv_attn_byte_mode=str(kv_attn_byte_mode),
        kv_group_size=kv_group_size,
        kv_asym=kv_asym,
    )

    fbf = flops_by_family_from_sim(
        model,
        batch_size=int(batch_size),
        ctx_len=int(ctx_len),
        hw_name=str(hw_name),
        kv_quant_key=str(kv_quant_key),
        weight_bpe=float(weight_bpe),
        norm_bpe=float(norm_bpe),
        gguf_tensor_bytes=gguf_tensor_bytes,
        kv_group_size=kv_group_size,
        kv_asym=kv_asym,
        attn_impl=str(attn_impl),
        fa_bc=int(fa_bc),
        attn_naive_spill=bool(attn_naive_spill),
        kv_attn_byte_mode=str(kv_attn_byte_mode),
    )
    flops = total_compute_flops(fbf)

    return {
        "weight_bytes": float(wbytes),
        "kv_attn_read_bytes": float(kv_bytes),
        "flops": float(flops),
        "flops_by_family": {k: float(v) for k, v in fbf.items()},
        "weight_source": wsrc,
    }


def _fixed_overhead_s(cal: dict[str, Any], batch_size: int) -> float:
    m = cal.get("fixed_overhead_ms_by_batch") or {}
    if not isinstance(m, dict):
        return 0.0
    v = m.get(str(int(batch_size)), m.get(int(batch_size)))
    if v is None:
        return 0.0
    return float(v) / 1000.0


def predict_decode_ms_per_tok(
    feats: dict[str, Any],
    *,
    batch_size: int,
    hw_name: str,
    kv_quant_key: str,
    cal: dict[str, Any],
) -> float:
    hw = dict(sim.HARDWARE_PRESETS[str(hw_name)])
    peak_bw = float(hw["memory_bw_gbps"]) * 1e9 * float(hw["efficiency"])
    peak_flops = float(hw["compute_tflops"]) * 1e12 * float(hw["efficiency"])

    eta_w = float(cal.get("eta_weight_bw", 0.40))
    eta_c = float(cal.get("eta_compute", 0.35))
    eta_kv_map = cal.get("eta_kv_bw_by_kv") or {}
    if not isinstance(eta_kv_map, dict):
        eta_kv_map = {}
    eta_kv = float(eta_kv_map.get(str(kv_quant_key), eta_kv_map.get("default", 0.30)))

    w_b = float(feats["weight_bytes"])
    kv_b = float(feats["kv_attn_read_bytes"])
    flops = float(feats["flops"])

    tw = (w_b / (peak_bw * eta_w)) if (peak_bw > 0 and eta_w > 0) else float("inf")
    tk = (kv_b / (peak_bw * eta_kv)) if (peak_bw > 0 and eta_kv > 0) else float("inf")
    tc = (flops / (peak_flops * eta_c)) if (peak_flops > 0 and eta_c > 0) else 0.0

    a = sorted([tc, tw, tk])
    t_small, t_mid, t_large = float(a[0]), float(a[1]), float(a[2])
    t_max = t_large
    alpha_wm = float(cal.get("alpha_wm", 0.0))
    alpha_mk = float(cal.get("alpha_mk", 0.0))
    t_step = (
        _fixed_overhead_s(cal, int(batch_size))
        + t_max
        + alpha_wm * min(tc, tw)
        + alpha_mk * min(max(tc, tw), tk)
        + float(cal.get("tail_ms", 0.0)) / 1000.0
    )
    return float(t_step) * 1000.0 / float(batch_size)


def dominant_stream(
    feats: dict[str, Any],
    *,
    batch_size: int,
    hw_name: str,
    kv_quant_key: str,
    cal: dict[str, Any],
) -> str:
    hw = dict(sim.HARDWARE_PRESETS[str(hw_name)])
    peak_bw = float(hw["memory_bw_gbps"]) * 1e9 * float(hw["efficiency"])
    peak_flops = float(hw["compute_tflops"]) * 1e12 * float(hw["efficiency"])

    eta_w = float(cal.get("eta_weight_bw", 0.40))
    eta_c = float(cal.get("eta_compute", 0.35))
    eta_kv_map = cal.get("eta_kv_bw_by_kv") or {}
    if not isinstance(eta_kv_map, dict):
        eta_kv_map = {}
    eta_kv = float(eta_kv_map.get(str(kv_quant_key), eta_kv_map.get("default", 0.30)))

    w_b = float(feats["weight_bytes"])
    kv_b = float(feats["kv_attn_read_bytes"])
    flops = float(feats["flops"])

    tw = (w_b / (peak_bw * eta_w)) if (peak_bw > 0 and eta_w > 0) else float("-inf")
    tk = (kv_b / (peak_bw * eta_kv)) if (peak_bw > 0 and eta_kv > 0) else float("-inf")
    tc = (flops / (peak_flops * eta_c)) if (peak_flops > 0 and eta_c > 0) else 0.0

    m = max(tc, tw, tk)
    if m == tc:
        return "compute"
    if m == tw:
        return "weight"
    return "kv"


def safe_rmse_mae(pred: list[float], meas: list[float]) -> tuple[float, float]:
    e = [p - m for p, m in zip(pred, meas)]
    mae = sum(abs(x) for x in e) / max(1, len(e))
    rmse = math.sqrt(sum(x * x for x in e) / max(1, len(e)))
    return mae, rmse
