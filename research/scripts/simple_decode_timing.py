#!/usr/bin/env python3
"""Simple GGUF-native decode timing estimator.

Design goals:
1. Read model dimensions directly from a local GGUF file.
2. Use actual per-tensor bytes from GGUF tensor records (supports mixed quant layers).
3. Estimate decode-step time as sum of per-op roofline times with fixed built-in efficiencies.
4. Emit sweep tables in a text layout compatible with decode_sweep*.txt for vimdiff.
5. Emit per-op detail (FLOPs, bytes, arithmetic intensity, estimated time, ideal min time).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_GGUF_PY_DIR = _REPO_ROOT / "gguf-py"
sys.path.insert(0, str(_GGUF_PY_DIR))

from gguf import GGUFReader  # type: ignore


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    compute_tflops: float
    mem_gbps: float


@dataclass(frozen=True)
class EffSpec:
    comp_eff: float
    mem_eff: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str
    n_layers: int
    d_model: int
    ffn_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    value_dim: int


@dataclass
class OpRecord:
    layer: int
    op: str
    family: str
    category: str
    flops: float
    bytes_total: float
    arithmetic_intensity: float
    compute_ms_est: float
    memory_ms_est: float
    est_ms: float
    compute_ms_min: float
    memory_ms_min: float
    min_ms: float


HARDWARE_PRESETS: dict[str, HardwareSpec] = {
    "h100-sxm": HardwareSpec("h100-sxm", compute_tflops=989.0, mem_gbps=3350.0),
    "a100-80g": HardwareSpec("a100-80g", compute_tflops=312.0, mem_gbps=2000.0),
    "rtx4090": HardwareSpec("rtx4090", compute_tflops=165.0, mem_gbps=1008.0),
    "a6000": HardwareSpec("a6000", compute_tflops=154.0, mem_gbps=768.0),
}


DEFAULT_EFF: dict[str, EffSpec] = {
    "gemm": EffSpec(comp_eff=0.36, mem_eff=0.42),
    "attn": EffSpec(comp_eff=0.30, mem_eff=0.34),
    "rope": EffSpec(comp_eff=0.35, mem_eff=0.50),
    "norm": EffSpec(comp_eff=0.30, mem_eff=0.45),
    "elem": EffSpec(comp_eff=0.30, mem_eff=0.50),
    "other": EffSpec(comp_eff=0.25, mem_eff=0.35),
}


ACT_BPE = 2.0  # fp16 activations
SCORE_BPE = 2.0  # fp16 attention score/intermediate assumption


def _product(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _tensor_bytes_map(reader: GGUFReader) -> dict[str, int]:
    return {t.name: int(t.n_bytes) for t in reader.tensors}


def _tensor_shape_map(reader: GGUFReader) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for t in reader.tensors:
        out[t.name] = [int(x) for x in t.shape]
    return out


def _field(reader: GGUFReader, key: str, required: bool = True) -> Any:
    f = reader.get_field(key)
    if f is None:
        if required:
            raise KeyError(f"missing GGUF field: {key}")
        return None
    return f.contents()


def load_model_spec_from_gguf(model_path: str) -> tuple[ModelSpec, dict[str, int], dict[str, list[int]], float]:
    reader = GGUFReader(model_path)
    arch = str(_field(reader, "general.architecture"))
    name = str(_field(reader, "general.name", required=False) or os.path.basename(model_path))

    n_layers = int(_field(reader, f"{arch}.block_count"))
    d_model = int(_field(reader, f"{arch}.embedding_length"))
    ffn_dim = int(_field(reader, f"{arch}.feed_forward_length"))
    n_heads = int(_field(reader, f"{arch}.attention.head_count"))
    n_kv_heads = int(_field(reader, f"{arch}.attention.head_count_kv", required=False) or n_heads)
    head_dim = int(_field(reader, f"{arch}.attention.key_length", required=False) or (d_model // n_heads))
    value_dim = int(_field(reader, f"{arch}.attention.value_length", required=False) or head_dim)

    tensor_bytes = _tensor_bytes_map(reader)
    tensor_shapes = _tensor_shape_map(reader)

    # KV bytes/value can be inferred from tensor metadata only when kv cache type is known at runtime.
    # Here we default to fp16 for attention/KV cache math; caller can override with --kv-bits.
    default_kv_bpe = 2.0

    spec = ModelSpec(
        name=name,
        arch=arch,
        n_layers=n_layers,
        d_model=d_model,
        ffn_dim=ffn_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        value_dim=value_dim,
    )
    return spec, tensor_bytes, tensor_shapes, default_kv_bpe


def _layer_tensor_name(layer: int, suffix: str) -> str:
    return f"blk.{layer}.{suffix}"


def _tensor_bytes(tensor_bytes: dict[str, int], layer: int, suffix: str, fallback: float) -> float:
    return float(tensor_bytes.get(_layer_tensor_name(layer, suffix), fallback))


def _tensor_shape(tensor_shapes: dict[str, list[int]], layer: int, suffix: str) -> Optional[list[int]]:
    return tensor_shapes.get(_layer_tensor_name(layer, suffix))


def _gemm_flops_from_shape(shape: Optional[list[int]], fallback_in: int, fallback_out: int, batch_size: int) -> float:
    if shape is not None and len(shape) >= 2:
        in_dim = int(shape[0])
        out_dim = int(shape[1])
    else:
        in_dim = int(fallback_in)
        out_dim = int(fallback_out)
    return 2.0 * float(batch_size) * float(in_dim) * float(out_dim)


def _calc_times_ms(flops: float, bytes_total: float, hw: HardwareSpec, eff: EffSpec) -> tuple[float, float, float, float, float, float]:
    peak_flops = hw.compute_tflops * 1e12
    peak_bw = hw.mem_gbps * 1e9
    compute_ms_min = 1e3 * flops / peak_flops if flops > 0 else 0.0
    memory_ms_min = 1e3 * bytes_total / peak_bw if bytes_total > 0 else 0.0
    min_ms = max(compute_ms_min, memory_ms_min)

    compute_ms_est = compute_ms_min / eff.comp_eff if eff.comp_eff > 0 else float("inf")
    memory_ms_est = memory_ms_min / eff.mem_eff if eff.mem_eff > 0 else float("inf")
    est_ms = max(compute_ms_est, memory_ms_est)
    return compute_ms_est, memory_ms_est, est_ms, compute_ms_min, memory_ms_min, min_ms


def _op(
    layer: int,
    op: str,
    family: str,
    category: str,
    flops: float,
    bytes_total: float,
    hw: HardwareSpec,
    eff: EffSpec,
) -> OpRecord:
    ai = (flops / bytes_total) if bytes_total > 0 else float("inf")
    c_est, m_est, est, c_min, m_min, min_ms = _calc_times_ms(flops, bytes_total, hw, eff)
    return OpRecord(
        layer=layer,
        op=op,
        family=family,
        category=category,
        flops=flops,
        bytes_total=bytes_total,
        arithmetic_intensity=ai,
        compute_ms_est=c_est,
        memory_ms_est=m_est,
        est_ms=est,
        compute_ms_min=c_min,
        memory_ms_min=m_min,
        min_ms=min_ms,
    )


def simulate_decode_step(
    spec: ModelSpec,
    tensor_bytes: dict[str, int],
    tensor_shapes: dict[str, list[int]],
    hw: HardwareSpec,
    batch_size: int,
    ctx_len: int,
    kv_bpe: float,
    per_step_other_ms: float,
) -> list[OpRecord]:
    B = int(batch_size)
    L = int(ctx_len)
    d = spec.d_model
    d_q = spec.n_heads * spec.head_dim
    d_k = spec.n_kv_heads * spec.head_dim
    d_v = spec.n_kv_heads * spec.value_dim
    ffn = spec.ffn_dim

    ops: list[OpRecord] = []

    # Amortize fixed overhead equally across layers for per-layer accounting.
    other_per_layer = per_step_other_ms / float(spec.n_layers) if spec.n_layers > 0 else 0.0

    for layer in range(spec.n_layers):
        # Attention RMSNorm
        norm_w = _tensor_bytes(tensor_bytes, layer, "attn_norm.weight", fallback=d * 4.0)
        norm_flops = 8.0 * B * d
        norm_bytes = norm_w + (2.0 * B * d * ACT_BPE)
        ops.append(_op(layer, "attn_norm", "norm", "Norm", norm_flops, norm_bytes, hw, DEFAULT_EFF["norm"]))

        # Q projection
        q_w = _tensor_bytes(tensor_bytes, layer, "attn_q.weight", fallback=d * d)
        q_shape = _tensor_shape(tensor_shapes, layer, "attn_q.weight")
        q_flops = _gemm_flops_from_shape(q_shape, d, d_q, B)
        q_bytes = q_w + (B * d + B * d_q) * ACT_BPE
        ops.append(_op(layer, "attn_q", "gemm", "QKV+O", q_flops, q_bytes, hw, DEFAULT_EFF["gemm"]))

        # K projection
        k_w = _tensor_bytes(tensor_bytes, layer, "attn_k.weight", fallback=d * d_k)
        k_shape = _tensor_shape(tensor_shapes, layer, "attn_k.weight")
        k_flops = _gemm_flops_from_shape(k_shape, d, d_k, B)
        k_bytes = k_w + (B * d + B * d_k) * ACT_BPE
        ops.append(_op(layer, "attn_k", "gemm", "QKV+O", k_flops, k_bytes, hw, DEFAULT_EFF["gemm"]))

        # V projection
        v_w = _tensor_bytes(tensor_bytes, layer, "attn_v.weight", fallback=d * d_v)
        v_shape = _tensor_shape(tensor_shapes, layer, "attn_v.weight")
        v_flops = _gemm_flops_from_shape(v_shape, d, d_v, B)
        v_bytes = v_w + (B * d + B * d_v) * ACT_BPE
        ops.append(_op(layer, "attn_v", "gemm", "QKV+O", v_flops, v_bytes, hw, DEFAULT_EFF["gemm"]))

        # Optional q/k norm weights (present in Qwen3)
        qn_w = _tensor_bytes(tensor_bytes, layer, "attn_q_norm.weight", fallback=spec.head_dim * 4.0)
        qn_flops = 4.0 * B * d_q
        qn_bytes = qn_w + (2.0 * B * d_q * ACT_BPE)
        ops.append(_op(layer, "attn_q_norm", "norm", "Norm", qn_flops, qn_bytes, hw, DEFAULT_EFF["norm"]))

        kn_w = _tensor_bytes(tensor_bytes, layer, "attn_k_norm.weight", fallback=spec.head_dim * 4.0)
        kn_flops = 4.0 * B * d_k
        kn_bytes = kn_w + (2.0 * B * d_k * ACT_BPE)
        ops.append(_op(layer, "attn_k_norm", "norm", "Norm", kn_flops, kn_bytes, hw, DEFAULT_EFF["norm"]))

        # RoPE on q and k (new token positions only)
        rope_flops = 4.0 * B * (d_q + d_k)
        rope_bytes = 2.0 * B * (d_q + d_k) * ACT_BPE
        ops.append(_op(layer, "rope", "rope", "RoPE", rope_flops, rope_bytes, hw, DEFAULT_EFF["rope"]))

        # KV cache write (1 new token per sequence)
        kvw_flops = 0.0
        kvw_bytes = B * (d_k + d_v) * kv_bpe
        ops.append(_op(layer, "kv_write", "other", "Other", kvw_flops, kvw_bytes, hw, DEFAULT_EFF["other"]))

        # Attention QK
        qk_flops = 2.0 * B * spec.n_heads * L * spec.head_dim
        qk_bytes = (B * d_q * ACT_BPE) + (B * L * d_k * kv_bpe) + (B * spec.n_heads * L * SCORE_BPE)
        ops.append(_op(layer, "attn_qk", "attn", "Attn", qk_flops, qk_bytes, hw, DEFAULT_EFF["attn"]))

        # Attention softmax
        sm_flops = 5.0 * B * spec.n_heads * L
        sm_bytes = 2.0 * B * spec.n_heads * L * SCORE_BPE
        ops.append(_op(layer, "attn_softmax", "attn", "Attn", sm_flops, sm_bytes, hw, DEFAULT_EFF["attn"]))

        # Attention PV
        pv_flops = 2.0 * B * spec.n_heads * L * spec.value_dim
        pv_bytes = (B * spec.n_heads * L * SCORE_BPE) + (B * L * d_v * kv_bpe) + (B * d_q * ACT_BPE)
        ops.append(_op(layer, "attn_pv", "attn", "Attn", pv_flops, pv_bytes, hw, DEFAULT_EFF["attn"]))

        # Output projection
        o_w = _tensor_bytes(tensor_bytes, layer, "attn_output.weight", fallback=d * d)
        o_shape = _tensor_shape(tensor_shapes, layer, "attn_output.weight")
        o_flops = _gemm_flops_from_shape(o_shape, d_q, d, B)
        o_bytes = o_w + (B * d_q + B * d) * ACT_BPE
        ops.append(_op(layer, "attn_output", "gemm", "QKV+O", o_flops, o_bytes, hw, DEFAULT_EFF["gemm"]))

        # Residual add after attention
        r1_flops = B * d
        r1_bytes = 3.0 * B * d * ACT_BPE
        ops.append(_op(layer, "residual_attn", "elem", "Other", r1_flops, r1_bytes, hw, DEFAULT_EFF["elem"]))

        # FFN norm
        ffn_norm_w = _tensor_bytes(tensor_bytes, layer, "ffn_norm.weight", fallback=d * 4.0)
        ffn_norm_flops = 8.0 * B * d
        ffn_norm_bytes = ffn_norm_w + (2.0 * B * d * ACT_BPE)
        ops.append(_op(layer, "ffn_norm", "norm", "Norm", ffn_norm_flops, ffn_norm_bytes, hw, DEFAULT_EFF["norm"]))

        # FFN gate
        gate_w = _tensor_bytes(tensor_bytes, layer, "ffn_gate.weight", fallback=d * ffn)
        gate_shape = _tensor_shape(tensor_shapes, layer, "ffn_gate.weight")
        gate_flops = _gemm_flops_from_shape(gate_shape, d, ffn, B)
        gate_bytes = gate_w + (B * d + B * ffn) * ACT_BPE
        ops.append(_op(layer, "ffn_gate", "gemm", "FFN", gate_flops, gate_bytes, hw, DEFAULT_EFF["gemm"]))

        # FFN up
        up_w = _tensor_bytes(tensor_bytes, layer, "ffn_up.weight", fallback=d * ffn)
        up_shape = _tensor_shape(tensor_shapes, layer, "ffn_up.weight")
        up_flops = _gemm_flops_from_shape(up_shape, d, ffn, B)
        up_bytes = up_w + (B * d + B * ffn) * ACT_BPE
        ops.append(_op(layer, "ffn_up", "gemm", "FFN", up_flops, up_bytes, hw, DEFAULT_EFF["gemm"]))

        # SwiGLU activation and multiply
        act_flops = 6.0 * B * ffn
        act_bytes = 3.0 * B * ffn * ACT_BPE
        ops.append(_op(layer, "ffn_swiglu", "elem", "FFN", act_flops, act_bytes, hw, DEFAULT_EFF["elem"]))

        # FFN down
        down_w = _tensor_bytes(tensor_bytes, layer, "ffn_down.weight", fallback=ffn * d)
        down_shape = _tensor_shape(tensor_shapes, layer, "ffn_down.weight")
        down_flops = _gemm_flops_from_shape(down_shape, ffn, d, B)
        down_bytes = down_w + (B * ffn + B * d) * ACT_BPE
        ops.append(_op(layer, "ffn_down", "gemm", "FFN", down_flops, down_bytes, hw, DEFAULT_EFF["gemm"]))

        # Residual add after FFN
        r2_flops = B * d
        r2_bytes = 3.0 * B * d * ACT_BPE
        ops.append(_op(layer, "residual_ffn", "elem", "Other", r2_flops, r2_bytes, hw, DEFAULT_EFF["elem"]))

        if other_per_layer > 0:
            # Convert "fixed per-step other ms" into a pseudo op for transparent accounting.
            pseudo_bw_bytes = other_per_layer * 1e-3 * hw.mem_gbps * 1e9 * DEFAULT_EFF["other"].mem_eff
            ops.append(_op(layer, "runtime_other", "other", "Other", 0.0, pseudo_bw_bytes, hw, DEFAULT_EFF["other"]))

    return ops


def aggregate_category_ms(ops: list[OpRecord]) -> dict[str, float]:
    out: dict[str, float] = {"QKV+O": 0.0, "RoPE": 0.0, "Attn": 0.0, "FFN": 0.0, "Norm": 0.0, "Other": 0.0}
    for op in ops:
        out[op.category] = out.get(op.category, 0.0) + op.est_ms
    return out


def format_sweep_tables(
    model_label: str,
    ntg: int,
    rows: list[dict[str, float | int]],
) -> str:
    sep_w = 110
    out: list[str] = []
    out.append("=" * sep_w)
    out.append(f"  DECODE DURATION (ms)  (model: {model_label}, ntg={ntg})")
    out.append("=" * sep_w)
    out.append("   npp  npl    QKV+O     RoPE     Attn      FFN     Norm    Other     Wall    tok/s")
    out.append("-" * sep_w)

    last_ctx = None
    for r in rows:
        ctx = int(r["npp"])
        if last_ctx is not None and ctx != last_ctx:
            out.append("-" * sep_w)
        last_ctx = ctx
        out.append(
            f"{int(r['npp']):6d}{int(r['npl']):5d}"
            f"{float(r['QKV+O']):9.2f}{float(r['RoPE']):9.2f}{float(r['Attn']):9.2f}"
            f"{float(r['FFN']):9.2f}{float(r['Norm']):9.2f}{float(r['Other']):9.2f}"
            f"{float(r['Wall']):9.2f}{float(r['tok/s']):9.1f}"
        )
    out.append("=" * sep_w)
    out.append("")
    out.append("")
    out.append("=" * sep_w)
    out.append(f"  DECODE PERCENTAGE (%)  (model: {model_label}, ntg={ntg})")
    out.append("=" * sep_w)
    out.append("   npp  npl    QKV+O     RoPE     Attn      FFN     Norm    Other     Wall    tok/s")
    out.append("-" * sep_w)

    last_ctx = None
    for r in rows:
        ctx = int(r["npp"])
        if last_ctx is not None and ctx != last_ctx:
            out.append("-" * sep_w)
        last_ctx = ctx
        wall = float(r["Wall"])
        if wall <= 0:
            p = {k: 0.0 for k in ("QKV+O", "RoPE", "Attn", "FFN", "Norm", "Other")}
        else:
            p = {k: 100.0 * float(r[k]) / wall for k in ("QKV+O", "RoPE", "Attn", "FFN", "Norm", "Other")}
        out.append(
            f"{int(r['npp']):6d}{int(r['npl']):5d}"
            f"{p['QKV+O']:9.1f}{p['RoPE']:9.1f}{p['Attn']:9.1f}"
            f"{p['FFN']:9.1f}{p['Norm']:9.1f}{p['Other']:9.1f}"
            f"{wall:9.2f}{float(r['tok/s']):9.1f}"
        )
    out.append("=" * sep_w)
    out.append("")
    return "\n".join(out)


def format_op_report(
    spec: ModelSpec,
    hw: HardwareSpec,
    ctx_len: int,
    batch_size: int,
    kv_bits: float,
    ops: list[OpRecord],
) -> str:
    lines: list[str] = []
    lines.append("# OP DETAIL REPORT")
    lines.append(f"model_name={spec.name}")
    lines.append(f"arch={spec.arch}")
    lines.append(
        "dims="
        f"layers:{spec.n_layers}, d_model:{spec.d_model}, ffn_dim:{spec.ffn_dim}, "
        f"heads:{spec.n_heads}, kv_heads:{spec.n_kv_heads}, head_dim:{spec.head_dim}, value_dim:{spec.value_dim}"
    )
    lines.append(f"hw={hw.name}, peak_compute_tflops={hw.compute_tflops}, peak_mem_gbps={hw.mem_gbps}")
    lines.append(f"scenario=ctx_len:{ctx_len}, batch_size:{batch_size}, kv_bits:{kv_bits}")
    lines.append("")
    lines.append(
        "Columns: layer op family category flops bytes ai est_ms min_ms comp_est_ms mem_est_ms comp_min_ms mem_min_ms"
    )
    lines.append(
        "---------------------------------------------------------------------------------------------------------------"
    )
    for r in ops:
        lines.append(
            f"{r.layer:4d}  {r.op:14s} {r.family:6s} {r.category:6s} "
            f"{r.flops:12.0f} {r.bytes_total:12.0f} {r.arithmetic_intensity:8.3f} "
            f"{r.est_ms:8.4f} {r.min_ms:8.4f} {r.compute_ms_est:10.4f} {r.memory_ms_est:10.4f} "
            f"{r.compute_ms_min:10.4f} {r.memory_ms_min:10.4f}"
        )

    lines.append("")
    lines.append("# AGGREGATED BY OP")
    by_op: dict[str, dict[str, float]] = {}
    for r in ops:
        it = by_op.setdefault(r.op, {"flops": 0.0, "bytes": 0.0, "est_ms": 0.0, "min_ms": 0.0})
        it["flops"] += r.flops
        it["bytes"] += r.bytes_total
        it["est_ms"] += r.est_ms
        it["min_ms"] += r.min_ms
    lines.append("op             flops_total      bytes_total    ai_total   est_ms_total   min_ms_total")
    lines.append("--------------------------------------------------------------------------------------")
    for op_name in sorted(by_op.keys()):
        it = by_op[op_name]
        ai = it["flops"] / it["bytes"] if it["bytes"] > 0 else float("inf")
        lines.append(
            f"{op_name:14s} {it['flops']:14.0f} {it['bytes']:14.0f} {ai:10.3f} {it['est_ms']:13.4f} {it['min_ms']:13.4f}"
        )

    lines.append("")
    lines.append("# AGGREGATED BY CATEGORY")
    by_cat: dict[str, float] = {}
    by_cat_min: dict[str, float] = {}
    for r in ops:
        by_cat[r.category] = by_cat.get(r.category, 0.0) + r.est_ms
        by_cat_min[r.category] = by_cat_min.get(r.category, 0.0) + r.min_ms
    for k in ("QKV+O", "RoPE", "Attn", "FFN", "Norm", "Other"):
        lines.append(f"{k:6s} est_ms={by_cat.get(k, 0.0):10.4f}  min_ms={by_cat_min.get(k, 0.0):10.4f}")

    total_est = sum(r.est_ms for r in ops)
    total_min = sum(r.min_ms for r in ops)
    lines.append("")
    lines.append(f"TOTAL_STEP_MS est={total_est:.6f}  min={total_min:.6f}")
    lines.append(f"MS_PER_TOKEN est={total_est/float(batch_size):.6f}  min={total_min/float(batch_size):.6f}")
    lines.append(f"TOK_PER_S est={1000.0*float(batch_size)/total_est:.4f}  min={1000.0*float(batch_size)/total_min:.4f}")
    return "\n".join(lines) + "\n"


def parse_int_list(csv: str) -> list[int]:
    vals = []
    for x in csv.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to local GGUF model")
    ap.add_argument("--hw", default="h100-sxm", choices=sorted(HARDWARE_PRESETS.keys()))
    ap.add_argument("--ctx-list", default="1024,2048,4096,8192,16384")
    ap.add_argument("--batch-list", default="1,2,4,8,16,32")
    ap.add_argument("--ntg", type=int, default=128, help="Printed in table header only")
    ap.add_argument("--kv-bits", type=float, default=16.0, help="KV cache precision in bits/value")
    ap.add_argument("--other-ms", type=float, default=0.0, help="Fixed per-step runtime overhead in ms")
    ap.add_argument("--out", default="", help="Write sweep text table")
    ap.add_argument("--op-report", default="", help="Write per-op report for one sweep point")
    ap.add_argument("--op-report-ctx", type=int, default=4096)
    ap.add_argument("--op-report-batch", type=int, default=8)
    args = ap.parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    hw = HARDWARE_PRESETS[str(args.hw)]
    ctx_list = parse_int_list(args.ctx_list)
    batch_list = parse_int_list(args.batch_list)
    kv_bpe = float(args.kv_bits) / 8.0

    spec, tensor_bytes, tensor_shapes, kv_bpe_default = load_model_spec_from_gguf(model_path)
    if args.kv_bits <= 0:
        kv_bpe = kv_bpe_default

    rows: list[dict[str, float | int]] = []
    for ctx_len in ctx_list:
        for batch_size in batch_list:
            ops = simulate_decode_step(
                spec=spec,
                tensor_bytes=tensor_bytes,
                tensor_shapes=tensor_shapes,
                hw=hw,
                batch_size=batch_size,
                ctx_len=ctx_len,
                kv_bpe=kv_bpe,
                per_step_other_ms=float(args.other_ms),
            )
            cat = aggregate_category_ms(ops)
            wall = sum(cat.values())
            toks = (1000.0 * float(batch_size) / wall) if wall > 0 else 0.0
            rows.append(
                {
                    "npp": int(ctx_len),
                    "npl": int(batch_size),
                    "QKV+O": cat.get("QKV+O", 0.0),
                    "RoPE": cat.get("RoPE", 0.0),
                    "Attn": cat.get("Attn", 0.0),
                    "FFN": cat.get("FFN", 0.0),
                    "Norm": cat.get("Norm", 0.0),
                    "Other": cat.get("Other", 0.0),
                    "Wall": wall,
                    "tok/s": toks,
                }
            )

    text = format_sweep_tables(str(args.model), int(args.ntg), rows)
    if args.out:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)

    if args.op_report:
        ops_detail = simulate_decode_step(
            spec=spec,
            tensor_bytes=tensor_bytes,
            tensor_shapes=tensor_shapes,
            hw=hw,
            batch_size=int(args.op_report_batch),
            ctx_len=int(args.op_report_ctx),
            kv_bpe=kv_bpe,
            per_step_other_ms=float(args.other_ms),
        )
        detail_text = format_op_report(
            spec=spec,
            hw=hw,
            ctx_len=int(args.op_report_ctx),
            batch_size=int(args.op_report_batch),
            kv_bits=float(args.kv_bits),
            ops=ops_detail,
        )
        op_path = os.path.abspath(os.path.expanduser(args.op_report))
        with open(op_path, "w", encoding="utf-8") as f:
            f.write(detail_text)


if __name__ == "__main__":
    main()
