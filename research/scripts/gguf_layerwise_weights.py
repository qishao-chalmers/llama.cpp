"""Load GGUF tensor sizes and map llama.cpp ``blk.L.*`` names to per-op weight bytes.

Used by layerwise_roofline_sim.py when ``--gguf`` is set so mixed per-layer / per-tensor
quantization is reflected in roofline byte counts (instead of a single effective bpw).
"""

from __future__ import annotations

import os
import sys
from typing import Any


def _gguf_py_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    return os.path.join(repo_root, "gguf-py")


def load_gguf_tensor_n_bytes(path: str) -> dict[str, int]:
    """Return tensor name -> stored size in bytes for all tensors in a GGUF file."""

    gguf_py = _gguf_py_dir()
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    from gguf.gguf_reader import GGUFReader  # noqa: PLC0415

    path = os.path.expanduser(path)
    reader = GGUFReader(path)
    return {t.name: int(t.n_bytes) for t in reader.tensors}


def resolve_layer_weight_bytes_from_gguf(
    layer: int,
    m: dict[str, Any],
    tb: dict[str, int],
    bpe: float,
    norm_bpe: float,
) -> dict[str, float]:
    """Map one decoder block's tensors to byte counts used by the layerwise sim.

    Tensor names follow ``gguf-py/gguf/constants.py`` ``TENSOR_NAMES`` (e.g.
    ``blk.{L}.attn_q.weight``, ``blk.{L}.attn_output.weight``, ``blk.{L}.ffn_gate.weight``).

    If ``blk.{L}.attn_qkv.weight`` is present and separate Q/K/V weights are not all
    present, fused bytes are split by parameter count (same as uniform bpw per element).

    Missing tensors fall back to ``n_params * bpe`` (or ``n_norm * norm_bpe`` for norms).
    """

    d = int(m["d_model"])
    nh = int(m["n_heads"])
    nkv = int(m["n_kv_heads"])
    hd = int(m["head_dim"])
    ffn = int(m["ffn_dim"])
    style = str(m.get("ffn_style", "swiglu"))

    def syn_params(n_floats: float) -> float:
        return float(n_floats) * float(bpe)

    def syn_norm(n_floats: float) -> float:
        return float(n_floats) * float(norm_bpe)

    p = f"blk.{layer}"
    an = f"{p}.attn_norm.weight"
    fn = f"{p}.ffn_norm.weight"

    attn_norm_b = float(tb[an]) if an in tb else syn_norm(float(d))
    ffn_norm_b = float(tb[fn]) if fn in tb else syn_norm(float(d))

    tq = f"{p}.attn_q.weight"
    tk = f"{p}.attn_k.weight"
    tv = f"{p}.attn_v.weight"
    tqkv = f"{p}.attn_qkv.weight"
    nq = float(d * nh * hd)
    nk = float(d * nkv * hd)
    nv = float(d * nkv * hd)

    if tq in tb and tk in tb and tv in tb:
        qb, kb, vb = float(tb[tq]), float(tb[tk]), float(tb[tv])
    elif tqkv in tb:
        tot = float(tb[tqkv])
        nt = nq + nk + nv
        qb = tot * (nq / nt)
        kb = tot * (nk / nt)
        vb = tot * (nv / nt)
    else:
        qb = float(tb[tq]) if tq in tb else syn_params(nq)
        kb = float(tb[tk]) if tk in tb else syn_params(nk)
        vb = float(tb[tv]) if tv in tb else syn_params(nv)

    to = f"{p}.attn_output.weight"
    no = float(nh * hd * d)
    ob = float(tb[to]) if to in tb else syn_params(no)

    out: dict[str, float] = {
        "attn_norm": attn_norm_b,
        "ffn_norm": ffn_norm_b,
        "q": qb,
        "k": kb,
        "v": vb,
        "o": ob,
    }
    if style == "swiglu":
        tg = f"{p}.ffn_gate.weight"
        tu = f"{p}.ffn_up.weight"
        td = f"{p}.ffn_down.weight"
        out["gate"] = float(tb[tg]) if tg in tb else syn_params(float(d * ffn))
        out["up"] = float(tb[tu]) if tu in tb else syn_params(float(d * ffn))
        out["down"] = float(tb[td]) if td in tb else syn_params(float(ffn * d))
    else:
        tu = f"{p}.ffn_up.weight"
        td = f"{p}.ffn_down.weight"
        out["up"] = float(tb[tu]) if tu in tb else syn_params(float(d * ffn))
        out["down"] = float(tb[td]) if td in tb else syn_params(float(ffn * d))
    return out
