"""
quant.py — simulated quantization functions for KV cache experiments.

All functions take float16 numpy arrays and return float16 arrays with
values rounded to the target precision.  No storage format change —
this is *simulated* quantization to measure accuracy impact.

Supported formats:
  fp16     — no-op baseline
  bf16     — round to bfloat16 range/precision
  fp8_e4m3 — IEEE 754 8-bit float, e4m3 format  (±448, high precision)
  fp8_e5m2 — IEEE 754 8-bit float, e5m2 format  (±57344, wider range)
  int8     — uniform [-127, 127], per-tensor (one scale for entire matrix)
  int8_ch  — uniform per-channel: one scale per dimension, shared across tokens  (axis=0)
  int8_tok — uniform per-token:   one scale per token,     shared across dims    (axis=1)
  int4     — uniform [-7, 7] 4-bit, per-tensor
  int4_ch  — uniform per-channel (axis=0)
  int4_tok — uniform per-token   (axis=1)
  int3     — uniform [-3, 3] 3-bit, per-tensor
  int3_ch  — uniform per-channel (axis=0)
  int3_tok — uniform per-token   (axis=1)
  int2     — uniform [-1, 1] 2-bit, per-tensor
  int2_ch  — uniform per-channel (axis=0)
  int2_tok — uniform per-token   (axis=1)
  nf4      — NormalFloat-4 lookup table

Non-uniform 4-level variant (sits between int3 and int2 in quality):
  int3_half     — 4 levels, default indices {0,3,5,7} → {0, 3/7, 5/7, 1}, per-tensor
  int3_half_ch  — per-channel (axis=0)
  int3_half_tok — per-token   (axis=1)
  Custom indices: use make_int3_half([0,2,4,7], axis=0) for any combo of 4 from 0–7.
  Step size = (max-min)/7 (int3's step); default has wide gap at bottom.

Arbitrary bin-count variants (fill gap between any two power-of-2 levels):
  q5 / q5_ch / q5_tok  — exactly 5 bins (between int2's 4 and int3's 8)
  q6 / q6_ch / q6_tok  — exactly 6 bins
  q7 / q7_ch / q7_tok  — exactly 7 bins
  Any q{N} / q{N}_ch / q{N}_tok for any N >= 2 works dynamically.
  Levels placed at linspace(-1, 1, N) * per-channel/token scale.
  Use --quants q4_ch q5_ch q6_ch q7_ch q8_ch to sweep bin counts.

Group-within-token variants (dim_group_size=G splits head_dim into groups):
  int4_tok_g16 / int4_tok_g32 / int4_tok_g64
  int3_tok_g16 / int3_tok_g32 / int3_tok_g64
  int8_tok_g16 / int8_tok_g32 / int8_tok_g64
  Each group of G adjacent dims gets its own scale — finer than per-token,
  coarser than per-channel. E.g. head_dim=128, g32 → 4 scales per token.

Group-within-channel variants (token_group_size=G groups G tokens together):
  int4_ch_g32 / int4_ch_g64 / int4_ch_g128
  int3_ch_g32 / int3_ch_g64 / int3_ch_g128
  int8_ch_g32 / int8_ch_g64 / int8_ch_g128
  Each group of G consecutive tokens shares one scale per dim.
  The hook fires every G tokens (not via --quant-group-size flag).
  E.g. g32 → 32 tokens per scale per dim; g128 is equivalent to plain int4_ch
  with --quant-group-size 128.

K/V split: pass "int8_ch:int4_tok" to use int8_ch for K and int4_tok for V.

Usage:
  from quant import get_quant_fn
  fn = get_quant_fn("fp8_e4m3")
  quantized = fn(arr)   # arr: np.ndarray, float16, any shape
"""

import numpy as np

try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False
    print("WARNING: ml_dtypes not available, FP8 will use manual implementation")

# ── NF4 lookup table ──────────────────────────────────────────────────────────
# 16 values symmetric around 0, calibrated for standard normal distribution
NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,  0.33791524171829224,
     0.44070982933044434,  0.5626170039176941,   0.7229568362236023,   1.0,
], dtype=np.float32)


# ── Quantization functions ─────────────────────────────────────────────────────

def quant_fp16(arr: np.ndarray) -> np.ndarray:
    """No-op: FP16 is the baseline."""
    return arr.astype(np.float16)


def quant_bf16(arr: np.ndarray) -> np.ndarray:
    """Round to bfloat16 precision (truncate lower 8 mantissa bits of float32)."""
    f32 = arr.astype(np.float32)
    # BF16: keep top 16 bits of float32
    i32 = f32.view(np.uint32)
    i32_rounded = (i32 + 0x8000) & 0xFFFF0000   # round to nearest
    return i32_rounded.view(np.float32).astype(np.float16)


def quant_fp8_e4m3(arr: np.ndarray) -> np.ndarray:
    """Round to FP8 E4M3 precision (stored as float16)."""
    if HAS_ML_DTYPES:
        f32 = arr.astype(np.float32)
        fp8 = f32.astype(ml_dtypes.float8_e4m3fn)
        return fp8.astype(np.float16)
    else:
        return _manual_fp8_e4m3(arr)


def quant_fp8_e5m2(arr: np.ndarray) -> np.ndarray:
    """Round to FP8 E5M2 precision (stored as float16)."""
    if HAS_ML_DTYPES:
        f32 = arr.astype(np.float32)
        fp8 = f32.astype(ml_dtypes.float8_e5m2)
        return fp8.astype(np.float16)
    else:
        return _manual_fp8_e5m2(arr)


def _uniform_quant(arr: np.ndarray, bits: int, axis=None, dim_group_size=None) -> np.ndarray:
    """
    Uniform symmetric quantization.
    axis=None : per-tensor  — one scale for the whole matrix
    axis=0    : per-channel — one scale per column (dimension), shared across tokens
    axis=1    : per-token   — one scale per row (token), shared across dimensions

    dim_group_size=G (requires axis != None, arr.ndim == 2):
      axis=1: split head_dim into groups of G → [n_tokens, n_groups, G], one scale per group
              e.g. head_dim=128, G=32 → 4 independent scales per token
      axis=0: split n_tokens into groups of G → [n_groups, G, n_embd], one scale per group
    """
    if arr.size == 0:
        return arr.astype(np.float16)
    n_levels = (1 << (bits - 1)) - 1   # int8→127, int4→7, int3→3, int4→7, int2→1
    f32 = arr.astype(np.float32)

    if dim_group_size is not None and axis is not None and f32.ndim == 2:
        n_outer, n_inner = f32.shape
        if axis == 1:
            # Group adjacent dims within each token row
            assert n_inner % dim_group_size == 0, \
                f"head_dim {n_inner} not divisible by dim_group_size {dim_group_size}"
            n_groups = n_inner // dim_group_size
            f32_g = f32.reshape(n_outer, n_groups, dim_group_size)
            amax = np.abs(f32_g).max(axis=2, keepdims=True)   # [n_tokens, n_groups, 1]
        else:  # axis == 0
            # Group adjacent tokens within each channel column
            assert n_outer % dim_group_size == 0, \
                f"n_tokens {n_outer} not divisible by dim_group_size {dim_group_size}"
            n_groups = n_outer // dim_group_size
            f32_g = f32.reshape(n_groups, dim_group_size, n_inner)
            amax = np.abs(f32_g).max(axis=1, keepdims=True)   # [n_groups, 1, n_embd]
        amax = np.where(amax == 0, 1.0, amax)
        scale = amax / n_levels
        q = np.round(f32_g / scale).clip(-n_levels, n_levels)
        return (q * scale).reshape(arr.shape).astype(np.float16)

    if axis is None:
        amax = np.abs(f32).max()
    else:
        amax = np.abs(f32).max(axis=axis, keepdims=True)

    amax = np.where(amax == 0, 1.0, amax)
    scale = amax / n_levels
    q = np.round(f32 / scale).clip(-n_levels, n_levels)
    return (q * scale).astype(np.float16)


def _uniform_quant_asym(arr: np.ndarray, bits: int = None, axis=None,
                        dim_group_size=None, n_levels: int = None) -> np.ndarray:
    """
    Asymmetric (affine) uniform quantization.  Stores scale + min-offset per group.

    Exactly one of bits or n_levels must be given:
      bits     → n_levels = 2^bits - 1  (int2→3, int3→7, int4→15, int8→255)
      n_levels → used directly          (q5_ch asym: n_levels=4 → 5 bins)

    scale = (xmax - xmin) / n_levels
    q     = round((x - xmin) / scale)  ∈ {0, 1, ..., n_levels}
    x̂     = q * scale + xmin

    axis=None : per-tensor
    axis=0    : per-channel (one scale+offset per column / head_dim element)
    axis=1    : per-token   (one scale+offset per row / token)
    dim_group_size: same sub-group logic as _uniform_quant.
    """
    if arr.size == 0:
        return arr.astype(np.float16)
    if n_levels is None:
        n_levels = (1 << bits) - 1  # int2→3, int3→7, int4→15, int8→255
    f32 = arr.astype(np.float32)

    if dim_group_size is not None and axis is not None and f32.ndim == 2:
        n_outer, n_inner = f32.shape
        if axis == 1:
            assert n_inner % dim_group_size == 0
            n_groups = n_inner // dim_group_size
            f32_g = f32.reshape(n_outer, n_groups, dim_group_size)
            xmin = f32_g.min(axis=2, keepdims=True)
            xmax = f32_g.max(axis=2, keepdims=True)
        else:  # axis == 0
            assert n_outer % dim_group_size == 0
            n_groups = n_outer // dim_group_size
            f32_g = f32.reshape(n_groups, dim_group_size, n_inner)
            xmin = f32_g.min(axis=1, keepdims=True)
            xmax = f32_g.max(axis=1, keepdims=True)
        scale = np.where(xmax == xmin, 1.0, (xmax - xmin) / n_levels)
        q = np.round((f32_g - xmin) / scale).clip(0, n_levels)
        return (q * scale + xmin).reshape(arr.shape).astype(np.float16)

    if axis is None:
        xmin = f32.min()
        xmax = f32.max()
    else:
        xmin = f32.min(axis=axis, keepdims=True)
        xmax = f32.max(axis=axis, keepdims=True)

    scale = np.where(xmax == xmin, 1.0, (xmax - xmin) / n_levels)
    q = np.round((f32 - xmin) / scale).clip(0, n_levels)
    return (q * scale + xmin).astype(np.float16)


def _nbin_quant(arr: np.ndarray, n_bins: int, axis=None) -> np.ndarray:
    """Symmetric quantization with exactly n_bins evenly-spaced levels.

    Levels are placed at linspace(-1, 1, n_bins) * scale, where scale is
    the per-tensor / per-channel / per-token abs-max.

    Examples:
      n_bins=4 : levels at {-1, -0.333, 0.333, 1}   (≈ int2 quality)
      n_bins=5 : levels at {-1, -0.5, 0, 0.5, 1}
      n_bins=6 : levels at {-1, -0.6, -0.2, 0.2, 0.6, 1}
      n_bins=7 : levels at {-1, -0.667, -0.333, 0, 0.333, 0.667, 1}
      n_bins=8 : levels at {-1, -0.714, ..., 1}      (≈ int3 quality)
    """
    if arr.size == 0:
        return arr.astype(np.float16)
    levels = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
    f32 = arr.astype(np.float32)

    if axis is None:
        amax = np.abs(f32).max()
        if amax == 0:
            return arr.astype(np.float16)
        norm = f32 / amax
    elif axis == 0:
        amax = np.abs(f32).max(axis=0, keepdims=True)
        amax = np.where(amax == 0, 1.0, amax)
        norm = f32 / amax
    else:  # axis == 1
        amax = np.abs(f32).max(axis=1, keepdims=True)
        amax = np.where(amax == 0, 1.0, amax)
        norm = f32 / amax

    norm = norm.clip(-1.0, 1.0)
    # Find nearest level for each element: shape [..., n_bins]
    idx = np.abs(norm[..., np.newaxis] - levels).argmin(axis=-1)
    q_norm = levels[idx]
    return (q_norm * amax).astype(np.float16)


def quant_int8(arr: np.ndarray) -> np.ndarray:
    """INT8 per-tensor symmetric quantization."""
    return _uniform_quant(arr, bits=8, axis=None)


def quant_int8_ch(arr: np.ndarray) -> np.ndarray:
    """INT8 per-channel (column) symmetric quantization."""
    return _uniform_quant(arr, bits=8, axis=0)


def quant_int4(arr: np.ndarray) -> np.ndarray:
    """INT4 per-tensor symmetric quantization."""
    return _uniform_quant(arr, bits=4, axis=None)


def quant_int4_ch(arr: np.ndarray) -> np.ndarray:
    """INT4 per-channel (column) symmetric quantization."""
    return _uniform_quant(arr, bits=4, axis=0)


def quant_nf4(arr: np.ndarray) -> np.ndarray:
    """
    NormalFloat-4 quantization. Uses a fixed lookup table of 16 values.
    Scale by abs-max before lookup, then dequantize.
    """
    f32 = arr.astype(np.float32)
    amax = np.abs(f32).max()
    if amax == 0:
        return arr.astype(np.float16)
    f32_norm = f32 / amax                    # normalize to [-1, 1]

    # Find nearest NF4 value for each element
    diff = np.abs(f32_norm[..., np.newaxis] - NF4_TABLE)   # (..., 16)
    idx  = diff.argmin(axis=-1)
    quantized_norm = NF4_TABLE[idx]
    return (quantized_norm * amax).astype(np.float16)


def quant_int2(arr: np.ndarray) -> np.ndarray:
    """INT2 per-tensor symmetric quantization (values ∈ {-1, 0, 1} effectively)."""
    return _uniform_quant(arr, bits=2, axis=None)


def quant_int2_ch(arr: np.ndarray) -> np.ndarray:
    """INT2 per-channel (column) symmetric quantization."""
    return _uniform_quant(arr, bits=2, axis=0)


def quant_int8_tok(arr: np.ndarray) -> np.ndarray:
    """INT8 per-token (row) symmetric quantization. One scale per token."""
    return _uniform_quant(arr, bits=8, axis=1)


def quant_int4_tok(arr: np.ndarray) -> np.ndarray:
    """INT4 per-token (row) symmetric quantization. One scale per token."""
    return _uniform_quant(arr, bits=4, axis=1)


def quant_int2_tok(arr: np.ndarray) -> np.ndarray:
    """INT2 per-token (row) symmetric quantization. One scale per token."""
    return _uniform_quant(arr, bits=2, axis=1)


def quant_int3(arr: np.ndarray) -> np.ndarray:
    """INT3 per-tensor symmetric quantization."""
    return _uniform_quant(arr, bits=3, axis=None)


def quant_int3_ch(arr: np.ndarray) -> np.ndarray:
    """INT3 per-channel (column) symmetric quantization."""
    return _uniform_quant(arr, bits=3, axis=0)


def quant_int3_tok(arr: np.ndarray) -> np.ndarray:
    """INT3 per-token (row) symmetric quantization. One scale per token."""
    return _uniform_quant(arr, bits=3, axis=1)


# ── int3_half: 4-level non-uniform quantizer using int3's step size ───────────
#
# int3 asymmetric divides [min, max] into 7 equal steps (8 bins: indices 0-7).
# int3_half selects 4 of those 8 positions; default is {0,3,5,7}:
#
#   {0,3,5,7} → {0, 3/7, 5/7, 1}  gaps: 3/7, 2/7, 2/7  (coarse bottom)
#   {0,2,4,7} → {0, 2/7, 4/7, 1}  gaps: 2/7, 2/7, 3/7  (coarse top)
#   {0,2,5,7} → {0, 2/7, 5/7, 1}  gaps: 2/7, 3/7, 2/7  (coarse middle)
#
# For custom indices use make_int3_half(indices, axis):
#   fn = make_int3_half([0, 2, 4, 7], axis=0)
#   out = fn(arr)

def _int3_half_quant(arr: np.ndarray, axis=None, indices=(0, 3, 5, 7)) -> np.ndarray:
    """4-level non-uniform quantizer using int3's step size (step = (max-min)/7).

    indices: any 4 positions from 0–7 selecting which int3 bins to keep.
             Default {0,3,5,7}: wide gap at bottom, finer at top.

    axis=None : per-tensor
    axis=0    : per-channel (one scale per column / head_dim element)
    axis=1    : per-token   (one scale per row / token)
    """
    if arr.size == 0:
        return arr.astype(np.float16)
    f32    = arr.astype(np.float32)
    LEVELS = np.array([i / 7.0 for i in indices], dtype=np.float32)

    if axis is None:
        xmin = f32.min()
        xmax = f32.max()
        rng  = xmax - xmin
        if rng == 0:
            return arr.copy()
        cb   = xmin + LEVELS * rng                     # [4]
        idx  = np.abs(f32[..., None] - cb).argmin(-1)  # [...]
        result = cb[idx]

    elif axis == 0:
        # Per-channel: one codebook per column
        xmin = f32.min(axis=0)                         # [n_embd]
        xmax = f32.max(axis=0)
        rng  = np.where(xmax - xmin == 0, 1.0, xmax - xmin)
        cb   = xmin[:, None] + LEVELS[None, :] * rng[:, None]  # [n_embd, 4]
        diff = np.abs(f32[:, :, None] - cb[None, :, :])              # [n_tok, n_embd, 4]
        idx  = diff.argmin(axis=-1)                                   # [n_tok, n_embd]
        result = cb[np.arange(f32.shape[1])[None, :], idx]           # [n_tok, n_embd]

    elif axis == 1:
        # Per-token: one codebook per row
        xmin = f32.min(axis=1)                         # [n_tok]
        xmax = f32.max(axis=1)
        rng  = np.where(xmax - xmin == 0, 1.0, xmax - xmin)
        cb   = xmin[:, None] + LEVELS[None, :] * rng[:, None]  # [n_tok, 4]
        diff = np.abs(f32[:, :, None] - cb[:, None, :])         # [n_tok, n_embd, 4]
        idx  = diff.argmin(axis=-1)                              # [n_tok, n_embd]
        result = cb[np.arange(f32.shape[0])[:, None], idx]      # [n_tok, n_embd]

    else:
        raise ValueError(f"axis must be None, 0, or 1; got {axis}")

    return result.astype(np.float16)


def quant_int3_half(arr: np.ndarray) -> np.ndarray:
    """INT3-half per-tensor, default indices {0,3,5,7}."""
    return _int3_half_quant(arr, axis=None)


def quant_int3_half_ch(arr: np.ndarray) -> np.ndarray:
    """INT3-half per-channel, default indices {0,3,5,7}."""
    return _int3_half_quant(arr, axis=0)


def quant_int3_half_tok(arr: np.ndarray) -> np.ndarray:
    """INT3-half per-token, default indices {0,3,5,7}."""
    return _int3_half_quant(arr, axis=1)


def make_int3_half(indices, axis=0):
    """Factory for custom int3_half with any 4 indices from 0–7.

    Usage:
        fn = make_int3_half([0, 2, 4, 7], axis=0)   # per-channel
        fn = make_int3_half([0, 2, 5, 7], axis=1)   # per-token
        out = fn(arr)
    """
    indices = tuple(indices)
    assert len(indices) == 4 and all(0 <= i <= 7 for i in indices), \
        f"indices must be 4 values in 0–7, got {indices}"
    def fn(arr, _idx=indices, _ax=axis):
        return _int3_half_quant(arr, axis=_ax, indices=_idx)
    fn.__name__ = f"int3_half_{''.join(str(i) for i in indices)}_{'ch' if axis==0 else 'tok' if axis==1 else 'tensor'}"
    return fn


# ── Dynamic grouped variants: int{N}_{ch,tok}_g{G} ───────────────────────────
# Any group size G is supported — no new functions needed for new G values.
#
#   int{N}_tok_g{G}: split head_dim into groups of G dims, one scale per group
#                    per token. E.g. head_dim=128, G=8 → 16 scales/token.
#   int{N}_ch_g{G}:  group G consecutive tokens together, one scale per
#                    (group, dim). Hook fires every G tokens.
#
# g1 in either family = per-element (one scale per value) → near-lossless.

import re as _re

_GROUPED_PAT = _re.compile(r'^int(\d+)_(ch|tok)_g(\d+)$')
_NBIN_PAT    = _re.compile(r'^q(\d+)(?:_(ch|tok))?$')
_VALID_BITS  = {2, 3, 4, 8}


def _parse_grouped_name(name):
    """Parse 'int4_ch_g32' → (bits=4, axis=0, gsize=32).  Returns None if no match."""
    m = _GROUPED_PAT.match(name)
    if not m:
        return None
    bits  = int(m.group(1))
    axis  = 0 if m.group(2) == "ch" else 1
    gsize = int(m.group(3))
    if bits not in _VALID_BITS or gsize < 1:
        return None
    return bits, axis, gsize


def _parse_nbin_name(name):
    """Parse 'q5' → (n_bins=5, axis=None), 'q6_ch' → (6, 0), 'q7_tok' → (7, 1).
    Returns None if no match or n_bins < 2."""
    m = _NBIN_PAT.match(name)
    if not m:
        return None
    n_bins = int(m.group(1))
    if n_bins < 2:
        return None
    granularity = m.group(2)   # 'ch', 'tok', or None
    axis = {"ch": 0, "tok": 1, None: None}[granularity]
    return n_bins, axis


def _make_grouped_fn(bits, axis, gsize):
    """Return a quantization function for the given (bits, axis, gsize)."""
    def fn(arr):
        return _uniform_quant(arr, bits=bits, axis=axis, dim_group_size=gsize)
    return fn


class _PerTokenSet:
    """Set-like: contains int{N}_tok, int{N}_tok_g{G}, and q{N}_tok for any valid N, G."""
    _explicit = {"int8_tok", "int4_tok", "int3_tok", "int2_tok"}

    def __contains__(self, name):
        if name in self._explicit:
            return True
        parsed = _parse_grouped_name(name)
        if parsed is not None and parsed[1] == 1:
            return True
        parsed2 = _parse_nbin_name(name)
        return parsed2 is not None and parsed2[1] == 1  # axis=1 → per-token


class _ChGroupSizeMap:
    """Dict-like: returns token group size G for int{N}_ch_g{G} names."""

    def __contains__(self, name):
        parsed = _parse_grouped_name(name)
        return parsed is not None and parsed[1] == 0  # axis=0 → per-channel

    def __getitem__(self, name):
        parsed = _parse_grouped_name(name)
        if parsed and parsed[1] == 0:
            return parsed[2]  # gsize
        raise KeyError(name)


# Per-token quants: hook fires every token (group_size=1).
# Per-channel _g{N} quants: hook fires every N tokens (from CH_QUANT_GROUP_SIZE).
PER_TOKEN_QUANTS   = _PerTokenSet()
CH_QUANT_GROUP_SIZE = _ChGroupSizeMap()


# ── Bin-index helpers (used by BinTracker) ────────────────────────────────────

def _sym_bin_indices(arr: np.ndarray, n_levels: int, axis=None,
                     dim_group_size=None) -> np.ndarray:
    """Integer bin index in {0,...,2*n_levels} for symmetric quantization."""
    f32 = arr.astype(np.float32)
    if dim_group_size is not None and axis is not None and f32.ndim == 2:
        n_outer, n_inner = f32.shape
        if axis == 1:
            f32_g = f32.reshape(n_outer, n_inner // dim_group_size, dim_group_size)
            amax  = np.abs(f32_g).max(axis=2, keepdims=True)
        else:
            f32_g = f32.reshape(n_outer // dim_group_size, dim_group_size, n_inner)
            amax  = np.abs(f32_g).max(axis=1, keepdims=True)
        amax  = np.where(amax == 0, 1.0, amax)
        scale = amax / n_levels
        q = np.round(f32_g / scale).clip(-n_levels, n_levels).astype(np.int32)
        return (q + n_levels).reshape(arr.shape)
    if axis is None:
        amax = float(np.abs(f32).max())
        amax = amax if amax != 0 else 1.0
    else:
        amax = np.abs(f32).max(axis=axis, keepdims=True)
        amax = np.where(amax == 0, 1.0, amax)
    scale = amax / n_levels
    q = np.round(f32 / scale).clip(-n_levels, n_levels).astype(np.int32)
    return q + n_levels


def _asym_bin_indices(arr: np.ndarray, n_levels: int, axis=None,
                      dim_group_size=None) -> np.ndarray:
    """Integer bin index in {0,...,n_levels} for asymmetric quantization."""
    f32 = arr.astype(np.float32)
    if dim_group_size is not None and axis is not None and f32.ndim == 2:
        n_outer, n_inner = f32.shape
        if axis == 1:
            f32_g = f32.reshape(n_outer, n_inner // dim_group_size, dim_group_size)
            xmin  = f32_g.min(axis=2, keepdims=True)
            xmax  = f32_g.max(axis=2, keepdims=True)
        else:
            f32_g = f32.reshape(n_outer // dim_group_size, dim_group_size, n_inner)
            xmin  = f32_g.min(axis=1, keepdims=True)
            xmax  = f32_g.max(axis=1, keepdims=True)
        scale = np.where(xmax == xmin, 1.0, (xmax - xmin) / n_levels)
        q = np.round((f32_g - xmin) / scale).clip(0, n_levels).astype(np.int32)
        return q.reshape(arr.shape)
    if axis is None:
        xmin, xmax = float(f32.min()), float(f32.max())
        scale = (xmax - xmin) / n_levels if xmax != xmin else 1.0
    else:
        xmin  = f32.min(axis=axis, keepdims=True)
        xmax  = f32.max(axis=axis, keepdims=True)
        scale = np.where(xmax == xmin, 1.0, (xmax - xmin) / n_levels)
    return np.round((f32 - xmin) / scale).clip(0, n_levels).astype(np.int32)


def _nbin_indices(arr: np.ndarray, n_bins: int, axis=None) -> np.ndarray:
    """Integer bin index in {0,...,n_bins-1} for q{N} symmetric quantization."""
    levels = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
    f32    = arr.astype(np.float32)
    if axis is None:
        amax = float(np.abs(f32).max())
        amax = amax if amax != 0 else 1.0
    else:
        amax = np.abs(f32).max(axis=axis, keepdims=True)
        amax = np.where(amax == 0, 1.0, amax)
    norm = (f32 / amax).clip(-1.0, 1.0)
    return np.abs(norm[..., np.newaxis] - levels).argmin(axis=-1).astype(np.int32)


def _make_bin_index_fn(name: str, asym: bool = False):
    """Return (fn, n_bins, labels) for BinTracker.

    fn(arr) -> int32 array of same shape containing the bin index for each element.
    Returns (None, 0, []) for float formats (fp16/bf16/fp8/nf4) — no integer bins.
    """
    if name in _FLOAT_QUANTS:
        return None, 0, []

    def _sym_info(bits, axis, gsize=None):
        nl = (1 << (bits - 1)) - 1
        nb = 2 * nl + 1
        lbs = [str(v) for v in range(-nl, nl + 1)]
        fn = lambda arr, _nl=nl, _ax=axis, _gs=gsize: \
            _sym_bin_indices(arr, _nl, _ax, dim_group_size=_gs)
        return fn, nb, lbs

    def _asym_info(n_levels, axis, gsize=None):
        nb  = n_levels + 1
        lbs = [str(v) for v in range(nb)]
        fn  = lambda arr, _nl=n_levels, _ax=axis, _gs=gsize: \
            _asym_bin_indices(arr, _nl, _ax, dim_group_size=_gs)
        return fn, nb, lbs

    if asym:
        parsed_g = _parse_grouped_name(name)
        if parsed_g:
            bits, axis, gsize = parsed_g
            return _asym_info((1 << bits) - 1, axis, gsize=gsize)
        m = _re.match(r'^int(\d+)(?:_(ch|tok))?$', name)
        if m and int(m.group(1)) in _VALID_BITS:
            bits = int(m.group(1))
            axis = {"ch": 0, "tok": 1, None: None}[m.group(2)]
            return _asym_info((1 << bits) - 1, axis)
        parsed2 = _parse_nbin_name(name)
        if parsed2:
            n_bins, axis = parsed2
            return _asym_info(n_bins - 1, axis)
    else:
        m = _re.match(r'^int(\d+)(?:_(ch|tok))?$', name)
        if m and int(m.group(1)) in _VALID_BITS:
            bits = int(m.group(1))
            axis = {"ch": 0, "tok": 1, None: None}[m.group(2)]
            return _sym_info(bits, axis)
        parsed_g = _parse_grouped_name(name)
        if parsed_g:
            bits, axis, gsize = parsed_g
            return _sym_info(bits, axis, gsize=gsize)
        parsed2 = _parse_nbin_name(name)
        if parsed2:
            n_bins, axis = parsed2
            lbs = [f"{v:.3f}" for v in np.linspace(-1, 1, n_bins)]
            fn  = lambda arr, _nb=n_bins, _ax=axis: _nbin_indices(arr, _nb, _ax)
            return fn, n_bins, lbs

    return None, 0, []


class BinTracker:
    """Wraps a quant function and counts how often each bin is hit.

    Drop-in callable replacement for get_quant_fn() output — returns the same
    dequantized array while accumulating integer bin hit counts in .counts.

    Usage:
        tracker = BinTracker("int3_ch")
        quantized = tracker(arr)          # identical to get_quant_fn("int3_ch")(arr)
        print(tracker.fractions())        # fraction of hits per bin
    """

    def __init__(self, name: str, asym: bool = False):
        self.name   = name
        self.asym   = asym
        self._qfn   = get_quant_fn(name, asym=asym)
        self._ifn, self.n_bins, self.labels = _make_bin_index_fn(name, asym)
        self.counts = np.zeros(max(self.n_bins, 1), dtype=np.int64)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        result = self._qfn(arr)
        if self._ifn is not None and self.n_bins > 0:
            self.counts += np.bincount(self._ifn(arr).ravel(),
                                       minlength=self.n_bins)
        return result

    def reset(self):
        self.counts[:] = 0

    def fractions(self) -> np.ndarray:
        total = self.counts.sum()
        return self.counts / total if total > 0 else np.zeros(self.n_bins)

    def to_dict(self) -> dict:
        return {"name": self.name, "asym": self.asym,
                "n_bins": self.n_bins, "labels": self.labels,
                "counts": self.counts.tolist()}

    @classmethod
    def from_dict(cls, d: dict):
        obj         = cls.__new__(cls)
        obj.name    = d["name"]
        obj.asym    = d["asym"]
        obj.n_bins  = d["n_bins"]
        obj.labels  = d["labels"]
        obj.counts  = np.array(d["counts"], dtype=np.int64)
        return obj


# ── Manual FP8 fallback (no ml_dtypes) ───────────────────────────────────────

def _fp8_e4m3_quantize_scalar(x: float) -> float:
    """Round float to nearest FP8-E4M3 representable value."""
    # E4M3: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
    # Max normal value: 448.0
    MAX_VAL = 448.0
    x = float(x)
    if np.isnan(x):
        return x
    x = np.clip(x, -MAX_VAL, MAX_VAL)
    if x == 0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    exp = np.floor(np.log2(x))
    exp = int(np.clip(exp, -6, 8))         # biased range
    mantissa = x / (2.0 ** exp)
    mantissa_int = int(round(mantissa * 8)) & 0xF  # 3-bit mantissa → scale by 8
    return sign * (mantissa_int / 8.0) * (2.0 ** exp)


def _manual_fp8_e4m3(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32).ravel()
    result = np.array([_fp8_e4m3_quantize_scalar(x) for x in f32], dtype=np.float32)
    return result.reshape(arr.shape).astype(np.float16)


def _manual_fp8_e5m2(arr: np.ndarray) -> np.ndarray:
    """Approximate FP8-E5M2: clip to range and round mantissa to 2 bits."""
    MAX_VAL = 57344.0
    f32 = arr.astype(np.float32)
    f32 = np.clip(f32, -MAX_VAL, MAX_VAL)
    # Round mantissa to 2 bits: scale by 4, round, scale back
    sign = np.sign(f32)
    x = np.abs(f32)
    # Find exponent
    safe_x = np.where(x == 0, 1.0, x)
    exp = np.floor(np.log2(safe_x)).astype(np.int32)
    exp = np.clip(exp, -14, 15)
    scale = np.power(2.0, exp.astype(np.float32))
    mantissa = x / scale          # in [1, 2)
    mantissa_q = np.round(mantissa * 4) / 4.0    # 2-bit mantissa
    result = sign * mantissa_q * scale
    result = np.where(x == 0, 0.0, result)
    return result.astype(np.float16)


# ── Registry ──────────────────────────────────────────────────────────────────

# ── Registry (named variants only; int{N}_{ch,tok}_g{G} resolved dynamically) ─

QUANT_FNS = {
    "fp16":      quant_fp16,
    "bf16":      quant_bf16,
    "fp8_e4m3":  quant_fp8_e4m3,
    "fp8_e5m2":  quant_fp8_e5m2,
    "int8":      quant_int8,
    "int8_ch":   quant_int8_ch,
    "int8_tok":  quant_int8_tok,
    "int4":      quant_int4,
    "int4_ch":   quant_int4_ch,
    "int4_tok":  quant_int4_tok,
    "int3":      quant_int3,
    "int3_ch":   quant_int3_ch,
    "int3_tok":  quant_int3_tok,
    "nf4":       quant_nf4,
    "int2":          quant_int2,
    "int2_ch":       quant_int2_ch,
    "int2_tok":      quant_int2_tok,
    "int3_half":     quant_int3_half,
    "int3_half_ch":  quant_int3_half_ch,
    "int3_half_tok": quant_int3_half_tok,
}


def _parse_int3_half_name(name: str):
    """Parse int3_half_ABCD[_ch|_tok] → (indices_tuple, axis) or None.

    Examples:
        int3_half_0247_ch  → ((0,2,4,7), 0)
        int3_half_0357     → ((0,3,5,7), None)
        int3_half_0267_tok → ((0,2,6,7), 1)
    """
    m = _re.match(r'^int3_half_(\d{4})(?:_(ch|tok))?$', name)
    if m is None:
        return None
    indices = tuple(int(c) for c in m.group(1))
    if len(set(indices)) != 4 or not all(0 <= i <= 7 for i in indices):
        return None
    axis = {"ch": 0, "tok": 1, None: None}[m.group(2)]
    return indices, axis


def _is_valid_quant(name: str) -> bool:
    return (name in QUANT_FNS
            or _parse_grouped_name(name) is not None
            or _parse_nbin_name(name) is not None
            or _parse_int3_half_name(name) is not None)


_FLOAT_QUANTS = {"fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "nf4"}


def get_quant_fn(name: str, asym: bool = False):
    """Return the quantization function for the given name.

    asym=True: use asymmetric (min+scale) quantization for all integer variants.
    Has no effect on fp16/bf16/fp8/nf4 which are float formats.

    Supports:
    - All named variants in QUANT_FNS (fp16, int2_ch, int4_tok, nf4, ...)
    - Any int{N}_{ch,tok}_g{G} for N in {2,3,4,8} and any G >= 1
    - Any q{N} / q{N}_ch / q{N}_tok for N >= 2
    """
    # Float formats are unaffected by --asym
    if name in _FLOAT_QUANTS:
        return QUANT_FNS[name]

    if asym:
        # Named int variants: extract bits and axis from the name
        parsed_g = _parse_grouped_name(name)
        if parsed_g:
            bits, axis, gsize = parsed_g
            def fn(arr, b=bits, ax=axis, gs=gsize):
                return _uniform_quant_asym(arr, bits=b, axis=ax, dim_group_size=gs)
            return fn
        # Plain int{N}, int{N}_ch, int{N}_tok
        _plain = _re.match(r'^int(\d+)(?:_(ch|tok))?$', name)
        if _plain:
            bits = int(_plain.group(1))
            gran = _plain.group(2)
            axis = {"ch": 0, "tok": 1, None: None}[gran]
            if bits in _VALID_BITS:
                def fn(arr, b=bits, ax=axis):
                    return _uniform_quant_asym(arr, bits=b, axis=ax)
                return fn
        # q{N} bin-count variants: asym uses n_levels = n_bins - 1
        # e.g. q5_ch --asym → 5 bins evenly spaced from [xmin, xmax]
        parsed2 = _parse_nbin_name(name)
        if parsed2:
            n_bins, axis = parsed2
            def fn(arr, nl=n_bins - 1, ax=axis):
                return _uniform_quant_asym(arr, axis=ax, n_levels=nl)
            return fn

    if name in QUANT_FNS:
        return QUANT_FNS[name]
    parsed = _parse_grouped_name(name)
    if parsed:
        bits, axis, gsize = parsed
        return _make_grouped_fn(bits, axis, gsize)
    parsed2 = _parse_nbin_name(name)
    if parsed2:
        n_bins, axis = parsed2
        def fn(arr, nb=n_bins, ax=axis):
            return _nbin_quant(arr, n_bins=nb, axis=ax)
        return fn
    parsed3 = _parse_int3_half_name(name)
    if parsed3:
        indices, axis = parsed3
        def fn(arr, _idx=indices, _ax=axis):
            return _int3_half_quant(arr, axis=_ax, indices=_idx)
        return fn
    raise ValueError(
        f"Unknown quantization: '{name}'. "
        f"Named variants: {list(QUANT_FNS)}. "
        f"Dynamic grouped: int{{2,3,4,8}}_{{ch,tok}}_g{{N}}. "
        f"Arbitrary bins: q{{N}} / q{{N}}_ch / q{{N}}_tok for any N >= 2. "
        f"int3_half combos: int3_half_ABCD[_ch|_tok] (4 digits 0-7)."
    )


def parse_kv_quant(spec: str):
    """
    Parse a K:V quant spec. Returns (k_name, v_name).
    "int8_ch"          → K=int8_ch, V=int8_ch
    "int8_ch:int4_tok" → K=int8_ch, V=int4_tok
    """
    if ":" in spec:
        k_name, v_name = spec.split(":", 1)
    else:
        k_name = v_name = spec
    for name in (k_name, v_name):
        if not _is_valid_quant(name):
            raise ValueError(f"Unknown quantization: '{name}'")
    return k_name, v_name


def parse_layer_spec(spec: str, n_layers: int):
    """
    Parse a per-layer quant spec into a list of quant names, one per layer.

    Formats:
      "int8_ch"                          -> all layers use int8_ch
      "int8_ch@0-15/int4_ch@16-31"       -> layers 0-15: int8_ch, 16-31: int4_ch
      "int8_ch@0-7/int4_ch@8-23/int2@24-31"  -> three explicit ranges

    Layers not covered by any segment default to "fp16".
    """
    spec = spec.strip()
    if "/" not in spec and "@" not in spec:
        if not _is_valid_quant(spec):
            raise ValueError(f"Unknown quantization: '{spec}'")
        return [spec] * n_layers

    result = ["fp16"] * n_layers
    for seg in spec.split("/"):
        seg = seg.strip()
        if not seg:
            continue
        if "@" in seg:
            name, rng = seg.rsplit("@", 1)
            name = name.strip()
            lo_s, hi_s = rng.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if not _is_valid_quant(name):
                raise ValueError(f"Unknown quantization in layer spec: '{name}'")
            for i in range(lo, hi + 1):
                if 0 <= i < n_layers:
                    result[i] = name
        else:
            if not _is_valid_quant(seg):
                raise ValueError(f"Unknown quantization in layer spec: '{seg}'")
            for i in range(n_layers):
                result[i] = seg
    return result


def resolve_quant_layers(spec: str, n_layers: int):
    """
    Parse a full quant spec with optional K:V split and per-layer ranges.
    Returns (k_names, v_names) — each a list[str] of length n_layers.

    Examples:
      "int8_ch"                              -> K=int8_ch all layers, V=int8_ch all layers
      "int8_ch:int4_tok"                     -> K=int8_ch, V=int4_tok, all layers
      "int8_ch@0-15/int4_ch@16-31"           -> K: layer-split, V: same split
      "int8_ch@0-15/int4_ch@16-31:int4_tok"  -> K: layer-split, V: int4_tok all layers
      "int8_ch:int8_tok@0-15/int4_tok@16-31" -> K: int8_ch all layers, V: layer-split
    """
    if ":" in spec:
        k_spec, v_spec = spec.split(":", 1)
    else:
        k_spec = v_spec = spec
    return parse_layer_spec(k_spec, n_layers), parse_layer_spec(v_spec, n_layers)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # (128, 128): 128 tokens and head_dim=128, divisible by all tested group sizes
    arr = rng.standard_normal((128, 128)).astype(np.float16)

    # Named variants
    for name in QUANT_FNS:
        fn = get_quant_fn(name)
        q  = fn(arr)
        mse = float(np.mean((arr.astype(np.float32) - q.astype(np.float32))**2))
        print(f"  {name:16s}  MSE={mse:.6f}")

    # Dynamic grouped variants (spot-check a range of group sizes)
    dynamic_names = [
        "int4_tok_g1", "int4_tok_g8", "int4_tok_g16", "int4_tok_g32", "int4_tok_g64",
        "int4_ch_g1",  "int4_ch_g8",  "int4_ch_g32",  "int4_ch_g64",  "int4_ch_g128",
        "int3_tok_g1", "int3_ch_g8",
    ]
    for name in dynamic_names:
        fn  = get_quant_fn(name)
        q   = fn(arr)
        mse = float(np.mean((arr.astype(np.float32) - q.astype(np.float32))**2))
        in_per_tok  = name in PER_TOKEN_QUANTS
        in_ch_gs    = name in CH_QUANT_GROUP_SIZE
        gs          = CH_QUANT_GROUP_SIZE[name] if in_ch_gs else "-"
        print(f"  {name:20s}  MSE={mse:.6f}  per_tok={in_per_tok}  ch_gs={gs}")

    print("quant.py self-test PASSED")
