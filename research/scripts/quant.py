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
    "int2":      quant_int2,
    "int2_ch":   quant_int2_ch,
    "int2_tok":  quant_int2_tok,
}


def _is_valid_quant(name: str) -> bool:
    return (name in QUANT_FNS
            or _parse_grouped_name(name) is not None
            or _parse_nbin_name(name) is not None)


def get_quant_fn(name: str):
    """Return the quantization function for the given name.

    Supports:
    - All named variants in QUANT_FNS (fp16, int2_ch, int4_tok, nf4, ...)
    - Any int{N}_{ch,tok}_g{G} for N in {2,3,4,8} and any G >= 1
    - Any q{N} / q{N}_ch / q{N}_tok for N >= 2
      (e.g. q5_ch, q6_tok, q7 — arbitrary bin counts between int2 and int3,
       or between int3 and int4, etc.)
    """
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
    raise ValueError(
        f"Unknown quantization: '{name}'. "
        f"Named variants: {list(QUANT_FNS)}. "
        f"Dynamic grouped: int{{2,3,4,8}}_{{ch,tok}}_g{{N}}. "
        f"Arbitrary bins: q{{N}} / q{{N}}_ch / q{{N}}_tok for any N >= 2."
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
