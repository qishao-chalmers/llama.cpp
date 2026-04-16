#!/usr/bin/env python3
"""
build_q4km_delta.py — Phase 0: validate and build Q4_K_M + int4-delta GGUF.

The delta is defined in Q8_0 *integer* space so verify reconstruction is exact:

    q4km_proj[i] = clamp(round(q4km_float[i] / d_q8[b]), -128, 127)
    delta[i]     = q8_int[i] - q4km_proj[i]   ∈ [-8, 7] for typical weights
    w_verify[i]  = (q4km_proj[i] + delta[i]) * d_q8[b]   == Q8_0 dequant exactly

One model in GPU memory.  Draft reads Q4_K_M base (~4.5 bpw); verify reads base +
delta (~9.0 bpw), recovering exact Q8_0 quality without a second model.

Usage
-----
  # Validate delta distribution and reconstruction error (no output written):
  python3 build_q4km_delta.py \\
      --q8   models/Qwen3-8B-Q8_0.gguf \\
      --q4km models/Qwen3-8B-Q8_0-Q4_K_M.gguf

  # Build the combined Q4_K_D4 GGUF:
  python3 build_q4km_delta.py \\
      --q8   models/Qwen3-8B-Q8_0.gguf \\
      --q4km models/Qwen3-8B-Q8_0-Q4_K_M.gguf \\
      --out  models/Qwen3-8B-Q4_K_D4.gguf

  # Limit to N Q4_K tensors (for quick testing):
  python3 build_q4km_delta.py ... --limit 4

  # Residual delta encoded as Q4_K blocks (same layout as base Q4_K; needs libggml.so):
  python3 build_q4km_delta.py ... --delta-format q4k

  # Compare int4 integer delta vs Q4_K float residual (recommended):
  python3 build_q4km_delta.py ... --delta-format both

  # Write a full GGUF with Q4_K tensors replaced by Q4_K_RES (288 B / 256 el; needs libggml):
  python3 build_q4km_delta.py ... --delta-format q4k --out-layout q4_k_res --out model_q4k_res.gguf

Dependencies: numpy (already required by gguf-py).
Optional: local `build/bin/libggml-base.so` (preferred) or `libggml.so`, or `GGML_SO` for `--delta-format q4k|both` and Q4_K_RES packing.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Allow running from any working directory
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "gguf-py"))

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType, Keys
from gguf.quants import Q4_K, quant_shape_from_byte_shape, quant_shape_to_byte_shape


# ── constants ─────────────────────────────────────────────────────────────────

QK_K    = 256           # Q4_K super-block size (elements)
Q4K_SZ  = 144           # bytes per Q4_K super-block
Q8_BSZ  = 34            # bytes per Q8_0 block (2 scale + 32 int8)
Q8_BS   = 32            # elements per Q8_0 block
N_SUB   = QK_K // Q8_BS  # = 8   Q8_0 sub-blocks per Q4_K super-block
DELTA_SUB_SZ = 2 + 16   # bytes per delta sub-block: fp16 scale + 16 bytes int4
DELTA_SZ     = N_SUB * DELTA_SUB_SZ   # = 144 bytes per super-block
BLOCK_SZ     = Q4K_SZ + DELTA_SZ      # = 288 bytes per block_q4km_d4
# block_q4_k_res: base Q4_K + residual Q4_K (matches GGML_TYPE_Q4_K_RES*)
Q4_K_RES_SZ  = Q4K_SZ * 2             # = 288 bytes per super-block

# Must match ggml.h / GGMLQuantizationType.Q4_K
GGML_TYPE_Q4_K = 12


# ── libggml: Q4_K quantize (Python Q4_K has dequant only) ───────────────────

def _find_ggml_quantize_library(repo_root: Path) -> Optional[Path]:
    """Locate a shared library that exports ggml_quantize_chunk (in libggml-base or libggml)."""
    env = os.environ.get("GGML_SO")
    if env and Path(env).is_file():
        return Path(env)
    for rel in ("build/bin", "build_release/bin"):
        d = repo_root / rel
        if not d.is_dir():
            continue
        # Prefer libggml-base: single .so, no extra NEEDED libs (avoids libggml-cpu dlopen issues).
        for name in ("libggml-base.so", "libggml-base.so.0"):
            p = d / name
            if p.is_file():
                return p
        for p in sorted(d.glob("libggml-base.so.*"), reverse=True):
            if p.is_file():
                return p
        for name in ("libggml.so", "libggml.so.0"):
            p = d / name
            if p.is_file():
                return p
        for p in sorted(d.glob("libggml.so.*"), reverse=True):
            if p.is_file():
                return p
    return None


def _preload_ggml_deps_for_main_so(lib_dir: Path, main_so: Path) -> None:
    """Load libggml-base / libggml-cpu before libggml.so so dlopen finds NEEDED libs without LD_LIBRARY_PATH."""
    if "ggml-base" in main_so.name:
        return
    # Same directory as the build output; order matches link line.
    for pattern in (
        "libggml-base.so.0",
        "libggml-base.so",
        "libggml-cpu.so.0",
        "libggml-cpu.so",
    ):
        p = lib_dir / pattern
        if p.is_file():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _load_ggml_quantize() -> Any:
    """Return ctypes lib with ggml_quantize_chunk set up, or None."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    so = _find_ggml_quantize_library(repo_root)
    if so is None:
        return None
    so = so.resolve()
    lib_dir = so.parent
    _preload_ggml_deps_for_main_so(lib_dir, so)
    lib = ctypes.CDLL(str(so))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = (
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    )
    return lib


def ggml_quantize_row_q4_k(src_f32_256: np.ndarray, lib: ctypes.CDLL) -> np.ndarray:
    """Quantize one 256-element float row to one Q4_K block (144 bytes)."""
    assert src_f32_256.shape == (256,) and src_f32_256.dtype == np.float32
    dst = np.empty(144, dtype=np.uint8)
    nbytes = lib.ggml_quantize_chunk(
        GGML_TYPE_Q4_K,
        src_f32_256.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.c_void_p),
        0,
        1,
        256,
        None,
    )
    assert nbytes == 144
    return dst


def ggml_quantize_blocks_q4_k(residual_2d: np.ndarray, lib: ctypes.CDLL) -> np.ndarray:
    """
    residual_2d: float32 (n_super_blocks, 256)
    Returns uint8 (n_super_blocks, 144)
    """
    n = residual_2d.shape[0]
    assert residual_2d.shape[1] == 256
    src = np.ascontiguousarray(residual_2d, dtype=np.float32)
    dst = np.empty((n, Q4K_SZ), dtype=np.uint8)
    nbytes = lib.ggml_quantize_chunk(
        GGML_TYPE_Q4_K,
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.c_void_p),
        0,
        ctypes.c_int64(n),
        256,
        None,
    )
    assert nbytes == n * Q4K_SZ
    return dst


# ── dequant helpers ────────────────────────────────────────────────────────────

def _flat_bytes(data: np.ndarray) -> np.ndarray:
    """Return a contiguous 1-D uint8 view regardless of input shape."""
    return np.asarray(data, dtype=np.uint8).reshape(-1)


def dequant_q8_0_tensor(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (q8_int, q8_float) for a raw Q8_0 tensor (any shape, uint8).

    q8_int  : int8   shape (n_elements,) — raw quantized integers
    q8_float: float32 shape (n_elements,) — dequantized floats
    """
    flat     = _flat_bytes(data)
    n_blocks = len(flat) // Q8_BSZ
    blocks   = flat[:n_blocks * Q8_BSZ].reshape(n_blocks, Q8_BSZ)

    d  = blocks[:, :2].view(np.float16).astype(np.float32)        # (n_blocks, 1)
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)           # (n_blocks, 32)

    q8_float = (qs * d).reshape(-1)                                # (n_elements,)
    q8_int   = blocks[:, 2:].view(np.int8).reshape(-1)            # (n_elements,)
    return q8_int, q8_float


def dequant_q4k_tensor(data: np.ndarray) -> np.ndarray:
    """
    Returns float32 dequant of a raw Q4_K tensor (any shape, uint8).
    shape: (n_elements,)
    """
    flat     = _flat_bytes(data)
    n_blocks = len(flat) // Q4K_SZ
    blocks   = flat[:n_blocks * Q4K_SZ].reshape(n_blocks, Q4K_SZ)
    return Q4_K.dequantize_blocks(blocks).reshape(-1)


def q8_scales_per_block(data: np.ndarray) -> np.ndarray:
    """
    Extract Q8_0 scale (fp16→float32) for each 32-element block.
    Returns shape (n_q8_blocks,).
    """
    flat     = _flat_bytes(data)
    n_blocks = len(flat) // Q8_BSZ
    blocks   = flat[:n_blocks * Q8_BSZ].reshape(n_blocks, Q8_BSZ)
    return blocks[:, :2].view(np.float16).astype(np.float32).reshape(-1)   # (n_blocks,)


# ── delta computation ──────────────────────────────────────────────────────────

def compute_delta(q8_flat: np.ndarray, q4k_flat: np.ndarray,
                  n_elements: int,
                  chunk_q4k_blocks: int = 4096) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute int4 delta (in Q8_0 integer units) for one tensor, in chunks.

    delta[i] = q8_int[i] - clamp(round(q4km_float[i] / d_q8[block_i]), -128, 127)

    Processes chunk_q4k_blocks Q4_K super-blocks at a time to limit peak RAM usage.
    One Q4_K super-block = 256 elements = 8 Q8_0 sub-blocks.

    Returns:
        d_q8   : float32 (n_q8_blocks,)  Q8_0 scale per 32-element sub-block
        delta  : int8    (n_elements,)   signed delta, clamped to [-8, 7]
        stats  : dict    histogram / saturation metrics
    """
    n_q4k_blocks  = len(q4k_flat) // Q4K_SZ
    n_q8_blocks   = n_q4k_blocks * N_SUB
    d_q8_out      = np.empty(n_q8_blocks, dtype=np.float32)
    delta_out     = np.empty(n_elements,  dtype=np.int8)

    # Aggregate stats
    n_sat_pos = 0;  n_sat_neg = 0
    delta_min =  16; delta_max = -16
    abs_sum   = 0.0; clamp_sq  = 0.0
    hist      = np.zeros(32, dtype=np.int64)   # bins -16..15

    for chunk_start in range(0, n_q4k_blocks, chunk_q4k_blocks):
        chunk_end  = min(chunk_start + chunk_q4k_blocks, n_q4k_blocks)
        n_chunk    = chunk_end - chunk_start
        elem_start = chunk_start * QK_K
        elem_end   = chunk_end   * QK_K
        q8blk_s    = chunk_start * N_SUB
        q8blk_e    = chunk_end   * N_SUB

        # Slice raw bytes for this chunk
        q4k_chunk = q4k_flat[chunk_start * Q4K_SZ : chunk_end * Q4K_SZ]
        q8_chunk  = q8_flat [q8blk_s * Q8_BSZ    : q8blk_e  * Q8_BSZ  ]

        # Dequantize
        q4km_float = Q4_K.dequantize_blocks(
            q4k_chunk.reshape(n_chunk, Q4K_SZ)).reshape(-1)          # (n_chunk*256,)
        q8_blocks  = q8_chunk.reshape(n_chunk * N_SUB, Q8_BSZ)
        d_q8_chunk = q8_blocks[:, :2].view(np.float16).astype(np.float32).reshape(-1)
        q8_int_c   = q8_blocks[:, 2:].view(np.int8).reshape(-1)     # (n_chunk*256,)

        # Project Q4_K_M float → Q8_0 int space
        d_per_elem = np.repeat(d_q8_chunk, Q8_BS)
        with np.errstate(divide="ignore", invalid="ignore"):
            q4km_proj = np.where(
                d_per_elem == 0, 0,
                np.round(q4km_float / d_per_elem)
            ).astype(np.int32)
        q4km_proj = np.clip(q4km_proj, -128, 127)

        delta_full = q8_int_c.astype(np.int32) - q4km_proj

        # Clamp to int4 range [-8, 7]
        nc_pos = int(np.sum(delta_full > 7))
        nc_neg = int(np.sum(delta_full < -8))
        n_sat_pos += nc_pos;  n_sat_neg += nc_neg
        delta_min  = min(delta_min, int(delta_full.min()))
        delta_max  = max(delta_max, int(delta_full.max()))
        abs_sum   += float(np.abs(delta_full).sum())
        delta_c    = np.clip(delta_full, -8, 7).astype(np.int8)

        clamp_err  = (delta_full - delta_c.astype(np.int32)).astype(np.float32)
        clamp_sq  += float(np.sum((clamp_err * d_per_elem) ** 2))

        h, _ = np.histogram(delta_full, bins=range(-16, 17))
        hist += h

        d_q8_out[q8blk_s:q8blk_e]   = d_q8_chunk
        delta_out[elem_start:elem_end] = delta_c

    n_sat = n_sat_pos + n_sat_neg
    stats = {
        "n_elements":     n_elements,
        "n_sat_pos":      n_sat_pos,
        "n_sat_neg":      n_sat_neg,
        "n_saturated":    n_sat,
        "sat_rate":       n_sat / max(n_elements, 1),
        "delta_min":      delta_min,
        "delta_max":      delta_max,
        "delta_abs_mean": abs_sum / max(n_elements, 1),
        "clamp_err_rms":  float(np.sqrt(clamp_sq / max(n_elements, 1))),
        "histogram":      hist.tolist(),  # 32 bins: -16..15
    }
    return d_q8_out, delta_out, stats


# ── pack / unpack delta sub-blocks ─────────────────────────────────────────────

def pack_delta_blocks(d_q8: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Pack delta into the block_q4km_d4 delta section.

    d_q8  : float32 (n_q8_blocks,)
    delta : int8    (n_elements,)     signed, clamped to [-8, 7]

    Returns uint8 array of shape (n_q4k_blocks, DELTA_SZ)
    where DELTA_SZ = 8 × (2 fp16 + 16 int4_packed) = 144 bytes.
    """
    n_q8_blocks = d_q8.shape[0]
    n_q4k_blocks = n_q8_blocks // N_SUB

    d_q8_fp16 = d_q8.astype(np.float16).view(np.uint8).reshape(n_q4k_blocks, N_SUB, 2)

    # bias-encode signed int4 [-8,7] → unsigned [0,15]
    biased = (delta.astype(np.int32) + 8).astype(np.uint8)           # [0..15]
    biased = biased.reshape(n_q4k_blocks, N_SUB, Q8_BS)              # (B, 8, 32)

    # pack two nibbles per byte
    lo = biased[:, :, 0::2]  # even elements → lower nibble
    hi = biased[:, :, 1::2]  # odd  elements → upper nibble
    packed = (lo | (hi << 4)).reshape(n_q4k_blocks, N_SUB, 16)       # (B, 8, 16)

    # interleave: [fp16_scale(2B), packed_nibbles(16B)] × 8 = 144 bytes
    out = np.zeros((n_q4k_blocks, N_SUB, DELTA_SUB_SZ), dtype=np.uint8)
    out[:, :, :2]  = d_q8_fp16
    out[:, :, 2:]  = packed
    return out.reshape(n_q4k_blocks, DELTA_SZ)


def unpack_delta_blocks(delta_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack delta section back to (d_q8, delta_int8).

    delta_raw : uint8 (n_q4k_blocks, DELTA_SZ)

    Returns:
        d_q8  : float32 (n_q8_blocks,)
        delta : int8    (n_elements,)   signed, range [-8, 7]
    """
    n_q4k_blocks = delta_raw.shape[0]
    raw = delta_raw.reshape(n_q4k_blocks, N_SUB, DELTA_SUB_SZ)

    d_q8 = raw[:, :, :2].reshape(-1, 2).view(np.float16).astype(np.float32).reshape(-1)

    packed = raw[:, :, 2:].reshape(n_q4k_blocks, N_SUB, 16)          # (B, 8, 16)
    lo = (packed & 0x0F).reshape(n_q4k_blocks, N_SUB, 16)
    hi = (packed >> 4).reshape(n_q4k_blocks, N_SUB, 16)
    # interleave lo/hi → 32 elements per sub-block
    biased = np.empty((n_q4k_blocks, N_SUB, 32), dtype=np.uint8)
    biased[:, :, 0::2] = lo
    biased[:, :, 1::2] = hi
    delta = (biased.astype(np.int32) - 8).astype(np.int8).reshape(-1)
    return d_q8, delta


# ── verify reconstruction accuracy ────────────────────────────────────────────

def verify_reconstruction(q4k_flat: np.ndarray, d_q8: np.ndarray,
                           delta: np.ndarray,
                           q8_flat: np.ndarray,
                           chunk_q4k_blocks: int = 4096) -> dict:
    """
    Confirm that Q4_K_M + delta reconstructs Q8_0 exactly (within clamping).
    Runs in the same chunked fashion to limit peak RAM.
    """
    n_q4k_blocks = len(q4k_flat) // Q4K_SZ
    all_exact   = True
    sq_err_v    = 0.0;  sq_err_q4 = 0.0
    max_abs_v   = 0.0
    n_total     = 0

    for chunk_start in range(0, n_q4k_blocks, chunk_q4k_blocks):
        chunk_end  = min(chunk_start + chunk_q4k_blocks, n_q4k_blocks)
        n_chunk    = chunk_end - chunk_start
        elem_s     = chunk_start * QK_K
        elem_e     = chunk_end   * QK_K
        q8blk_s    = chunk_start * N_SUB
        q8blk_e    = chunk_end   * N_SUB

        q4k_chunk = q4k_flat[chunk_start * Q4K_SZ : chunk_end * Q4K_SZ]
        q8_chunk  = q8_flat [q8blk_s * Q8_BSZ     : q8blk_e  * Q8_BSZ ]

        q4km_float = Q4_K.dequantize_blocks(
            q4k_chunk.reshape(n_chunk, Q4K_SZ)).reshape(-1)
        q8_blocks  = q8_chunk.reshape(n_chunk * N_SUB, Q8_BSZ)
        d_q8_c     = q8_blocks[:, :2].view(np.float16).astype(np.float32).reshape(-1)
        q8_int_c   = q8_blocks[:, 2:].view(np.int8).reshape(-1)
        q8_float_c = (q8_blocks[:, 2:].view(np.int8).astype(np.float32)
                      * d_q8_c.reshape(-1, 1)).reshape(-1)

        d_pe       = np.repeat(d_q8_c, Q8_BS)
        with np.errstate(divide="ignore", invalid="ignore"):
            q4km_proj = np.where(d_pe == 0, 0,
                                 np.round(q4km_float / d_pe)).astype(np.int32)
        q4km_proj = np.clip(q4km_proj, -128, 127)

        delta_c   = delta[elem_s:elem_e].astype(np.int32)
        q8_rec    = q4km_proj + delta_c
        w_verify  = q8_rec.astype(np.float32) * d_pe

        if not np.all(q8_rec == q8_int_c.astype(np.int32)):
            all_exact = False

        err_v  = w_verify - q8_float_c
        err_q4 = q4km_float - q8_float_c
        sq_err_v  += float(np.sum(err_v  ** 2))
        sq_err_q4 += float(np.sum(err_q4 ** 2))
        max_abs_v  = max(max_abs_v, float(np.abs(err_v).max()))
        n_total   += len(q4km_float)

    return {
        "exact_int_reconstruction": all_exact,
        "verify_rms_err":  float(np.sqrt(sq_err_v  / max(n_total, 1))),
        "verify_max_abs":  float(max_abs_v),
        "q4km_rms_err":    float(np.sqrt(sq_err_q4 / max(n_total, 1))),
    }


def pack_and_stats_q4k_residual(
        q4k_flat: np.ndarray,
        q8_flat: np.ndarray,
        lib: ctypes.CDLL,
        out_packed: np.ndarray,          # pre-allocated uint8 (n_q4k_blocks * Q4_K_RES_SZ,)
        chunk_q4k_blocks: int = 4096) -> dict:
    """
    Single-pass: compute packed block_q4_k_res bytes AND error stats.

    Replaces three separate passes (verify_reconstruction_q4k_residual,
    pack_tensor_q4_k_res, validate_packed_q4_k_res) with one loop that
    dequantizes Q4_K + Q8_0 exactly once per chunk.

        residual[i] = q8_float[i] - q4km_float[i]
        packed[b]   = [base_q4k_block(144B) | quantize_q4_K(residual)(144B)]
        w_verify[i] = q4km_float[i] + dequant(residual_q4k)[i]

    out_packed: caller-provided output buffer; filled in-place so the caller
                can write it to disk immediately per chunk (streaming).
    """
    n_q4k_blocks = len(q4k_flat) // Q4K_SZ
    sq_err_v   = 0.0
    sq_err_q4  = 0.0
    sq_res_enc = 0.0
    max_abs_v  = 0.0
    n_total    = 0

    for chunk_start in range(0, n_q4k_blocks, chunk_q4k_blocks):
        chunk_end = min(chunk_start + chunk_q4k_blocks, n_q4k_blocks)
        n_chunk   = chunk_end - chunk_start
        q8blk_s   = chunk_start * N_SUB
        q8blk_e   = chunk_end   * N_SUB

        q4k_chunk = q4k_flat[chunk_start * Q4K_SZ : chunk_end * Q4K_SZ]
        q8_chunk  = q8_flat [q8blk_s * Q8_BSZ     : q8blk_e  * Q8_BSZ ]

        # ── single dequant of Q4_K and Q8_0 for this chunk ──────────────────
        q4k_blocks_2d = q4k_chunk.reshape(n_chunk, Q4K_SZ)
        q4km_float    = Q4_K.dequantize_blocks(q4k_blocks_2d).reshape(-1)

        q8_blocks  = q8_chunk.reshape(n_chunk * N_SUB, Q8_BSZ)
        d_q8_c     = q8_blocks[:, :2].view(np.float16).astype(np.float32).reshape(-1)
        q8_float_c = (q8_blocks[:, 2:].view(np.int8).astype(np.float32)
                      * d_q8_c.reshape(-1, 1)).reshape(-1)

        # ── pack: base + quantized residual ──────────────────────────────────
        residual    = (q8_float_c - q4km_float).reshape(n_chunk, QK_K)
        packed_res  = ggml_quantize_blocks_q4_k(residual, lib)  # (n_chunk, 144)
        res_recon   = Q4_K.dequantize_blocks(packed_res).reshape(-1)

        merged = np.concatenate([q4k_blocks_2d, packed_res], axis=1)  # (n_chunk, 288)
        out_packed[chunk_start * Q4_K_RES_SZ : chunk_end * Q4_K_RES_SZ] = merged.ravel()

        # ── error stats (no extra dequant needed) ────────────────────────────
        w_verify = q4km_float + res_recon
        err_v    = w_verify - q8_float_c
        err_q4   = q4km_float - q8_float_c
        err_e    = res_recon - residual.reshape(-1)

        sq_err_v   += float(np.sum(err_v  ** 2))
        sq_err_q4  += float(np.sum(err_q4 ** 2))
        sq_res_enc += float(np.sum(err_e  ** 2))
        max_abs_v   = max(max_abs_v, float(np.abs(err_v).max()))
        n_total    += len(q4km_float)

    nt = max(n_total, 1)
    return {
        "verify_rms_err":   float(np.sqrt(sq_err_v   / nt)),
        "verify_max_abs":   float(max_abs_v),
        "q4km_rms_err":     float(np.sqrt(sq_err_q4  / nt)),
        "residual_q4k_rms": float(np.sqrt(sq_res_enc / nt)),
        "sum_sq_verify":    float(sq_err_v),
        "sum_sq_q4km":      float(sq_err_q4),
        "sum_sq_res_enc":   float(sq_res_enc),
        "n_elements":       int(n_total),
    }


def verify_reconstruction_q4k_residual(
    q4k_flat: np.ndarray,
    q8_flat: np.ndarray,
    lib: ctypes.CDLL,
    chunk_q4k_blocks: int = 4096,
) -> dict:
    """
    Same numerics as pack_and_stats_q4k_residual: RMS stats for Q4_K residual packing.
    Allocates a temporary packed buffer (discarded); use pack_and_stats_q4k_residual if you need the bytes.
    """
    n_q4k_blocks = len(q4k_flat) // Q4K_SZ
    out = np.empty(n_q4k_blocks * Q4_K_RES_SZ, dtype=np.uint8)
    return pack_and_stats_q4k_residual(
        q4k_flat, q8_flat, lib, out, chunk_q4k_blocks=chunk_q4k_blocks)


# ── block_q4_k_res: pack + NumPy dequant (matches ggml block_q4_k_res layout) ─


def pack_tensor_q4_k_res(
    q4k_flat: np.ndarray,
    q8_flat: np.ndarray,
    lib: ctypes.CDLL,
    chunk_q4k_blocks: int = 4096,
) -> np.ndarray:
    """
    Build raw bytes for GGML_TYPE_Q4_K_RES: per 256 elements, 144 B base (Q4_K_M) +
    144 B residual quantized as Q4_K from (q8_float − q4km_float).
    """
    n_q4k_blocks = len(q4k_flat) // Q4K_SZ
    out = np.empty(n_q4k_blocks * Q4_K_RES_SZ, dtype=np.uint8)

    for chunk_start in range(0, n_q4k_blocks, chunk_q4k_blocks):
        chunk_end = min(chunk_start + chunk_q4k_blocks, n_q4k_blocks)
        n_chunk = chunk_end - chunk_start
        q8blk_s = chunk_start * N_SUB
        q8blk_e = chunk_end * N_SUB

        q4k_chunk = q4k_flat[chunk_start * Q4K_SZ : chunk_end * Q4K_SZ]
        q8_chunk = q8_flat[q8blk_s * Q8_BSZ : q8blk_e * Q8_BSZ]

        q4km_float = Q4_K.dequantize_blocks(
            q4k_chunk.reshape(n_chunk, Q4K_SZ)).reshape(-1)
        q8_blocks = q8_chunk.reshape(n_chunk * N_SUB, Q8_BSZ)
        d_q8_c = q8_blocks[:, :2].view(np.float16).astype(np.float32).reshape(-1)
        q8_float_c = (q8_blocks[:, 2:].view(np.int8).astype(np.float32)
                      * d_q8_c.reshape(-1, 1)).reshape(-1)

        residual = q8_float_c - q4km_float
        res_blocks = residual.reshape(n_chunk, QK_K)
        packed_res = ggml_quantize_blocks_q4_k(res_blocks, lib)
        base_blocks = q4k_chunk.reshape(n_chunk, Q4K_SZ)
        merged = np.concatenate([base_blocks, packed_res], axis=1)

        out[chunk_start * Q4_K_RES_SZ : chunk_end * Q4_K_RES_SZ] = merged.ravel()

    return out


def dequant_q4_k_res_draft_numpy(packed_flat: np.ndarray) -> np.ndarray:
    """float32 draft weights: dequantize .base only (first 144 B per super-block)."""
    n_blocks = len(packed_flat) // Q4_K_RES_SZ
    b = packed_flat.reshape(n_blocks, Q4_K_RES_SZ)
    base = b[:, :Q4K_SZ]
    return Q4_K.dequantize_blocks(base).reshape(-1)


def dequant_q4_k_res_verify_numpy(packed_flat: np.ndarray) -> np.ndarray:
    """float32 verify weights: dequant(base) + dequant(res)."""
    n_blocks = len(packed_flat) // Q4_K_RES_SZ
    b = packed_flat.reshape(n_blocks, Q4_K_RES_SZ)
    base = b[:, :Q4K_SZ]
    res = b[:, Q4K_SZ:]
    return (
        Q4_K.dequantize_blocks(base) + Q4_K.dequantize_blocks(res)
    ).reshape(-1)


def validate_packed_q4_k_res(
    packed_flat: np.ndarray,
    q4k_flat: np.ndarray,
    q8_flat: np.ndarray,
    chunk_q4k_blocks: int = 4096,
) -> dict:
    """
    Compare packed block_q4_k_res against Q8 float and Q4_K_M-only float (draft path).
    RMS values should match verify_reconstruction_q4k_residual on the same inputs.
    """
    n_q4k_blocks = len(q4k_flat) // Q4K_SZ
    sq_err_v = 0.0
    sq_err_d = 0.0
    n_total = 0

    for chunk_start in range(0, n_q4k_blocks, chunk_q4k_blocks):
        chunk_end = min(chunk_start + chunk_q4k_blocks, n_q4k_blocks)
        n_chunk = chunk_end - chunk_start
        elem_s = chunk_start * QK_K
        elem_e = chunk_end * QK_K
        q8blk_s = chunk_start * N_SUB
        q8blk_e = chunk_end * N_SUB

        pk = packed_flat[chunk_start * Q4_K_RES_SZ : chunk_end * Q4_K_RES_SZ]
        q4k_chunk = q4k_flat[chunk_start * Q4K_SZ : chunk_end * Q4K_SZ]
        q8_chunk = q8_flat[q8blk_s * Q8_BSZ : q8blk_e * Q8_BSZ]

        q4km_float = Q4_K.dequantize_blocks(
            q4k_chunk.reshape(n_chunk, Q4K_SZ)).reshape(-1)
        q8_blocks = q8_chunk.reshape(n_chunk * N_SUB, Q8_BSZ)
        d_q8_c = q8_blocks[:, :2].view(np.float16).astype(np.float32).reshape(-1)
        q8_float_c = (q8_blocks[:, 2:].view(np.int8).astype(np.float32)
                      * d_q8_c.reshape(-1, 1)).reshape(-1)

        w_v = dequant_q4_k_res_verify_numpy(pk)
        w_d = dequant_q4_k_res_draft_numpy(pk)

        sq_err_v += float(np.sum((w_v - q8_float_c) ** 2))
        sq_err_d += float(np.sum((w_d - q4km_float) ** 2))
        n_total += len(q8_float_c)

    nt = max(n_total, 1)
    return {
        "pack_verify_rms": float(np.sqrt(sq_err_v / nt)),
        "pack_draft_rms":  float(np.sqrt(sq_err_d / nt)),
        "n_elements": int(n_total),
        "sum_sq_verify": float(sq_err_v),
    }


def copy_gguf_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy KV fields from reader (skip GGUF.* and architecture — writer already set arch)."""
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == Keys.General.ARCHITECTURE:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        val = field.contents()
        if val is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)


def write_gguf_q4_k_res(
    out_path: str | Path,
    reader_src: GGUFReader,
    packed_by_name: dict[str, np.ndarray],
    arch: str,
) -> None:
    """
    Clone reader_src to out_path, replacing tensors in packed_by_name with Q4_K_RES uint8 blobs.
    """
    writer = GGUFWriter(str(out_path), arch=arch, endianess=reader_src.endianess)
    writer.data_alignment = int(reader_src.alignment)
    copy_gguf_metadata(reader_src, writer)

    for t in reader_src.tensors:
        if t.name in packed_by_name:
            logical = quant_shape_from_byte_shape(t.data.shape, GGMLQuantizationType.Q4_K)
            byte_shape = quant_shape_to_byte_shape(logical, GGMLQuantizationType.Q4_K_RES)
            data = packed_by_name[t.name].reshape(byte_shape)
            writer.add_tensor(
                t.name,
                data,
                raw_dtype=GGMLQuantizationType.Q4_K_RES,
                tensor_endianess=reader_src.endianess,
            )
        elif t.tensor_type == GGMLQuantizationType.F32:
            writer.add_tensor(t.name, t.data.astype(np.float32), tensor_endianess=reader_src.endianess)
        elif t.tensor_type == GGMLQuantizationType.F16:
            writer.add_tensor(t.name, t.data, tensor_endianess=reader_src.endianess)
        elif t.tensor_type == GGMLQuantizationType.BF16:
            writer.add_tensor(t.name, t.data, tensor_endianess=reader_src.endianess)
        else:
            writer.add_tensor(
                t.name,
                t.data,
                raw_dtype=t.tensor_type,
                tensor_endianess=reader_src.endianess,
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def reader_architecture(reader: GGUFReader) -> str:
    f = reader.get_field(Keys.General.ARCHITECTURE)
    if f is None:
        return "llama"
    return str(f.contents())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build or validate Q4_K_M + delta (int4 integer or Q4_K float residual).")
    ap.add_argument("--q8",   required=True,
                    help="Pure Q8_0 GGUF (reference weights, not loaded to GPU).")
    ap.add_argument("--q4km", required=True,
                    help="Mixed Q8_0-Q4_K_M GGUF (draft model base).")
    ap.add_argument("--out",  default=None,
                    help="Output path for Q4_K_D4 GGUF (omit for validate-only).")
    ap.add_argument("--delta-format", choices=("int4", "q4k", "both"),
                    default="int4",
                    help="int4: integer delta in Q8 scale units (default). "
                         "q4k: float residual q8−q4km quantized per 256 el as Q4_K (needs libggml). "
                         "both: run int4 + q4k error checks.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N Q4_K tensors (for quick testing).")
    ap.add_argument("--max-elem", type=int, default=100_000_000,
                    help="Skip tensors with more than this many elements (default 100M).")
    ap.add_argument("--chunk-blocks", type=int, default=4096,
                    help="Q4_K super-blocks to process per chunk (controls peak RAM, default 4096 ≈ 1B elements).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-tensor stats.")
    ap.add_argument(
        "--out-layout",
        choices=("none", "q4_k_res"),
        default="none",
        help="none: do not write merged GGUF. q4_k_res: write --out with Q4_K → Q4_K_RES tensors (needs --delta-format q4k or both).",
    )
    args = ap.parse_args()

    if args.out_layout == "q4_k_res":
        if not args.out:
            print("ERROR: --out-layout q4_k_res requires --out <path.gguf>", file=sys.stderr)
            sys.exit(1)
        if args.delta_format not in ("q4k", "both"):
            print("ERROR: --out-layout q4_k_res needs --delta-format q4k or both", file=sys.stderr)
            sys.exit(1)

    do_int4 = args.delta_format in ("int4", "both")
    do_q4k  = args.delta_format in ("q4k", "both")
    ggml_lib = _load_ggml_quantize() if do_q4k else None
    if do_q4k and ggml_lib is None:
        so_hint = _find_ggml_quantize_library(Path(__file__).resolve().parent.parent.parent)
        print(
            "ERROR: --delta-format q4k/both requires libggml-base.so or libggml.so (ggml_quantize_chunk).\n"
            "  Build the project (e.g. build/bin/libggml-base.so), or set GGML_SO to that path.\n"
            "  If you only have libggml.so, ensure build/bin is on LD_LIBRARY_PATH or use libggml-base.so.\n"
            f"  Tried: {so_hint}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.out_layout == "q4_k_res":
        if not do_q4k or ggml_lib is None:
            print(
                "ERROR: --out-layout q4_k_res requires --delta-format q4k or both and libggml (quantize)",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loading Q8_0  GGUF: {args.q8}")
    reader_q8   = GGUFReader(args.q8,   mode="r")
    print(f"Loading Q4_KM GGUF: {args.q4km}")
    reader_q4km = GGUFReader(args.q4km, mode="r")

    # Index Q8_0 tensors by name
    q8_by_name = {t.name: t for t in reader_q8.tensors}
    print(f"  Q8  tensors: {len(q8_by_name)}")

    # Find Q4_K tensors in mixed GGUF
    q4k_tensors = [t for t in reader_q4km.tensors
                   if t.tensor_type == GGMLQuantizationType.Q4_K]
    if args.limit:
        q4k_tensors = q4k_tensors[:args.limit]
    print(f"  Q4K tensors in mixed GGUF: {len(q4k_tensors)}"
          + (f" (limited to {args.limit})" if args.limit else ""))
    print(f"  Delta format: {args.delta_format}")

    # Aggregate stats
    agg_n_elem   = 0
    agg_n_sat    = 0
    agg_exact    = 0
    agg_tensors  = 0
    delta_store  = {}   # name → (d_q8, delta_int8)
    # Q4_K residual: global RMS accumulators (sum of squared errors, element count)
    q4k_sum_sq_verify = 0.0
    q4k_sum_sq_resenc = 0.0
    q4k_sum_sq_q4only = 0.0
    q4k_n_elem        = 0
    packed_by_name: dict[str, np.ndarray] = {}

    print("\n--- Per-tensor validation ---")
    for t in q4k_tensors:
        name = t.name
        if name not in q8_by_name:
            print(f"  SKIP {name}: not found in Q8_0 GGUF")
            continue
        q8t = q8_by_name[name]
        if q8t.tensor_type != GGMLQuantizationType.Q8_0:
            print(f"  SKIP {name}: Q8 GGUF has type {q8t.tensor_type.name} (expected Q8_0)")
            continue

        n_elem = t.n_elements
        if n_elem % QK_K != 0:
            print(f"  SKIP {name}: n_elements={n_elem} not divisible by {QK_K}")
            continue
        if n_elem > args.max_elem:
            print(f"  SKIP {name}: n_elements={n_elem:,} > --max-elem={args.max_elem:,}")
            continue

        # Raw bytes — flatten to 1D; GGUFReader returns 2D (rows × byte_row)
        q8_flat  = _flat_bytes(q8t.data)
        q4k_flat = _flat_bytes(t.data)

        stats = None
        vrec  = None
        vq4k  = None
        d_q8 = delta = None

        if do_int4:
            d_q8, delta, stats = compute_delta(q8_flat, q4k_flat, n_elem,
                                               chunk_q4k_blocks=args.chunk_blocks)
            vrec = verify_reconstruction(q4k_flat, d_q8, delta, q8_flat,
                                       chunk_q4k_blocks=args.chunk_blocks)
            delta_store[name] = (d_q8, delta)

        if do_q4k:
            # Single pass: pack Q4_K_RES bytes + compute error stats simultaneously.
            n_q4k_blks = n_elem // QK_K
            out_packed = np.empty(n_q4k_blks * Q4_K_RES_SZ, dtype=np.uint8)
            vq4k = pack_and_stats_q4k_residual(
                q4k_flat, q8_flat, ggml_lib, out_packed,
                chunk_q4k_blocks=args.chunk_blocks)
            q4k_sum_sq_verify += vq4k["sum_sq_verify"]
            q4k_sum_sq_resenc += vq4k["sum_sq_res_enc"]
            q4k_sum_sq_q4only += vq4k["sum_sq_q4km"]
            q4k_n_elem        += vq4k["n_elements"]
            if args.out_layout == "q4_k_res":
                packed_by_name[name] = out_packed

        agg_n_elem   += n_elem
        if stats is not None:
            agg_n_sat += stats["n_saturated"]
        if vrec is not None:
            agg_exact += int(vrec["exact_int_reconstruction"])
        agg_tensors  += 1

        if args.verbose:
            print(f"  {name}")
            print(f"    shape={t.shape}  n_elem={n_elem}")
            if do_int4 and stats is not None and vrec is not None:
                print(f"    [int4] delta [{stats['delta_min']}, {stats['delta_max']}]"
                      f"  sat={stats['sat_rate']*100:.3f}%"
                      f"  clamp_rms={stats['clamp_err_rms']:.6e}"
                      f"  verify_rms={vrec['verify_rms_err']:.6e}"
                      f"  exact_int={vrec['exact_int_reconstruction']}")
            if do_q4k and vq4k is not None:
                print(f"    [q4k]  verify_rms={vq4k['verify_rms_err']:.6e}"
                      f"  residual_q_enc_rms={vq4k['residual_q4k_rms']:.6e}"
                      f"  q4km_rms={vq4k['q4km_rms_err']:.6e}")
        else:
            parts = []
            if do_int4 and stats is not None and vrec is not None:
                ex = "exact" if vrec["exact_int_reconstruction"] else "APPROX"
                parts.append(f"int4 sat={stats['sat_rate']*100:.3f}% {ex}")
            if do_q4k and vq4k is not None:
                parts.append(f"q4k rms={vq4k['verify_rms_err']:.4e}")
            print(f"  {name:55s}  " + "  ".join(parts))

    print(f"\n--- Aggregate ({agg_tensors} Q4_K tensors, {agg_n_elem:,} elements) ---")
    if do_int4:
        print(f"  [int4] Total saturated elements: {agg_n_sat:,} ({100*agg_n_sat/max(agg_n_elem,1):.4f}%)")
        print(f"  [int4] Exact int reconstruction: {agg_exact}/{agg_tensors} tensors")
        n_inexact = agg_tensors - agg_exact
        if n_inexact > 0:
            print(f"  [int4] WARNING: {n_inexact} tensors have saturation → 1-ULP error on some elements")
        else:
            print(f"  [int4] All tensors reconstruct Q8_0 exactly (no saturation).")
    if do_q4k and q4k_n_elem > 0:
        g_v = float(np.sqrt(q4k_sum_sq_verify / q4k_n_elem))
        g_r = float(np.sqrt(q4k_sum_sq_resenc / q4k_n_elem))
        g_q = float(np.sqrt(q4k_sum_sq_q4only / q4k_n_elem))
        print(f"  [q4k]  Global RMS vs Q8: verify={g_v:.6e}  (base Q4_K_M alone: {g_q:.6e})")
        print(f"  [q4k]  Residual encode RMS (q4k quant of r = q8−q4km): {g_r:.6e}")

    q4k_bpw    = 4.5
    verify_bpw = 9.0
    print(f"\n  Draft  bandwidth: {q4k_bpw:.1f} bpw  (Q4_K_M base only)")
    if do_int4:
        print(f"  Verify bandwidth (int4 delta design): {verify_bpw:.1f} bpw  (Q4_K_M + int4×32 per block)")
    if do_q4k:
        print(f"  Verify bandwidth (Q4_K residual design): {verify_bpw:.1f} bpw  (Q4_K_M + Q4_K residual)")
    print(f"  Speedup draft/verify vs verify-only: {verify_bpw / q4k_bpw:.2f}× (theory, FFN tensors only)")

    # ── Write output GGUF ─────────────────────────────────────────────────────
    if args.out and args.out_layout == "q4_k_res":
        if len(packed_by_name) == 0:
            print("ERROR: no Q4_K tensors were packed; check tensor list / --limit", file=sys.stderr)
            sys.exit(1)
        arch = reader_architecture(reader_q4km)
        print(f"\nWriting Q4_K_RES GGUF → {args.out} (arch={arch!r}, {len(packed_by_name)} tensors repacked) ...")
        write_gguf_q4_k_res(args.out, reader_q4km, packed_by_name, arch)
        print("Done.")
    elif args.out:
        print(f"\nOutput GGUF: only --out-layout q4_k_res is implemented in this script.")
        print(f"  For int4 block_q4km_d4 GGUF writing, use a future exporter or extend this script.")
        if do_int4:
            print(f"  (Validated int4 delta in memory: {len(delta_store)} tensors.)")

if __name__ == "__main__":
    main()
