#!/usr/bin/env python3
"""build_q8_recovered.py — Materialise the Q4_K_RES verify-path weights as Q8_0.

A Q4_K_RES GGUF stores each weight tensor as:
    block = [base_q4_K (144 B)] + [residual_q4_K (144 B)]   per 256 elements
The adaptive-gen verify pass reconstructs:
    weight ≈ dequant(base) + dequant(residual)
which is NOT exactly Q8_0 (two quantisation losses stacked).

This script materialises that reconstruction and re-quantises to Q8_0, producing
a standard GGUF loadable by any llama.cpp build.  Running the same benchmarks on
it directly measures the quality gap: Q4_K_RES-verify vs true Q8_0.

Tensor handling:
  Q4_K_RES (type 45) → dequant base + dequant residual → sum → Q8_0
  Q8_0     (type  8) → copied as-is (already target precision)
  F32/F16            → kept at original dtype (CUDA kernels assert F32|F16)
  everything else    → copied as-is with a warning

Usage:
    python3 research/scripts/build_q8_recovered.py \\
        models/Qwen3-8B-Q8_0-Q4_K_M_q4k_res.gguf \\
        --out models/Qwen3-8B-Q8_0-Q4_K_RES_recovered.gguf

    # dry-run: print what each tensor would become
    python3 research/scripts/build_q8_recovered.py \\
        models/Qwen3-8B-Q8_0-Q4_K_M_q4k_res.gguf --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

# ── Register custom quantisation types that the stock gguf library doesn't know ──
# GGUFReader calls GGMLQuantizationType(raw_dtype) for every tensor and raises
# ValueError for unknown values.  Monkey-patching the enum before opening any
# reader prevents that crash.
from gguf.constants import GGMLQuantizationType as _GQTYPE, GGML_QUANT_SIZES as _GQS
from gguf.quants import quant_shape_to_byte_shape as _quant_shape_to_byte_shape
# Q4_K_RES stores base_q4_K (144 B) + residual_q4_K (144 B) per 256 elements → 288 B/block
for _tid, _tname in [(45, "Q4_K_RES"), (46, "Q4_K_RES_DRAFT")]:
    if _tid not in _GQTYPE._value2member_map_:
        _pseudo = int.__new__(_GQTYPE, _tid)
        _pseudo._name_  = _tname
        _pseudo._value_ = _tid
        _GQTYPE._value2member_map_[_tid]   = _pseudo
        _GQTYPE._member_map_[_tname]       = _pseudo
    # Register block geometry so GGUFReader can compute data offsets
    if _GQTYPE._value2member_map_[_tid] not in _GQS:
        _GQS[_GQTYPE._value2member_map_[_tid]] = (256, 288)  # 256 elem, 2×144 B

# ── GGML type IDs ─────────────────────────────────────────────────────────────
GGML_TYPE_F32      = 0
GGML_TYPE_F16      = 1
GGML_TYPE_Q4_K    = 12
GGML_TYPE_Q6_K    = 14
GGML_TYPE_Q8_0    = 8
GGML_TYPE_Q4_K_RES       = 45   # custom: base_q4_K + residual_q4_K
GGML_TYPE_Q4_K_RES_DRAFT = 46   # same layout, draft-mode marker

Q4_K_BLOCK_SIZE  = 144   # sizeof(block_q4_K) in bytes, for 256 elements
Q4_K_RES_BLOCK_ELEMENTS = 256
Q8_0_BLOCK_ELEMENTS     = 32
Q8_0_BLOCK_BYTES        = 34    # 2 (fp16 scale) + 32 (int8)


# ── Q4_K dequantisation (pure numpy) ─────────────────────────────────────────

def _get_scale_min_k4(j: int, q: np.ndarray):
    """Decode one (scale, min) pair from the 6-bit packed scales array.
    Mirrors get_scale_min_k4() in ggml-quants.c exactly.
    """
    if j < 4:
        sc = int(q[j])     & 63
        m  = int(q[j + 4]) & 63
    else:
        sc = (int(q[j + 4]) & 0x0F) | ((int(q[j - 4]) >> 6) << 4)
        m  = (int(q[j + 4]) >> 4)   | ((int(q[j])     >> 6) << 4)
    return sc, m


def dequant_q4_k(raw: bytes | np.ndarray, n_blocks: int) -> np.ndarray:
    """Dequantise n_blocks of block_q4_K → float32 (n_blocks × 256 elements).

    Block layout (144 bytes each):
        [0:2]   d     fp16 super-scale
        [2:4]   dmin  fp16 super-min
        [4:16]  scales  12 bytes (6-bit packed: 8 sub-scale + 8 sub-min)
        [16:144] qs   128 bytes (256 × 4-bit values, lower/upper nibbles)

    Mirrors dequantize_row_q4_K() in ggml-quants.c.
    """
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()

    result = np.empty(n_blocks * Q4_K_RES_BLOCK_ELEMENTS, dtype=np.float32)

    for i in range(n_blocks):
        blk = raw[i * Q4_K_BLOCK_SIZE : (i + 1) * Q4_K_BLOCK_SIZE]

        d    = float(np.frombuffer(blk[0:2], dtype=np.float16)[0])
        dmin = float(np.frombuffer(blk[2:4], dtype=np.float16)[0])
        scales = np.frombuffer(blk[4:16],   dtype=np.uint8)
        qs     = np.frombuffer(blk[16:144], dtype=np.uint8)  # 128 bytes

        out = result[i * 256 : (i + 1) * 256]
        q_ptr = 0
        y_ptr = 0
        is_ = 0

        for _ in range(4):   # QK_K/64 = 4 groups of 64 elements
            sc1, m1 = _get_scale_min_k4(is_ + 0, scales)
            sc2, m2 = _get_scale_min_k4(is_ + 1, scales)
            d1, m1f = d * sc1, dmin * m1
            d2, m2f = d * sc2, dmin * m2

            chunk = qs[q_ptr : q_ptr + 32]   # 32 bytes → 32 lo nibbles + 32 hi nibbles
            out[y_ptr      : y_ptr + 32] = d1 * (chunk & 0x0F).astype(np.float32) - m1f
            out[y_ptr + 32 : y_ptr + 64] = d2 * (chunk >> 4).astype(np.float32)   - m2f

            q_ptr += 32
            y_ptr += 64
            is_ += 2

    return result


# ── Q8_0 dequantisation ───────────────────────────────────────────────────────

def dequant_q8_0(raw: bytes | np.ndarray, n_blocks: int) -> np.ndarray:
    """Dequantise n_blocks of block_q8_0 → float32 (n_blocks × 32 elements)."""
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()

    result = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        blk = raw[i * 34 : (i + 1) * 34]
        d   = float(np.frombuffer(blk[0:2], dtype=np.float16)[0])
        qs  = np.frombuffer(blk[2:34], dtype=np.int8).astype(np.float32)
        result[i * 32 : (i + 1) * 32] = d * qs

    return result


# ── Q8_0 quantisation ─────────────────────────────────────────────────────────

def quant_q8_0(data: np.ndarray) -> bytes:
    """Quantise float32 array → Q8_0 bytes (32 elements/block, 34 bytes/block)."""
    assert data.ndim == 1 and len(data) % 32 == 0, \
        f"data must be flat and divisible by 32, got shape {data.shape}"

    n_blocks = len(data) // 32
    out = bytearray(n_blocks * 34)

    for i in range(n_blocks):
        blk  = data[i * 32 : (i + 1) * 32]
        amax = float(np.max(np.abs(blk)))
        d    = amax / 127.0 if amax > 0.0 else 0.0
        out[i * 34 : i * 34 + 2] = np.array([d], dtype=np.float16).tobytes()
        if d > 0.0:
            qs = np.clip(np.round(blk / d), -127, 127).astype(np.int8)
            out[i * 34 + 2 : i * 34 + 34] = qs.tobytes()

    return bytes(out)


# ── Per-tensor conversion ──────────────────────────────────────────────────────

def convert_tensor(t, dry_run: bool = False):
    """Return (action_str, raw_q8_0_bytes | None, shape) for a GGUF tensor."""
    ttype = int(t.tensor_type)
    shape = list(t.shape)
    n_elem = int(np.prod(shape))

    if ttype in (GGML_TYPE_Q4_K_RES, GGML_TYPE_Q4_K_RES_DRAFT):
        n_blocks = n_elem // Q4_K_RES_BLOCK_ELEMENTS
        action = f"Q4_K_RES → Q8_0  ({n_elem:,} elem, {n_blocks} blocks)"
        if dry_run:
            return action, None, shape
        raw = t.data.tobytes()
        base_bytes     = b"".join(raw[i * 288       : i * 288 + 144] for i in range(n_blocks))
        residual_bytes = b"".join(raw[i * 288 + 144 : i * 288 + 288] for i in range(n_blocks))
        base_f    = dequant_q4_k(base_bytes,     n_blocks)
        residual_f = dequant_q4_k(residual_bytes, n_blocks)
        recovered  = base_f + residual_f
        return action, quant_q8_0(recovered), shape

    elif ttype == GGML_TYPE_Q8_0:
        action = "Q8_0 → Q8_0  (copy)"
        if dry_run:
            return action, None, shape
        return action, None, shape  # use t.data directly (correct numpy byte shape)

    elif ttype == GGML_TYPE_F32:
        # Keep F32 as-is: CUDA kernels (RMS-norm, binbcast, etc.) assert F32|F16
        action = "F32  → F32  (keep)"
        return action, None, shape

    elif ttype == GGML_TYPE_F16:
        action = "F16  → F16  (keep)"
        return action, None, shape

    else:
        type_name = getattr(gguf.GGMLQuantizationType, str(ttype), f"type{ttype}")
        action = f"{type_name} → copy-as-is  (unsupported, kept at original dtype)"
        print(f"  [warn] {t.name}: {action}", file=sys.stderr)
        # Return None so caller writes tensor at original dtype via separate path
        return action, None, shape


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Q4_K_RES GGUF source file")
    ap.add_argument("--out", help="Output GGUF path (default: src stem + _q8_recovered.gguf)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print per-tensor actions without writing output")
    args = ap.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        sys.exit(f"Source not found: {src_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        # Strip _q4k_res suffix if present, then append _recovered
        # e.g. Qwen3-8B-Q8_0-Q4_K_M_q4k_res.gguf → Qwen3-8B-Q8_0-Q4_K_M_recovered.gguf
        stem = src_path.stem.replace("_q4k_res", "")
        out_path = src_path.parent / (stem + "_recovered.gguf")

    print(f"source  : {src_path}  ({src_path.stat().st_size / 1e9:.2f} GB)")
    if not args.dry_run:
        print(f"output  : {out_path}")
    print()

    reader = gguf.GGUFReader(str(src_path), "r")

    # ── count Q4_K_RES tensors ────────────────────────────────────────────────
    n_q4kres = sum(1 for t in reader.tensors
                   if int(t.tensor_type) in (GGML_TYPE_Q4_K_RES, GGML_TYPE_Q4_K_RES_DRAFT))
    print(f"tensors : {len(reader.tensors)} total, {n_q4kres} Q4_K_RES to reconstruct")
    print()

    if args.dry_run:
        print(f"{'tensor':<50} {'action'}")
        print("-" * 90)
        for t in reader.tensors:
            action, _, _ = convert_tensor(t, dry_run=True)
            print(f"  {t.name:<48} {action}")
        return

    # ── write output GGUF ─────────────────────────────────────────────────────
    # Infer architecture name from source GGUF fields
    arch = "llama"
    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            import struct
            parts = kv.parts
            arch = str(bytes(parts[-1]), "utf-8").rstrip("\x00")
            break

    writer = gguf.GGUFWriter(str(out_path), arch)

    # ── Copy all metadata key-value pairs ────────────────────────────────────
    # Use field.contents() — the reader's own accessor — instead of manually
    # interpreting field.parts/field.data.  For ARRAY of scalar types the reader
    # stores each element as a separate 1-element part (one per array entry), so
    # data_parts[0].tolist() only returns the *first* element.  contents() does
    # the right thing for every type:
    #   scalar      → single Python value
    #   STRING      → Python str
    #   ARRAY[str]  → list of str
    #   ARRAY[num]  → list of Python scalars (all elements concatenated)
    # GGUFReader exposes GGUF.* pseudo-fields (version, tensor_count, kv_count)
    # that represent the file header — GGUFWriter writes those automatically.
    # Skip them and general.architecture (set via arch arg above).
    _SKIP_KV = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for kv in reader.fields.values():
        if kv.name in _SKIP_KV or kv.name.startswith("GGUF."):
            continue
        try:
            val = kv.contents()
            writer.add_key_value(kv.name, val, kv.types[0])
        except Exception:
            pass   # skip fields we can't reproduce

    # Convert tensors
    t0 = time.time()
    for idx, t in enumerate(reader.tensors):
        action, raw_bytes, shape = convert_tensor(t)
        elapsed = time.time() - t0
        print(f"  [{idx+1:3d}/{len(reader.tensors)}] {t.name:<50} {action}  ({elapsed:.1f}s)",
              flush=True)

        if raw_bytes is None:
            ttype = int(t.tensor_type)
            if ttype == GGML_TYPE_F32:
                # F32 array: pass as-is with float32 dtype — writer sets ggml type F32
                writer.add_tensor(t.name, t.data.view(np.float32))
            elif ttype == GGML_TYPE_F16:
                # F16 array: pass as-is with float16 dtype
                writer.add_tensor(t.name, t.data.view(np.float16))
            else:
                # Q8_0/Q6_K/Q4_K copy — t.data already has numpy byte shape;
                # pass as uint8 so add_tensor derives element shape from byte shape.
                orig_type = gguf.GGMLQuantizationType(ttype)
                writer.add_tensor(t.name, t.data.view(np.uint8), raw_dtype=orig_type)
            continue

        # New Q8_0 bytes (flat).  Reshape to numpy byte shape before handing to
        # the writer; otherwise add_tensor would treat ggml dim-0 as the byte axis
        # and quant_shape_from_byte_shape would compute a wrong element shape.
        #
        # ggml stores shape as [ne0, ne1, ...] where ne0 is the innermost/quantized
        # dim.  The writer expects numpy order (ne_last, ..., ne0) and reverses it
        # when writing the GGUF header.  t.data already uses numpy order internally.
        ggml_shape = list(t.shape)                         # e.g. [ne0, ne1]
        numpy_elem_shape = tuple(reversed(ggml_shape))    # (ne1, ne0)
        numpy_byte_shape = _quant_shape_to_byte_shape(
            numpy_elem_shape, gguf.GGMLQuantizationType.Q8_0)  # (ne1, ne0//32*34)
        raw_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(numpy_byte_shape)
        writer.add_tensor(t.name, raw_np, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    print()
    print("writing header + tensors ...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out_size = out_path.stat().st_size
    print(f"\ndone  → {out_path}  ({out_size / 1e9:.2f} GB)  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
