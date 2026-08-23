#!/usr/bin/env python3
"""build_selective_precision.py — Re-quantise selected Q8_0 groups to N-bit-in-Q8 in a GGUF.

Ablation tool: output remains standard Q8_0 tensors (same size, same inference path).
See research/context/selective_weight_precision.md for design.

Usage:
    python3 research/scripts/build_selective_precision.py --input models/Qwen3-8B-Q8_0.gguf \\
        --output models/out.gguf --attn-bits 4 --ffn-bits 4 --essential-bits 6

    python3 research/scripts/build_selective_precision.py --input model.gguf --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))
sys.path.insert(0, str(_REPO_ROOT / "research" / "scripts"))

from gguf.quants import quant_shape_to_byte_shape  # noqa: E402
from quant import _uniform_quant_asym  # noqa: E402

# ── GGML type IDs (same as build_q8_recovered.py) ─────────────────────────────
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

Q8_BLOCK_BYTES = 34
Q8_BLOCK_ELEMS = 32

_SKIP_KV = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def classify_tensor(name: str) -> str:
    """Return 'attn' | 'ffn' | 'essential' | 'skip'."""
    m = re.match(r"^blk\.(\d+)\.(.*?)\.weight$", name)
    if m is None:
        return "skip"
    role = m.group(2)
    if role in ("attn_q", "attn_k", "attn_output"):
        return "attn"
    if role in ("ffn_gate", "ffn_up"):
        return "ffn"
    if role in ("attn_v", "ffn_down"):
        return "essential"
    return "skip"


def layer_index(name: str) -> int | None:
    m = re.match(r"^blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def compute_n_layers(tensor_names: list[str]) -> int:
    indices = [layer_index(n) for n in tensor_names]
    valid = [i for i in indices if i is not None]
    if not valid:
        return 0
    return max(valid) + 1


def should_skip_layer(layer: int, n_layers: int, first: int, last: int) -> bool:
    return layer < first or layer >= (n_layers - last)


def bits_for_group(group: str, args: argparse.Namespace) -> int:
    if group == "attn":
        return int(args.attn_bits)
    if group == "ffn":
        return int(args.ffn_bits)
    if group == "essential":
        return int(args.essential_bits)
    return 8


def cli_flag_for_group(group: str) -> str:
    """Which CLI arg controls bits for this tensor group (for log messages)."""
    if group == "attn":
        return "--attn-bits"
    if group == "ffn":
        return "--ffn-bits"
    if group == "essential":
        return "--essential-bits"
    return ""


def requantise_q8_to_nbit(q8_flat: np.ndarray, bits: int) -> np.ndarray:
    """Re-quantise Q8_0 flat bytes to N-bit asymmetric precision, re-encoded as Q8_0."""
    if q8_flat.size % Q8_BLOCK_BYTES != 0:
        raise ValueError(f"Q8_0 data length {q8_flat.size} not multiple of {Q8_BLOCK_BYTES}")

    n_blocks = q8_flat.size // Q8_BLOCK_BYTES
    blocks = q8_flat.reshape(n_blocks, Q8_BLOCK_BYTES)

    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    w = qs * d

    w_q = _uniform_quant_asym(w, bits=bits, axis=1).astype(np.float32)

    abs_max = np.abs(w_q).max(axis=1, keepdims=True)
    d_new = (abs_max / 127.0).astype(np.float16)
    d_new_f32 = d_new.astype(np.float32)
    qs_new = np.clip(
        np.round(w_q / np.where(d_new_f32 == 0.0, 1.0, d_new_f32)),
        -127,
        127,
    ).astype(np.int8)

    out = np.zeros((n_blocks, Q8_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d_new.view(np.uint8)
    out[:, 2:] = qs_new.view(np.uint8)
    return out.reshape(-1)


def _asym_bin_saturation_rate(w: np.ndarray, bits: int) -> float:
    """Fraction of elems with q==0 or q==n_levels after per-row (block) asymmetric quant."""
    n_levels = (1 << bits) - 1
    xmin = w.min(axis=1, keepdims=True)
    xmax = w.max(axis=1, keepdims=True)
    scale = np.where(xmax == xmin, 1.0, (xmax - xmin) / n_levels)
    q = np.round((w - xmin) / scale).clip(0, n_levels)
    sat = (q == 0) | (q == n_levels)
    return float(np.mean(sat))


def _dequant_q8_blocks(w_flat: np.ndarray) -> np.ndarray:
    """Dequant Q8_0 flat bytes → float32 (n_blocks, 32)."""
    n_blocks = w_flat.size // Q8_BLOCK_BYTES
    blocks = w_flat.reshape(n_blocks, Q8_BLOCK_BYTES)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return qs * d


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).ravel()
    return float(np.sqrt(np.mean(d * d)))


def verbose_requant_stats(w_flat: np.ndarray, bits: int) -> tuple[float, float]:
    """RMS error dequant(orig) vs dequant(requant), and saturation rate on pre-Q8 lattice."""
    w = _dequant_q8_blocks(w_flat)
    new_flat = requantise_q8_to_nbit(w_flat, bits)
    w2 = _dequant_q8_blocks(new_flat)

    sat = _asym_bin_saturation_rate(w, bits)
    return _rms(w, w2), sat


def default_output_path(inp: Path, args: argparse.Namespace) -> Path:
    stem = inp.stem
    parts = [stem, f"attn{args.attn_bits}", f"ffn{args.ffn_bits}", f"ess{args.essential_bits}"]
    if int(args.first_layers) > 0:
        parts.append(f"first{args.first_layers}")
    if int(args.last_layers) > 0:
        parts.append(f"last{args.last_layers}")
    name = "-".join(parts) + ".gguf"
    return inp.parent / name


@dataclass
class RunStats:
    requant_attn: int = 0
    requant_ffn: int = 0
    requant_essential: int = 0
    skipped: int = 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path, help="Input GGUF (typically Q8_0)")
    ap.add_argument("--output", type=Path, default=None, help="Output GGUF (default: auto from bits)")
    ap.add_argument("--attn-bits", type=int, default=8, choices=range(2, 9), help="attn_q/k/output")
    ap.add_argument("--ffn-bits", type=int, default=8, choices=range(2, 9), help="ffn_gate/up")
    ap.add_argument("--essential-bits", type=int, default=8, choices=range(2, 9), help="attn_v/ffn_down")
    ap.add_argument("--first-layers", type=int, default=0, help="Protect first N layers (full Q8_0)")
    ap.add_argument("--last-layers", type=int, default=0, help="Protect last N layers (full Q8_0)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; no output file")
    ap.add_argument("--verbose", action="store_true", help="RMS + saturation for requantised tensors")
    args = ap.parse_args()

    if int(args.first_layers) < 0 or int(args.last_layers) < 0:
        sys.exit("--first-layers and --last-layers must be >= 0")

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"Input not found: {src}")

    out_path = args.output
    if out_path is None and not args.dry_run:
        out_path = default_output_path(src, args)
    elif out_path is not None:
        out_path = out_path.expanduser().resolve()

    print(f"source  : {src}  ({src.stat().st_size / 1e9:.2f} GB)")
    if not args.dry_run and out_path is not None:
        print(f"output  : {out_path}")
    print()
    print(
        "Tensor groups:  --attn-bits  → blk.N.attn_q / attn_k / attn_output\n"
        "                --ffn-bits   → blk.N.ffn_gate / ffn_up\n"
        "                --essential-bits → blk.N.attn_v / ffn_down\n"
        f"(current: attn={args.attn_bits} ffn={args.ffn_bits} essential={args.essential_bits})\n"
    )

    reader = gguf.GGUFReader(str(src), "r")
    names = [t.name for t in reader.tensors]
    n_layers = compute_n_layers(names)
    print(f"layers  : {n_layers} total  (first={args.first_layers} last={args.last_layers} protected)")
    if args.first_layers + args.last_layers >= n_layers > 0:
        print(f"  [warn] --first-layers {args.first_layers} + --last-layers {args.last_layers}"
              f" >= {n_layers} total layers; all layers protected, nothing will be requantised",
              file=sys.stderr)
    stats = RunStats()

    def dry_run_desc(
        *,
        group: str,
        layer: int | None,
        ttype: int,
        protected: bool,
        target_bits: int,
        do_requantise: bool,
    ) -> str:
        if ttype == GGML_TYPE_F32:
            base = "F32 copy-as-is"
        elif ttype == GGML_TYPE_F16:
            base = "F16 copy-as-is"
        elif ttype == GGML_TYPE_Q8_0:
            if protected:
                lid = int(layer) if layer is not None else -1
                # By design: first/last N layers stay full Q8_0 (no N-bit requant pass).
                base = (
                    "Q8_0 unchanged (layer %d in --first-layers/--last-layers guard; full precision)"
                    % lid
                )
            elif group == "skip":
                base = "Q8_0 copy-as-is (not attn/ffn/essential projection weight)"
            elif target_bits >= 8:
                flg = cli_flag_for_group(group)
                base = (
                    f"Q8_0 unchanged ({flg}={target_bits}; "
                    f"set < 8 to requant this {group} group)"
                )
            elif do_requantise:
                nl = (1 << target_bits) - 1
                base = f"Q{target_bits}-in-Q8 (bits={target_bits}, {nl} levels/block)"
            else:
                base = "Q8_0 copy-as-is"
        else:
            base = f"type{ttype} copy-as-is"
        return base

    if args.dry_run:
        print(f"Tensor classification ({n_layers} layers total, first={args.first_layers} last={args.last_layers}):")
        for t in reader.tensors:
            ttype = int(t.tensor_type)
            group = classify_tensor(t.name)
            layer = layer_index(t.name)
            target_bits = bits_for_group(group, args)
            protected = (layer is not None) and should_skip_layer(
                layer, n_layers, int(args.first_layers), int(args.last_layers)
            )
            do_rq = (
                (not protected)
                and (group != "skip")
                and (ttype == GGML_TYPE_Q8_0)
                and (target_bits < 8)
            )
            if (
                (not protected)
                and group != "skip"
                and target_bits < 8
                and ttype != GGML_TYPE_Q8_0
            ):
                print(
                    f"  [warn] {t.name}: expected Q8_0 for selective requant, got type {ttype}; copy-as-is",
                    file=sys.stderr,
                )

            desc = dry_run_desc(
                group=group,
                layer=layer,
                ttype=ttype,
                protected=protected,
                target_bits=target_bits,
                do_requantise=do_rq,
            )
            print(f"  {t.name:<48} [{group:10s}] → {desc}")

            if do_rq:
                if group == "attn":
                    stats.requant_attn += 1
                elif group == "ffn":
                    stats.requant_ffn += 1
                else:
                    stats.requant_essential += 1
                if args.verbose:
                    q8_flat = np.asarray(t.data, dtype=np.uint8).reshape(-1)
                    rms, sat = verbose_requant_stats(q8_flat, target_bits)
                    print(
                        f"    {t.name}  [{group}  bits={target_bits}]  "
                        f"rms_err={rms:.3e}  sat={100.0 * sat:.2f}%"
                    )
            else:
                stats.skipped += 1

        total_rq = stats.requant_attn + stats.requant_ffn + stats.requant_essential
        print(
            f"\nSummary: {total_rq} tensors requantised "
            f"(attn:{stats.requant_attn} ffn:{stats.requant_ffn} essential:{stats.requant_essential}), "
            f"{stats.skipped} skipped"
        )
        return

    arch = "llama"
    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            arch = str(bytes(kv.parts[-1]), "utf-8").rstrip("\x00")
            break

    writer = gguf.GGUFWriter(str(out_path), arch)

    for kv in reader.fields.values():
        if kv.name in _SKIP_KV or kv.name.startswith("GGUF."):
            continue
        try:
            writer.add_key_value(kv.name, kv.contents(), kv.types[0])
        except Exception:
            pass

    t0 = time.time()
    for idx, t in enumerate(reader.tensors):
        ttype = int(t.tensor_type)
        group = classify_tensor(t.name)
        layer = layer_index(t.name)
        target_bits = bits_for_group(group, args)
        protected = (layer is not None) and should_skip_layer(
            layer, n_layers, int(args.first_layers), int(args.last_layers)
        )
        do_requantise = (
            (not protected)
            and (group != "skip")
            and (ttype == GGML_TYPE_Q8_0)
            and (target_bits < 8)
        )

        if (
            (not protected)
            and group != "skip"
            and target_bits < 8
            and ttype != GGML_TYPE_Q8_0
        ):
            print(
                f"  [warn] {t.name}: expected Q8_0 for selective requant, got type {ttype}; copy-as-is",
                file=sys.stderr,
            )

        desc = dry_run_desc(
            group=group,
            layer=layer,
            ttype=ttype,
            protected=protected,
            target_bits=target_bits,
            do_requantise=do_requantise,
        )
        elapsed = time.time() - t0
        print(f"  [{idx+1:3d}/{len(reader.tensors)}] {t.name:<46} {desc}  ({elapsed:.1f}s)", flush=True)

        if do_requantise:
            if group == "attn":
                stats.requant_attn += 1
            elif group == "ffn":
                stats.requant_ffn += 1
            else:
                stats.requant_essential += 1

            q8_flat = np.asarray(t.data, dtype=np.uint8).reshape(-1)
            new_flat = requantise_q8_to_nbit(q8_flat, target_bits)
            if args.verbose:
                w_orig = _dequant_q8_blocks(q8_flat)
                w_new  = _dequant_q8_blocks(new_flat)
                rms    = _rms(w_orig, w_new)
                sat    = _asym_bin_saturation_rate(w_orig, target_bits)
                print(
                    f"      [{group}  bits={target_bits}]  rms_err={rms:.3e}  sat={100.0 * sat:.2f}%"
                )
            ggml_shape = list(t.shape)
            numpy_elem_shape = tuple(reversed(ggml_shape))
            numpy_byte_shape = quant_shape_to_byte_shape(
                numpy_elem_shape, gguf.GGMLQuantizationType.Q8_0
            )
            raw_np = new_flat.reshape(numpy_byte_shape)
            writer.add_tensor(t.name, raw_np, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif ttype == GGML_TYPE_F32:
            stats.skipped += 1
            writer.add_tensor(t.name, t.data.view(np.float32))
        elif ttype == GGML_TYPE_F16:
            stats.skipped += 1
            writer.add_tensor(t.name, t.data.view(np.float16))
        else:
            stats.skipped += 1
            orig_type = gguf.GGMLQuantizationType(ttype)
            writer.add_tensor(t.name, t.data.view(np.uint8), raw_dtype=orig_type)

    total_rq = stats.requant_attn + stats.requant_ffn + stats.requant_essential
    print()
    print(
        f"Summary: {total_rq} tensors requantised "
        f"(attn:{stats.requant_attn} ffn:{stats.requant_ffn} essential:{stats.requant_essential}), "
        f"{stats.skipped} skipped"
    )
    print("\nwriting header + tensors ...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    if out_path is not None:
        sz = out_path.stat().st_size
        print(f"\ndone  → {out_path}  ({sz / 1e9:.2f} GB)  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
