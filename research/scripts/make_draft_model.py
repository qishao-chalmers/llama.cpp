#!/usr/bin/env python3
"""
Create a lower-precision draft model from an existing GGUF file.

Usage:
    python3 research/scripts/make_draft_model.py models/Qwen3-8B-Q8_0.gguf Q4_K_M
    python3 research/scripts/make_draft_model.py models/Qwen3-8B-Q8_0.gguf Q3_K_M
    python3 research/scripts/make_draft_model.py models/Qwen3-8B-Q8_0.gguf Q2_K

Output path is auto-derived: src.replace(".gguf", f"-{quant_type}.gguf")
Skips if output already exists (unless --force).

Wraps: build_release/bin/llama-quantize --allow-requantize src out type
"""
import argparse
import os
import subprocess
import sys


def find_llama_quantize(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, "build_release", "bin", "llama-quantize"),
        os.path.join(repo_root, "build", "bin", "llama-quantize"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    sys.exit(
        f"llama-quantize not found; tried:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def derive_output_path(src: str, quant_type: str) -> str:
    if not src.endswith(".gguf"):
        sys.exit(f"Source must be a .gguf file, got: {src}")
    return src[: -len(".gguf")] + f"-{quant_type}.gguf"


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a GGUF model to a lower-precision draft model"
    )
    parser.add_argument("src", help="Source GGUF (e.g. models/Qwen3-8B-Q8_0.gguf)")
    parser.add_argument(
        "quant_type",
        help="Target quant type (e.g. Q4_K_M, Q3_K_M, Q2_K)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path (default: auto-derived from src and quant_type)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-quantize even if output already exists",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for quantization (default: llama-quantize default)",
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isfile(src):
        sys.exit(f"Source file not found: {src}")

    out = os.path.abspath(args.out) if args.out else derive_output_path(src, args.quant_type)

    if os.path.isfile(out) and not args.force:
        print(f"Output already exists, skipping: {out}")
        print("Use --force to re-quantize.")
        return 0

    # Find llama-quantize relative to this script's repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))  # scripts/ -> research/ -> repo
    quantize_bin = find_llama_quantize(repo_root)

    cmd = [
        quantize_bin,
        "--allow-requantize",
        src,
        out,
        args.quant_type,
    ]
    if args.threads is not None:
        cmd.append(str(args.threads))

    print(f"src:  {src}")
    print(f"out:  {out}")
    print(f"type: {args.quant_type}")
    print(f"cmd:  {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"llama-quantize failed with exit code {result.returncode}")

    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"\nDone: {out} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
