#!/usr/bin/env python3
"""
fetch_wikitext.py — Download standard academic text corpora for perplexity evaluation.

Supported datasets (all used in quantization papers: AWQ, GPTQ, SmoothQuant, KIVI):
  wikitext2   — WikiText-2 test split (~245K tokens).  Standard small benchmark.
  wikitext103 — WikiText-103 test split (~268K tokens). Larger, same domain as WT2.
  ptb         — Penn Treebank test split (~82K tokens).  Classic NLP benchmark.
  c4          — C4 validation subset (~500K tokens).     Modern large-scale benchmark.
  all         — Download all four.

Requirements:
    pip install datasets

Usage:
    python3 research/poc/fetch_wikitext.py                      # wikitext2 only
    python3 research/poc/fetch_wikitext.py --dataset all        # all four
    python3 research/poc/fetch_wikitext.py --dataset wikitext103 c4
    python3 research/poc/fetch_wikitext.py --dataset c4 --c4-tokens 200000
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def clean_wikitext(raw: str) -> str:
    """Remove WikiText section headers, decode @-@ tokens, collapse blank lines."""
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        if re.fullmatch(r"=+\s.*\s=+", line.strip()):
            continue
        line = re.sub(r" @-@ ", "-", line)
        line = re.sub(r" @,@ ", ",", line)
        line = re.sub(r" @.@ ", ".", line)
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ptb(raw: str) -> str:
    """PTB uses <unk> tokens and has trailing spaces; just normalise whitespace."""
    text = re.sub(r" +", " ", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_c4(raw: str) -> str:
    """C4 is already clean prose; just join and normalise."""
    text = re.sub(r"\n{3,}", "\n\n", raw)
    return text.strip()


# ── Dataset downloaders ───────────────────────────────────────────────────────

def _load(name, config, split, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not found. Install with: pip install datasets")
        sys.exit(1)
    print(f"Downloading {name} ({config}, {split})...", flush=True)
    return load_dataset(name, config, split=split, trust_remote_code=False, **kwargs)


def fetch_wikitext2() -> str:
    ds = _load("wikitext", "wikitext-2-raw-v1", "test")
    raw = "\n".join(ds["text"])
    text = clean_wikitext(raw)
    print(f"  wikitext2: {len(text):,} chars  (~{len(text)//4:,} tokens est.)", flush=True)
    return text


def fetch_wikitext103() -> str:
    ds = _load("wikitext", "wikitext-103-raw-v1", "test")
    raw = "\n".join(ds["text"])
    text = clean_wikitext(raw)
    print(f"  wikitext103: {len(text):,} chars  (~{len(text)//4:,} tokens est.)", flush=True)
    return text


def fetch_ptb() -> str:
    # PTB is an LDC-licensed dataset; the Hub version uses a legacy dataset script
    # that is no longer supported in datasets >= 3.0.
    # Try anyway; if it fails, print a clear message and skip.
    try:
        ds = _load("ptb_text_only", "penn_treebank", "test")
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print(
                "  ptb: SKIPPED — Penn Treebank requires datasets < 3.0 (LDC license;\n"
                "  no parquet mirror available). Use WikiText or C4 instead.",
                flush=True,
            )
            return None
        raise
    raw = "\n".join(ds["sentence"])
    text = clean_ptb(raw)
    print(f"  ptb: {len(text):,} chars  (~{len(text)//4:,} tokens est.)", flush=True)
    return text


def fetch_c4(max_tokens: int = 500_000) -> str:
    """Download C4 validation; stop after ~max_tokens tokens (4 chars/token est.)."""
    max_chars = max_tokens * 4
    ds = _load("allenai/c4", "en", "validation", streaming=True)
    chunks = []
    total = 0
    for ex in ds:
        chunks.append(ex["text"])
        total += len(ex["text"])
        if total >= max_chars:
            break
    raw = "\n\n".join(chunks)
    text = clean_c4(raw)
    print(f"  c4: {len(text):,} chars  (~{len(text)//4:,} tokens est.)", flush=True)
    return text


# ── Cropping ──────────────────────────────────────────────────────────────────

def crop_by_tokens(text: str, n_tokens: int, chars_per_token: float = 4.0) -> str:
    n_chars = int(n_tokens * chars_per_token)
    cropped = text[:n_chars]
    last_para = cropped.rfind("\n\n")
    if last_para > n_chars // 2:
        cropped = cropped[:last_para]
    return cropped.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = {
    "wikitext2":   (fetch_wikitext2,   "wikitext2_test.txt"),
    "wikitext103": (fetch_wikitext103, "wikitext103_test.txt"),
    "ptb":         (fetch_ptb,         "ptb_test.txt"),
    "c4":          (fetch_c4,          "c4_val.txt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DATA_DIR)
    parser.add_argument("--dataset", nargs="+",
                        default=["wikitext2"],
                        choices=list(DATASETS) + ["all"],
                        help="Dataset(s) to download. Default: wikitext2")
    parser.add_argument("--crops", nargs="+", default=["full"],
                        help="Additional crops by approx token count (e.g. 512 4096). "
                             "Always saves the full file. Default: full only")
    parser.add_argument("--c4-tokens", type=int, default=500_000,
                        help="Max tokens to pull from C4 (streaming). Default: 500000")
    args = parser.parse_args()

    requested = list(DATASETS) if "all" in args.dataset else args.dataset
    os.makedirs(args.out_dir, exist_ok=True)

    for ds_name in requested:
        fetch_fn, filename = DATASETS[ds_name]

        if ds_name == "c4":
            text = fetch_fn(max_tokens=args.c4_tokens)
        else:
            text = fetch_fn()

        if text is None:
            continue  # dataset skipped (e.g. PTB unavailable)

        full_path = os.path.join(args.out_dir, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {full_path}", flush=True)

        for crop_spec in args.crops:
            if crop_spec.lower() == "full":
                continue
            try:
                n_tokens = int(crop_spec)
            except ValueError:
                print(f"WARNING: unrecognised crop '{crop_spec}', skipping.")
                continue
            stem = filename.replace(".txt", "")
            crop_path = os.path.join(args.out_dir, f"{stem}_{n_tokens}tok.txt")
            with open(crop_path, "w", encoding="utf-8") as f:
                f.write(crop_by_tokens(text, n_tokens))
            print(f"Saved: {crop_path}  (~{n_tokens} tokens)", flush=True)

    print("\nDone. Standard sweep command:")
    print("  python3 research/poc/run_sweep.py models/Qwen3-8B-Q8_0.gguf \\")
    print(f"      {os.path.join(args.out_dir, 'wikitext2_test.txt')} \\")
    print("      --n-ctx 128 --n-chunks 20 --n-threads 8 \\")
    print("      --quants fp16 int8_ch int4_ch")


if __name__ == "__main__":
    main()
