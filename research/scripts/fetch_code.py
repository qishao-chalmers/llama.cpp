#!/usr/bin/env python3
"""
fetch_code.py — Download code corpora for coding-agent perplexity evaluation.

Two output formats (--output-format):
  text   — Flat concatenation of code, used with run_sweep.py in default (flat) mode.
           Same sliding-window method as WikiText experiments.
  jsonl  — Structured per-example file: each line is
           {"prompt": "<repo files truncated to budget>",
            "completion": "<target file, up to --completion-tokens>",
            "repo": "...", "file": "..."}
           Used with run_sweep.py --corpus-mode structured.

Supported datasets:
  codesearch  — Python functions from code_search_net (open, no auth).
                Produces flat .txt only (no per-example repo structure).
  longcode    — LongCodeArena small_context split (JetBrains Research).
                144 real GitHub repos, median ~22K token context per example.
                Produces both .txt and .jsonl.
  all         — Both datasets.

Requirements:
    pip install datasets

Usage:
    # Flat corpus for sliding-window perplexity (Approach 1):
    python3 research/scripts/fetch_code.py --dataset codesearch
    python3 research/scripts/fetch_code.py --dataset longcode --output-format text

    # Structured JSONL for per-function evaluation (Approach 2):
    python3 research/scripts/fetch_code.py --dataset longcode --output-format jsonl
    python3 research/scripts/fetch_code.py --dataset longcode --output-format jsonl \\
        --context-tokens 15872 --completion-tokens 512

    # Both formats at once:
    python3 research/scripts/fetch_code.py --dataset longcode --output-format both \\
        --crops 4096 16384 32768 full

Output files in research/data/:
    code_codesearch.txt
    code_longcode.txt            (flat)
    code_longcode.jsonl          (structured)
    code_codesearch_<N>tok.txt   (if --crops specified)
    code_longcode_<N>tok.txt
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))

CHARS_PER_TOKEN = 2.8   # code averages ~2.8 chars/token (short tokens: brackets, indent, punctuation)
MIN_CHARS       = 100
MAX_CHARS       = 100_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(name, config, split, streaming=False):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not found.  pip install datasets")
        sys.exit(1)
    label = f"{name} ({config}, {split})"
    if streaming:
        print(f"Streaming {label}...", flush=True)
    else:
        print(f"Downloading {label}...", flush=True)
    return load_dataset(name, config, split=split,
                        streaming=streaming, trust_remote_code=False)


def clean_code(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def file_separator(name: str) -> str:
    return f"\n\n# ── {name} ──\n\n"


def crop_by_tokens(text: str, n_tokens: int) -> str:
    n_chars = int(n_tokens * CHARS_PER_TOKEN)
    cropped = text[:n_chars]
    sep = cropped.rfind("\n\n# ── ")
    if sep > n_chars // 2:
        cropped = cropped[:sep]
    return cropped.strip()


# ── Flat corpus builders ──────────────────────────────────────────────────────

def fetch_codesearch_flat(max_tokens: int = 200_000) -> str:
    """code_search_net Python — function-level, open access."""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    ds = _load("code_search_net", "python", "train", streaming=True)
    chunks, total, skipped = [], 0, 0
    for ex in ds:
        content = ex.get("whole_func_string", "")
        if len(content) < MIN_CHARS or len(content) > MAX_CHARS:
            skipped += 1
            continue
        chunks.append(file_separator(ex.get("repository_name", "unknown")) + content)
        total += len(content)
        if total >= max_chars:
            break
        if len(chunks) % 500 == 0:
            print(f"  codesearch: {len(chunks)} funcs, "
                  f"~{total // int(CHARS_PER_TOKEN):,} tokens...", flush=True)
    print(f"  codesearch: {len(chunks)} funcs collected "
          f"({skipped} skipped), ~{total // int(CHARS_PER_TOKEN):,} tokens", flush=True)
    return clean_code("\n".join(chunks))


def fetch_longcode_flat(max_tokens: int = 200_000) -> str:
    """LongCodeArena small_context — flatten all repo files + completion files."""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    ds = _load("JetBrains-Research/lca-project-level-code-completion",
               "small_context", "test")
    chunks, total = [], 0
    for ex in ds:
        repo = ex["repo"]
        # Add each file in the repo snapshot
        filenames = ex["repo_snapshot"]["filename"]
        contents  = ex["repo_snapshot"]["content"]
        for fname, content in zip(filenames, contents):
            if not isinstance(content, str) or len(content) < MIN_CHARS:
                continue
            chunks.append(file_separator(f"{repo}/{fname}") + content)
            total += len(content)
        # Also add the completion file
        cf_name    = ex["completion_file"]["filename"]
        cf_content = ex["completion_file"]["content"]
        chunks.append(file_separator(f"{repo}/{cf_name}") + cf_content)
        total += len(cf_content)
        if total >= max_chars:
            break
    print(f"  longcode flat: {len(chunks)} files, "
          f"~{total // int(CHARS_PER_TOKEN):,} tokens", flush=True)
    return clean_code("\n".join(chunks))


# ── Structured JSONL builder ──────────────────────────────────────────────────

def fetch_longcode_jsonl(context_tokens: int = 3584,
                         completion_tokens: int = 512) -> list:
    """
    LongCodeArena small_context → list of dicts:
      {"prompt": str, "completion": str, "repo": str, "file": str,
       "n_prompt_tokens": int, "n_completion_tokens": int}

    Prompt  = repo files packed in order until context_tokens budget is full.
    Completion = first completion_tokens tokens of the completion file.

    Files are sorted: Python files first (most relevant for a coder), then others,
    each truncated at MAX_CHARS to avoid one giant file consuming the whole budget.
    """
    context_chars    = int(context_tokens    * CHARS_PER_TOKEN)
    completion_chars = int(completion_tokens * CHARS_PER_TOKEN)

    ds = _load("JetBrains-Research/lca-project-level-code-completion",
               "small_context", "test")
    records = []
    for ex in ds:
        repo      = ex["repo"]
        filenames = ex["repo_snapshot"]["filename"]
        contents  = ex["repo_snapshot"]["content"]

        # Sort: .py files first, then by descending size (most content first)
        file_pairs = [(fn, ct) for fn, ct in zip(filenames, contents)
                      if isinstance(ct, str) and len(ct) >= MIN_CHARS]
        file_pairs.sort(key=lambda x: (0 if x[0].endswith(".py") else 1,
                                       -len(x[1])))

        # Pack files into prompt up to context budget
        prompt_parts = []
        used = 0
        for fname, content in file_pairs:
            content = content[:MAX_CHARS]   # cap single file
            if used + len(content) > context_chars:
                # Partial fill: take what fits
                remaining = context_chars - used
                if remaining > MIN_CHARS:
                    content = content[:remaining]
                else:
                    break
            prompt_parts.append(file_separator(f"{repo}/{fname}") + content)
            used += len(content)
            if used >= context_chars:
                break

        prompt = clean_code("\n".join(prompt_parts))

        # Completion = first completion_chars of the completion file
        cf_name    = ex["completion_file"]["filename"]
        cf_content = ex["completion_file"]["content"][:completion_chars]
        completion = clean_code(cf_content)

        if not prompt or not completion:
            continue

        records.append({
            "prompt":              prompt,
            "completion":          completion,
            "repo":                repo,
            "file":                cf_name,
            "n_prompt_tokens":     len(prompt) // int(CHARS_PER_TOKEN),
            "n_completion_tokens": len(completion) // int(CHARS_PER_TOKEN),
        })

    print(f"  longcode jsonl: {len(records)} examples | "
          f"prompt budget={context_tokens:,} tok | "
          f"completion cap={completion_tokens:,} tok", flush=True)
    if records:
        p_lens = [r["n_prompt_tokens"] for r in records]
        c_lens = [r["n_completion_tokens"] for r in records]
        print(f"  actual prompt   median={sorted(p_lens)[len(p_lens)//2]:,} tok  "
              f"min={min(p_lens):,}  max={max(p_lens):,}", flush=True)
        print(f"  actual completion median={sorted(c_lens)[len(c_lens)//2]:,} tok  "
              f"min={min(c_lens):,}  max={max(c_lens):,}", flush=True)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download code corpora for coding-agent perplexity evaluation.")
    parser.add_argument("--out-dir", default=DATA_DIR,
                        help="Output directory. Default: research/data/")
    parser.add_argument("--dataset", nargs="+", default=["longcode"],
                        choices=["codesearch", "longcode", "all"],
                        help="Dataset(s) to download. Default: longcode")
    parser.add_argument("--output-format", default="both",
                        choices=["text", "jsonl", "both"],
                        help="text=flat .txt, jsonl=structured, both=save both. "
                             "Default: both  (codesearch only supports text)")
    parser.add_argument("--tokens", type=int, default=200_000,
                        help="Token budget for flat .txt output. Default: 200000")
    parser.add_argument("--context-tokens", type=int, default=3584,
                        help="Prompt token budget per JSONL example. Default: 3584 "
                             "(fits n_ctx=4096 with 512 completion tokens)")
    parser.add_argument("--completion-tokens", type=int, default=512,
                        help="Max completion tokens per JSONL example. Default: 512")
    parser.add_argument("--crops", nargs="+", default=["full"],
                        help="Flat .txt crops (e.g. --crops 4096 16384 32768 full)")
    args = parser.parse_args()

    requested = ["codesearch", "longcode"] if "all" in args.dataset else args.dataset
    want_text = args.output_format in ("text", "both")
    want_jsonl = args.output_format in ("jsonl", "both")
    os.makedirs(args.out_dir, exist_ok=True)

    for ds_name in requested:
        # ── Flat text ──
        if want_text:
            if ds_name == "codesearch":
                text = fetch_codesearch_flat(max_tokens=args.tokens)
            else:
                text = fetch_longcode_flat(max_tokens=args.tokens)

            out_name = f"code_{ds_name}.txt"
            out_path = os.path.join(args.out_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved: {out_path}", flush=True)

            for crop_spec in args.crops:
                if crop_spec.lower() == "full":
                    continue
                try:
                    n = int(crop_spec)
                except ValueError:
                    print(f"WARNING: unrecognised crop '{crop_spec}', skipping.")
                    continue
                stem = out_name.replace(".txt", "")
                crop_path = os.path.join(args.out_dir, f"{stem}_{n}tok.txt")
                with open(crop_path, "w", encoding="utf-8") as f:
                    f.write(crop_by_tokens(text, n))
                print(f"Saved: {crop_path}  (~{n} tokens)", flush=True)

        # ── Structured JSONL (longcode only) ──
        if want_jsonl and ds_name == "longcode":
            records = fetch_longcode_jsonl(
                context_tokens=args.context_tokens,
                completion_tokens=args.completion_tokens,
            )
            out_path = os.path.join(args.out_dir, "code_longcode.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Saved: {out_path}  ({len(records)} examples)", flush=True)

        if want_jsonl and ds_name == "codesearch":
            print("  codesearch: jsonl format not supported "
                  "(no per-repo structure). Use --output-format text.", flush=True)

    print("\nDone.")
    print("\nApproach 1 — flat sliding window:")
    print("  python3 research/scripts/run_sweep.py "
          "models/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf \\")
    print(f"      research/data/code_longcode.txt \\")
    print("      --n-ctx 16384 --n-prompt 15872 --n-chunks 5 --prefill-batch \\")
    print("      --quants fp16 int8_ch int4_ch \\")
    print("      --out research/results/results_coder_flat_ctx16384.json")
    print("\nApproach 2 — structured per-repo evaluation:")
    print("  python3 research/scripts/run_sweep.py "
          "models/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf \\")
    print(f"      research/data/code_longcode.jsonl \\")
    print("      --corpus-mode structured \\")
    print("      --quants fp16 int8_ch int4_ch \\")
    print("      --out research/results/results_coder_structured.json")


if __name__ == "__main__":
    main()
