#!/usr/bin/env python3
"""merge_results.py — Merge partial per-quant result files into one combined file.

When experiments are split across SLURM jobs (e.g. aime_fp16.json + aime_int8_ch.json),
this script merges them back into a single aime.json without overwriting existing quants.

Usage:
    # Merge all aime_*.json in a model dir into aime.json
    python3 research/scripts/merge_results.py research/results/server_results/qwen3-8b aime

    # Preview without writing
    python3 research/scripts/merge_results.py research/results/server_results/qwen3-8b aime --dry-run

    # Merge all known tasks in a directory
    python3 research/scripts/merge_results.py research/results/server_results/qwen3-8b --all-tasks

    # Explicit file list
    python3 research/scripts/merge_results.py --files aime_fp16.json aime_int8_ch.json --out aime.json
"""

import argparse
import json
import os
import sys

CANONICAL_TASKS = [
    "wikitext2", "gsm8k", "longbench_qasper",
    "niah_4k", "niah_16k", "niah_32k", "niah_128k",
    "humaneval", "humaneval_pass1", "aime", "aime_2024",
]

SKIP_SUFFIXES = ("_per_example", "_bins", "_diags")

QUANT_ORDER = [
    "fp16", "int8_ch", "int4_ch", "int3_ch",
    "int3_half_1357_ch", "int3_half_1357_ch:int3_half_1357_tok", "int2_ch",
]


def quant_sort_key(q):
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)


def load_json(path):
    """Load JSON, returning None if file is empty or unreadable."""
    try:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            return None
        return json.loads(content)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] skipping {path}: {e}", file=sys.stderr)
        return None


def is_source(filename, task):
    """True if filename is a partial result for task (not a derivative file)."""
    if not filename.endswith(".json"):
        return False
    stem = filename[:-5]
    if not stem.startswith(task):
        return False
    for s in SKIP_SUFFIXES:
        if stem.endswith(s):
            return False
    return True


def merge_dicts(dicts):
    """Merge list of {quant: metrics} dicts. Later entries win on conflict."""
    merged = {}
    for d in dicts:
        for quant, metrics in d.items():
            if quant in merged:
                print(f"  [warn] duplicate quant '{quant}' — keeping later entry", file=sys.stderr)
            merged[quant] = metrics
    return merged


def sort_merged(merged):
    return dict(sorted(merged.items(), key=lambda kv: quant_sort_key(kv[0])))


def merge_task(task_dir, task, out_path, dry_run):
    all_files = sorted(os.listdir(task_dir))
    sources = [f for f in all_files if is_source(f, task)]

    if not sources:
        print(f"  {task}: no partial files found — skipping")
        return False

    # Canonical file first, then alphabetical
    sources.sort(key=lambda f: (0 if f == f"{task}.json" else 1, f))

    print(f"  {task}: found {len(sources)} file(s):")
    dicts = []
    for fname in sources:
        path = os.path.join(task_dir, fname)
        d = load_json(path)
        if d is None:
            print(f"    {fname}  →  [empty/invalid, skipped]")
            continue
        print(f"    {fname}  →  {list(d.keys())}")
        dicts.append(d)

    if not dicts:
        print(f"  {task}: all files empty — skipping")
        return False

    merged = sort_merged(merge_dicts(dicts))
    print(f"    => {len(merged)} quants: {list(merged.keys())}")
    print(f"    => output: {out_path}")

    if dry_run:
        print("    [dry-run] not written")
        return True

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"    written.")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_dir", nargs="?", help="Directory with partial result files")
    p.add_argument("task", nargs="?", help="Task stem (e.g. 'aime', 'gsm8k')")
    p.add_argument("--files", nargs="+", help="Explicit list of JSON files to merge")
    p.add_argument("--out", help="Output path (default: <task_dir>/<task>.json)")
    p.add_argument("--all-tasks", action="store_true",
                   help="Merge all known tasks in task_dir")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be merged without writing")
    args = p.parse_args()

    # Explicit file list mode
    if args.files:
        if not args.out:
            p.error("--files requires --out")
        dicts = []
        for path in args.files:
            d = load_json(path)
            if d is None:
                print(f"  [warn] {path} empty/invalid — skipped")
                continue
            print(f"  {path}  ->  {list(d.keys())}")
            dicts.append(d)
        if not dicts:
            print("No valid files to merge.")
            return
        merged = sort_merged(merge_dicts(dicts))
        print(f"  => {len(merged)} quants: {list(merged.keys())}")
        if not args.dry_run:
            with open(args.out, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"  written: {args.out}")
        else:
            print("  [dry-run] not written")
        return

    if not args.task_dir:
        p.error("task_dir is required")

    if args.all_tasks:
        for task in CANONICAL_TASKS:
            out_path = args.out or os.path.join(args.task_dir, f"{task}.json")
            merge_task(args.task_dir, task, out_path, args.dry_run)
        return

    if not args.task:
        p.error("task is required (or use --all-tasks)")

    out_path = args.out or os.path.join(args.task_dir, f"{args.task}.json")
    merge_task(args.task_dir, args.task, out_path, args.dry_run)


if __name__ == "__main__":
    main()
