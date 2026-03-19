#!/usr/bin/env python3
"""
fetch_longbench.py — Download LongBench tasks and format as JSONL for run_sweep structured mode.

LongBench (THUDM/LongBench) is a long-context benchmark. Supported tasks:

  Single-doc QA  : qasper (~4K ctx), multifieldqa_en (~4K), narrativeqa (~18K)
  Multi-doc QA   : hotpotqa (~9K), 2wikimqa (~5K), musique (~11K)
  Code completion: lcc (~1-4K), repobench-p (~4K)

Why useful for KV compression research:
  - qasper/hotpotqa: moderate context, short answer → easy accuracy eval
  - narrativeqa: very long (~18K) — stresses KV at large prefill sizes
  - lcc/repobench-p: code; sensitive to quantization noise

Evaluation:
  - PPL/KL: standard, works for all tasks.
  - Accuracy: use substring match — gold answer must appear in generated text.
    Run with: --eval-accuracy --answer-regex '(.+)' and compare via contains check.
    (F1 scoring is standard for LongBench but requires post-processing outside run_sweep.)

Each output record:
  {"prompt":     "<context + question/instruction>",
   "completion": "<first gold answer>",
   "dataset":    "longbench/<task>",
   "id":         <int>,
   "answers":    [<all gold answers>],
   "length":     <approximate context token count>}

Usage:
  python3 research/scripts/fetch_longbench.py --task qasper \\
      --out research/data/longbench_qasper.jsonl

  python3 research/scripts/fetch_longbench.py --task narrativeqa \\
      --n-examples 50 --out research/data/longbench_narrativeqa.jsonl

  python3 research/scripts/fetch_longbench.py --task hotpotqa \\
      --n-examples 100 --max-ctx-chars 32000 \\
      --out research/data/longbench_hotpotqa.jsonl
"""

import argparse
import gzip
import json
import os


# ── Prompt templates (from LongBench paper) ───────────────────────────────────

TEMPLATES = {
    "qasper": (
        "Please answer the following question based on the article below.\n\n"
        "Article: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text carefully, then answer the question with a brief response.\n\n"
        "{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "narrativeqa": (
        "You are given a story and a question. Answer the question as concisely as possible, "
        "using a single phrase if possible.\n\n"
        "Story: {context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Give only the answer, no explanation.\n\n"
        "{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. "
        "Give only the answer, no explanation.\n\n"
        "{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. "
        "Give only the answer, no explanation.\n\n"
        "{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "lcc": (
        "Complete the following code. Output only the next line.\n\n"
        "{context}"
    ),
    "repobench-p": (
        "Complete the following code. Output only the next line.\n\n"
        "{context}"
    ),
}

# Raw data file names inside the THUDM/LongBench HuggingFace repo.
# LongBench stores data as gzip-compressed jsonl files.
# English tasks use _e suffix; code tasks have no suffix.
TASK_FILES = {
    "qasper":          "qasper_e.jsonl",
    "multifieldqa_en": "multifieldqa_en_e.jsonl",
    "narrativeqa":     "narrativeqa.jsonl",
    "hotpotqa":        "hotpotqa_e.jsonl",
    "2wikimqa":        "2wikimqa_e.jsonl",
    "musique":         "musique.jsonl",
    "lcc":             "lcc_e.jsonl",
    "repobench-p":     "repobench-p_e.jsonl",
}

# Default local data directory (relative to repo root)
DEFAULT_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "../../research/data/LongBench")

SUPPORTED_TASKS = sorted(TEMPLATES.keys())


def _load_records(task, local_dir=None):
    """Load task data from local LongBench directory, return list of dicts."""
    if local_dir is None:
        local_dir = os.path.normpath(DEFAULT_LOCAL_DIR)

    filename = TASK_FILES[task]
    path = os.path.join(local_dir, filename)

    # Try _e variant first, then bare name, then .gz
    candidates = [path,
                  path.replace("_e.jsonl", ".jsonl"),
                  path + ".gz"]
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    else:
        raise FileNotFoundError(
            f"Could not find {filename} in {local_dir}.\n"
            f"Tried: {candidates}\n"
            f"Pass --local-dir path/to/LongBench to specify the directory.")

    print(f"  Reading {path}...", flush=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    print(f"  Loaded {len(records)} records.", flush=True)
    return records


def build_prompt(task, rec):
    template = TEMPLATES[task]
    context  = rec.get("context", "")
    input_   = rec.get("input", "")
    return template.format(context=context, input=input_)


def get_completion(task, rec):
    """Return the primary gold completion string."""
    answers = rec.get("answers", [])
    if answers:
        return answers[0]
    # code tasks store the answer differently
    return rec.get("output", rec.get("answer", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         required=True, choices=SUPPORTED_TASKS,
                        help=f"LongBench task. Choices: {', '.join(SUPPORTED_TASKS)}")
    parser.add_argument("--out",          default=None,
                        help="Output JSONL path (default: research/data/longbench_<task>.jsonl)")
    parser.add_argument("--local-dir",    default=None,
                        help="Path to directory with decompressed LongBench .jsonl files "
                             "(default: research/data/LongBench/)")
    parser.add_argument("--n-examples",   type=int, default=0,
                        help="Max examples to export (0 = all)")
    parser.add_argument("--max-ctx-chars", type=int, default=0,
                        help="Truncate context to this many chars before building prompt "
                             "(0 = no limit). ~4 chars per token as rough guide.")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"research/data/longbench_{args.task}.jsonl"

    print(f"Loading LongBench/{args.task}...", flush=True)
    records = _load_records(args.task, local_dir=args.local_dir)
    if args.n_examples > 0:
        records = records[:args.n_examples]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prompt_lens = []
    comp_lens   = []
    skipped     = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            # Optionally truncate context
            if args.max_ctx_chars > 0 and len(rec.get("context", "")) > args.max_ctx_chars:
                rec = dict(rec)
                rec["context"] = rec["context"][:args.max_ctx_chars]

            prompt     = build_prompt(args.task, rec)
            completion = get_completion(args.task, rec)

            if not completion.strip():
                skipped += 1
                continue

            prompt_lens.append(len(prompt.split()))
            comp_lens.append(len(completion.split()))

            out = {
                "prompt":     prompt,
                "completion": completion,
                "dataset":    f"longbench/{args.task}",
                "id":         i,
                "answers":    rec.get("answers", [completion]),
                "length":     rec.get("length", 0),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(prompt_lens)
    print(f"Wrote {n} examples to {args.out}"
          + (f"  ({skipped} skipped — empty answer)" if skipped else ""))
    if n:
        avg_p = sum(prompt_lens) // n
        max_p = max(prompt_lens)
        avg_c = sum(comp_lens) // n
        print(f"  Prompt length  : avg ~{avg_p} words  (max {max_p})")
        print(f"  Completion len : avg ~{avg_c} words")
        suggested_ctx = min(131072, (max_p * 4 // 512 + 1) * 512)
        print(f"  Suggested: --n-ctx {suggested_ctx} --n-chunks {min(n, 20)}")
        print(f"  Accuracy eval: use --eval-accuracy with F1 substring check")
        if args.task in ("lcc", "repobench-p"):
            print(f"  Code task: --answer-regex '([^\\n]+)' (first generated line)")
        else:
            print(f"  QA task: --answer-regex '(.+?)(?:\\n|$)' (first generated line)")


if __name__ == "__main__":
    main()
