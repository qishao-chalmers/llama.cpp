#!/usr/bin/env python3
"""demo_benchmarks.py — Show prompt, gold answer, and model predictions per benchmark.

Reads directly from dataset JSONLs and per_example result JSONs.
No model loading required.

Usage:
    python3 research/scripts/demo_benchmarks.py
    python3 research/scripts/demo_benchmarks.py --benchmarks gsm8k niah humaneval
    python3 research/scripts/demo_benchmarks.py --results-dir research/results/qwen3-8b
    python3 research/scripts/demo_benchmarks.py --example-idx 3
    python3 research/scripts/demo_benchmarks.py --quants fp16 int2_ch
"""

import argparse
import json
import os

DATA_DIR    = "research/data"
RESULTS_DIR = "research/results"

BENCHMARKS = {
    "wikitext2": dict(
        jsonl        = None,   # flat text file, not JSONL
        txt          = "wikitext2_test.txt",
        results_file = "wikitext2.json",
        desc         = "WikiText2 — flat perplexity (teacher-forced, sliding 1024-token chunks)",
        chunk_chars  = 2000,   # chars to show as sample
    ),
    "gsm8k": dict(
        jsonl        = "gsm8k_test.jsonl",
        prompt_field = "prompt",
        gold_field   = "completion",
        per_example  = "gsm8k_per_example.json",
        desc         = "GSM8K — Math reasoning (8-shot, extract final number)",
        prompt_head  = 400,   # chars to show from start of prompt
        prompt_tail  = 200,   # chars to show from end of prompt
    ),
    "longbench": dict(
        jsonl        = "LongBench/qasper.jsonl",
        prompt_field = None,  # context + input
        gold_field   = "answers",
        per_example  = "longbench_qasper_per_example.json",
        desc         = "LongBench Qasper — Long-document QA (F1 scoring)",
        prompt_head  = 300,
        prompt_tail  = 400,
    ),
    "niah": dict(
        jsonl        = "niah_4096.jsonl",
        prompt_field = "prompt",
        gold_field   = "answers",
        per_example  = "niah_4k_per_example.json",
        desc         = "NIAH 4K — Needle-in-a-haystack retrieval",
        prompt_head  = 200,
        prompt_tail  = 300,
    ),
    "humaneval": dict(
        jsonl        = "humaneval.jsonl",
        prompt_field = "prompt",
        gold_field   = "completion",
        per_example  = "humaneval_per_example.json",
        desc         = "HumanEval — Python function completion (pass@1)",
        prompt_head  = 600,
        prompt_tail  = 0,
    ),
    "code": dict(
        jsonl        = "code_longcode.jsonl",
        prompt_field = "prompt",
        gold_field   = "completion",
        per_example  = None,   # PPL only, no per-example predictions
        results_file = "code_ppl.json",
        desc         = "LongCodeArena — Long-context code continuation (PPL only, ~4K prompt + ~600 completion tokens)",
        prompt_head  = 300,
        prompt_tail  = 300,
    ),
}


def load_example(jsonl_path, idx):
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    return None


def get_prompt_text(rec, prompt_field):
    if prompt_field and prompt_field in rec:
        return rec[prompt_field]
    if "context" in rec and "input" in rec:
        return rec["context"] + "\n\n" + rec["input"]
    return rec.get("input", "")


def get_gold(rec, gold_field):
    val = rec.get(gold_field, "")
    if isinstance(val, list):
        return val[0] if val else ""
    return val or ""


def truncate(text, head, tail):
    if tail == 0 or len(text) <= head + 20:
        return text[:head + tail] if head + tail > 0 else text
    if len(text) <= head + tail + 20:
        return text
    omitted = len(text) - head - tail
    return text[:head] + f"\n  [...{omitted:,} chars omitted...]\n" + text[-tail:]


def sep(title=""):
    w = 68
    if title:
        p = (w - len(title) - 2) // 2
        print("  " + "─" * p + f" {title} " + "─" * (w - p - len(title) - 2))
    else:
        print("  " + "─" * w)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benchmarks",   nargs="+", default=list(BENCHMARKS.keys()),
                    help=f"Benchmarks to show (default: all). Choices: {list(BENCHMARKS.keys())}")
    ap.add_argument("--results-dir",  default=None,
                    help="Directory containing per_example JSONs (default: research/results)")
    ap.add_argument("--model",        default=None,
                    help="Model subfolder under research/results/ (e.g. qwen3-8b)")
    ap.add_argument("--example-idx",  type=int, default=None,
                    help="Show a specific example index (default: show first --n-examples)")
    ap.add_argument("--n-examples",   type=int, default=3,
                    help="Number of examples to show per benchmark (default: 3)")
    ap.add_argument("--quants",       nargs="+", default=None,
                    help="Which quants to show predictions for (default: all)")
    args = ap.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join(RESULTS_DIR, args.model) if args.model else RESULTS_DIR

    for benchmark in args.benchmarks:
        if benchmark not in BENCHMARKS:
            print(f"Unknown benchmark: {benchmark}"); continue

        cfg = BENCHMARKS[benchmark]
        print()
        sep(benchmark.upper())
        print(f"  {cfg['desc']}")

        # ── WikiText2: flat text, show sample + PPL table ──────────────────────
        if cfg.get("jsonl") is None:
            txt_path = os.path.join(DATA_DIR, cfg["txt"])
            if not os.path.exists(txt_path):
                print(f"  Dataset not found: {txt_path}"); continue

            with open(txt_path) as f:
                sample = f.read(cfg["chunk_chars"])

            print(f"\n  SAMPLE TEXT  (first {cfg['chunk_chars']:,} chars of {txt_path})")
            sep()
            print(sample)
            sep()
            print("  (Evaluation: model predicts each next token — no gold answer, no generation)")
            print("  Metric: perplexity at decode lengths 512 / 1024 / 2048 tokens")

            results_path = os.path.join(results_dir, cfg["results_file"])
            if os.path.exists(results_path):
                with open(results_path) as f:
                    results = json.load(f)
                print(f"\n  PPL RESULTS  ({results_path})")
                sep()
                # results may have score_windows structure: {quant: {ppl@512: x, ...}} or flat ppl
                first = next(iter(results.values()), {})
                if isinstance(first, dict):
                    ppl_keys = [k for k in first if "ppl" in k.lower() or "@" in k]
                    header = f"  {'quant':<22}" + "".join(f"  {k:>10}" for k in ppl_keys)
                    print(header)
                    print("  " + "─" * (len(header) - 2))
                    quants_to_show = args.quants or list(results.keys())
                    for q in quants_to_show:
                        if q not in results: continue
                        row = f"  {q:<22}"
                        for k in ppl_keys:
                            v = results[q].get(k)
                            row += f"  {v:>10.3f}" if isinstance(v, float) else f"  {'—':>10}"
                        print(row)
                sep()
            else:
                print(f"\n  (No results found: {results_path})")
            continue

        # ── Load JSONL ─────────────────────────────────────────────────────────
        jsonl_path = os.path.join(DATA_DIR, cfg["jsonl"])
        if not os.path.exists(jsonl_path):
            print(f"  Dataset not found: {jsonl_path}"); continue

        # ── Load per_example predictions ───────────────────────────────────────
        ppl_only = cfg.get("per_example") is None
        per_ex   = None
        if not ppl_only:
            per_ex_path = os.path.join(results_dir, cfg["per_example"])
            if os.path.exists(per_ex_path):
                with open(per_ex_path) as f:
                    per_ex = json.load(f)
            else:
                print(f"  (No predictions found: {per_ex_path})")

        quants = args.quants or (list(per_ex.keys()) if per_ex else [])

        # Determine which indices to show
        if args.example_idx is not None:
            indices = [args.example_idx]
        else:
            indices = list(range(args.n_examples))

        for idx in indices:
            rec = load_example(jsonl_path, idx)
            if rec is None:
                print(f"  Example {idx} not found"); continue

            prompt_text = get_prompt_text(rec, cfg["prompt_field"])
            gold        = get_gold(rec, cfg["gold_field"])

            print()
            sep(f"Example #{idx}")

            print(f"\n  PROMPT  ({len(prompt_text):,} chars, ~{len(prompt_text)//4:,} tokens)")
            sep()
            print(truncate(prompt_text, cfg["prompt_head"], cfg["prompt_tail"]))
            sep()

            print(f"\n  GOLD ANSWER")
            sep()
            print(f"  {gold[:300]}")
            sep()

            if ppl_only:
                print(f"  (PPL-only benchmark — evaluation via perplexity, no per-example predictions)")
                if idx == indices[-1] and cfg.get("results_file"):
                    results_path = os.path.join(results_dir, cfg["results_file"])
                    if os.path.exists(results_path):
                        with open(results_path) as f:
                            results = json.load(f)
                        print(f"\n  PPL RESULTS  ({results_path})")
                        sep()
                        first = next(iter(results.values()), {})
                        if isinstance(first, dict):
                            ppl_keys = [k for k in first if "ppl" in k.lower() or "@" in k]
                            if ppl_keys:
                                header = f"  {'quant':<22}" + "".join(f"  {k:>10}" for k in ppl_keys)
                                print(header)
                                print("  " + "─" * (len(header) - 2))
                                quants_to_show = args.quants or list(results.keys())
                                for q in quants_to_show:
                                    if q not in results: continue
                                    row = f"  {q:<22}"
                                    for k in ppl_keys:
                                        v = results[q].get(k)
                                        row += f"  {v:>10.3f}" if isinstance(v, float) else f"  {'—':>10}"
                                    print(row)
                        sep()
                    else:
                        print(f"\n  (No results found: {results_path})")
                continue

            if per_ex is None:
                continue

            print(f"\n  MODEL PREDICTIONS")
            sep()
            for q in quants:
                if q not in per_ex:
                    continue
                exs = per_ex[q]
                if idx >= len(exs):
                    print(f"  [{q}] example {idx} not available ({len(exs)} total)")
                    continue
                ex        = exs[idx]
                score     = ex.get("score", "?")
                pred      = ex.get("pred", "")
                gen_len   = ex.get("gen_len", "?")
                trunc     = " (truncated)" if ex.get("truncated") else ""
                score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                print(f"  [{q:<20}]  score={score_str}  gen_len={gen_len}{trunc}")
                print(f"    pred: {repr(pred[:200])}")
            sep()


if __name__ == "__main__":
    main()
