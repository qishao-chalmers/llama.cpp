#!/usr/bin/env python3
"""
Compare adaptive-gen acceptance across weight precision × KV quant.

Runs (or reads) two sets of adaptive-gen experiments:
  Full model  (--model only):         kv in {fp16(*), int2_ch, int3_ch, int4_ch}
  Draft model (--model + --draft-model): kv in {fp16, int2_ch, int3_ch, int4_ch}

  (*) fp16 KV + full model = trivially 100% accepted; included only as sanity row.
      fp16 KV + draft model = weight-quant-only baseline (Q4_K_M vs Q8_0+int8_ch).

Each combination gets its own run_sweep.py invocation with a single --quants entry
so bootstrap always uses that exact KV quant (no multi-candidate picking).

Usage — run everything fresh:
  python3 research/scripts/compare_weight_kv.py \\
      --model models/Qwen3-8B-Q8_0.gguf \\
      --draft-model models/Qwen3-8B-Q8_0-Q4_K_M.gguf \\
      --corpus research/data/gsm8k_test.jsonl \\
      --out-dir research/results/weight_kv_comparison/ \\
      -- --corpus-mode structured --eval-accuracy --skip-ppl \\
         --n-ctx 4096 --max-gen-tokens 2048 --n-chunks 20 \\
         --n-gpu-layers 99 --flash-attn \\
         --verifier-quant int8_ch --bootstrap-window 64 \\
         --bootstrap-pick-mode cost \\
         --bootstrap-ms-quant-file research/context/bootstrap_ms_quant_example.json \\
         --sink-tokens 32 --recent-tokens 32

Usage — print table from existing results only (no re-runs):
  python3 research/scripts/compare_weight_kv.py \\
      --model models/Qwen3-8B-Q8_0.gguf \\
      --draft-model models/Qwen3-8B-Q8_0-Q4_K_M.gguf \\
      --corpus research/data/gsm8k_test.jsonl \\
      --out-dir research/results/weight_kv_comparison/ \\
      --results-only

Arguments before '--' configure this script.
Arguments after '--' are forwarded verbatim to run_sweep.py.
"""
import argparse
import json
import os
import subprocess
import sys


KV_QUANTS = ["fp16", "int2_ch", "int3_ch", "int4_ch"]


def result_path(out_dir: str, weights_tag: str, kv_quant: str) -> str:
    return os.path.join(out_dir, f"{weights_tag}_{kv_quant}.json")


def run_experiment(sweep_py: str, model: str, draft_model: str | None,
                   corpus: str, out_path: str, kv_quant: str,
                   extra_args: list[str], dry_run: bool) -> None:
    cmd = [
        sys.executable, sweep_py,
        model, corpus,
        "--quants", kv_quant,
        "--adaptive-gen",
        "--adaptive-window", "8",
        "--out", out_path,
    ]
    if draft_model:
        cmd += ["--draft-model", draft_model]
    cmd += extra_args

    print(f"\n{'='*70}", flush=True)
    print(f"  weights={'Q4_K_M' if draft_model else 'Q8_0'}  kv={kv_quant}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)

    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[warn] run_sweep.py exited with code {result.returncode}", flush=True)


def load_result(out_dir: str, weights_tag: str, kv_quant: str):
    """Return (ag_acc, draft_frac, task_acc) or None if missing."""
    p = result_path(out_dir, weights_tag, kv_quant)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    # Find the entry for this kv_quant (or fp16 if kv_quant is fp16)
    entry = data.get(kv_quant)
    if entry is None:
        # fp16 row might be stored under "fp16" key
        entry = next(iter(data.values()), None)
    if entry is None:
        return None

    ag = entry.get("adaptive_gen", {})
    ag_acc   = ag.get("mean_acceptance")
    df       = ag.get("mean_draft_frac")
    task_acc = entry.get("accuracy") or entry.get("f1")
    return (ag_acc, df, task_acc)


def print_table(out_dir: str, draft_model: str | None, metric: str) -> None:
    rows = []

    # Full model rows
    for kv in KV_QUANTS:
        r = load_result(out_dir, "full", kv)
        if r is None:
            rows.append(("Q8_0", kv, None, None, None))
        else:
            rows.append(("Q8_0", kv) + r)

    # Draft model rows
    if draft_model:
        for kv in KV_QUANTS:
            r = load_result(out_dir, "draft", kv)
            if r is None:
                rows.append(("Q4_K_M", kv, None, None, None))
            else:
                rows.append(("Q4_K_M", kv) + r)

    def _fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "  ---  "

    hdr = f"{'weights':<10}  {'kv_quant':<20}  {'ag_acc':>7}  {'draft_frac':>10}  {metric:>8}"
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    prev_weights = None
    for weights, kv, ag_acc, df, task in rows:
        if prev_weights is not None and weights != prev_weights:
            print(sep)
        print(f"{weights:<10}  {kv:<20}  {_fmt(ag_acc):>7}  {_fmt(df):>10}  {_fmt(task):>8}")
        prev_weights = weights
    print(sep)


def main():
    # Split on '--' to separate this script's args from run_sweep.py args
    argv = sys.argv[1:]
    try:
        split = argv.index("--")
        own_argv = argv[:split]
        extra_args = argv[split + 1:]
    except ValueError:
        own_argv = argv
        extra_args = []

    parser = argparse.ArgumentParser(
        description="Compare adaptive-gen acceptance: full vs draft model × KV quant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Full-precision model GGUF (e.g. Qwen3-8B-Q8_0.gguf)")
    parser.add_argument("--draft-model", default=None,
                        help="Lower-precision draft model GGUF (e.g. Qwen3-8B-Q8_0-Q4_K_M.gguf)")
    parser.add_argument("--corpus", required=True,
                        help="Corpus file (e.g. research/data/gsm8k_test.jsonl)")
    parser.add_argument("--out-dir", required=True,
                        help="Directory for per-combination result JSONs")
    parser.add_argument("--kv-quants", nargs="+", default=KV_QUANTS,
                        help=f"KV quants to test (default: {KV_QUANTS})")
    parser.add_argument("--metric", default="accuracy",
                        choices=["accuracy", "f1"],
                        help="Task metric label in JSON (default: accuracy)")
    parser.add_argument("--results-only", action="store_true",
                        help="Skip running experiments; just print table from existing JSONs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands but do not execute them")
    args = parser.parse_args(own_argv)

    os.makedirs(args.out_dir, exist_ok=True)
    sweep_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_sweep.py")

    if not args.results_only:
        # Full model runs
        print(f"\n{'#'*70}")
        print(f"# FULL MODEL (Q8_0 weights)")
        print(f"{'#'*70}")
        for kv in args.kv_quants:
            if kv == "fp16":
                # fp16 KV + full model = trivially 100% accepted; skip unless requested
                print(f"\n[skip] fp16 KV + full model (trivially 100% acceptance, not informative)")
                continue
            run_experiment(
                sweep_py, args.model, None,
                args.corpus,
                result_path(args.out_dir, "full", kv),
                kv, extra_args, args.dry_run)

        # Draft model runs
        if args.draft_model:
            print(f"\n{'#'*70}")
            print(f"# DRAFT MODEL (Q4_K_M weights)")
            print(f"{'#'*70}")
            for kv in args.kv_quants:
                run_experiment(
                    sweep_py, args.model, args.draft_model,
                    args.corpus,
                    result_path(args.out_dir, "draft", kv),
                    kv, extra_args, args.dry_run)

    print_table(args.out_dir, args.draft_model, args.metric)


if __name__ == "__main__":
    main()
