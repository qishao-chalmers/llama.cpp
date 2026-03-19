#!/usr/bin/env python3
"""
kv_sweep_viz.py — Aggregate attention-analysis statistics across multiple
(dataset, offset, context_length) configurations.

For each configuration it runs visualize_kv_quant.py in a subprocess, parses
the resulting kv_viz.txt, and collects per-quant statistics.  At the end it
prints a consolidated table and writes kv_sweep_results.json.

Usage:
  python3 research/scripts/kv_sweep_viz.py \
      models/Qwen3-8B-Q8_0.gguf \
      --n-threads 8 \
      --out research/results/kv_sweep.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_SCRIPT  = os.path.join(SCRIPT_DIR, "visualize_kv_quant.py")

QUANTS = [
    "int8_ch", "int8_tok",
    "int4_ch", "int4_tok", "int4_ch:int4_tok",
    "int3_ch", "int3_tok",
    "int2_ch",
]

# K / V RMS headers appear in order: first block = K, second = V
K_QUANT_ORDER = QUANTS
V_QUANT_ORDER = QUANTS


def parse_viz_output(path):
    """
    Parse a kv_viz.txt file.  Returns a dict keyed by quant name with:
      k_rms, k_max, v_rms, v_max,
      kl    : list[float] per token
      out   : list[float] per token
      kp    : list[float] per token  (K-part of output error)
      vp    : list[float] per token  (V-part of output error)
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    result = {}
    # RMS lines
    k_rms_lines = []
    v_rms_lines = []
    in_v = False
    for line in lines:
        if "V  layer=" in line:
            in_v = True
        m = re.match(r"\s+(\S+):\s+RMS=([\d.]+)\s+max=([\d.]+)", line)
        if m:
            rec = (m.group(1), float(m.group(2)), float(m.group(3)))
            if in_v:
                v_rms_lines.append(rec)
            else:
                k_rms_lines.append(rec)

    for (qname, rms, mx) in k_rms_lines:
        result.setdefault(qname, {})["k_rms"] = rms
        result.setdefault(qname, {})["k_max"] = mx
    for (qname, rms, mx) in v_rms_lines:
        result.setdefault(qname, {})["v_rms"] = rms
        result.setdefault(qname, {})["v_max"] = mx

    # Causal attention blocks
    current = None
    for line in lines:
        m = re.match(r"\s+(\S+)\s+causal attention analysis", line)
        if m:
            current = m.group(1)
            result.setdefault(current, {}).update(kl=[], out=[], kp=[], vp=[])
            continue
        if current:
            m2 = re.match(
                r"\s+tok\s+(\d+):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line
            )
            if m2:
                result[current]["kl"].append(float(m2.group(2)))
                result[current]["out"].append(float(m2.group(3)))
                result[current]["kp"].append(float(m2.group(4)))
                result[current]["vp"].append(float(m2.group(5)))

    return result


def run_one(model, corpus, offset, n_tokens, layer, n_threads, tmp_dir, label):
    out_path = os.path.join(tmp_dir, f"viz_{label}.txt")
    cmd = [
        sys.executable, VIZ_SCRIPT, model,
        "--quants", *QUANTS,
        "--text-file", corpus,
        "--token-offset", str(offset),
        "--n-tokens", str(n_tokens),
        "--layer", str(layer),
        "--head", "0",
        "--head-dim", "128",
        "--n-threads", str(n_threads),
        "--n-gpu-layers", "0",
        "--n-show", "8",
        "--out", out_path,
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [FAILED] {label}  (returncode={proc.returncode})", flush=True)
        return None
    # Check Q was captured
    if "WARNING: Q not captured" in proc.stdout:
        print(f"  [NO Q] {label}", flush=True)
        return None
    parsed = parse_viz_output(out_path)
    print(f"  [OK] {label}  ({int(elapsed)}s, {len(next(iter(parsed.values())).get('kl',[]))} tokens analysed)",
          flush=True)
    return parsed


def aggregate(records):
    """
    records : list of dicts from parse_viz_output
    Returns a dict quant → aggregated stats.
    """
    agg = {}
    for q in QUANTS:
        vals = {k: [] for k in ["k_rms","k_max","v_rms","v_max","kl","out","kp","vp"]}
        for rec in records:
            if q not in rec:
                continue
            d = rec[q]
            for k in ["k_rms","k_max","v_rms","v_max"]:
                if k in d:
                    vals[k].append(d[k])
            for k in ["kl","out","kp","vp"]:
                vals[k].extend(d.get(k, []))
        if not vals["out"]:
            continue
        out = np.array(vals["out"])
        kp  = np.array(vals["kp"])
        vp  = np.array(vals["vp"])
        kl  = np.array(vals["kl"])
        early = out[out.shape[0]//8: out.shape[0]//4].mean() if len(out) > 4 else float("nan")
        late  = out[3*out.shape[0]//4:].mean()               if len(out) > 4 else float("nan")

        agg[q] = {
            "k_rms_mean": float(np.mean(vals["k_rms"])) if vals["k_rms"] else float("nan"),
            "v_rms_mean": float(np.mean(vals["v_rms"])) if vals["v_rms"] else float("nan"),
            "mean_kl":    float(kl.mean()),
            "mean_out":   float(out.mean()),
            "mean_kp":    float(kp.mean()),
            "mean_vp":    float(vp.mean()),
            "k_pct":      float(kp.mean() / out.mean() * 100) if out.mean() > 0 else 0,
            "early_out":  float(early),
            "late_out":   float(late),
            "growth":     float(late / early) if early > 0 else float("nan"),
            "n_samples":  int(len(out)),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n-threads",  type=int, default=8)
    ap.add_argument("--layer",      type=int, default=-1,
                    help="Layer index (-1 = n_layer//2)")
    ap.add_argument("--n-tokens",   type=int, default=256,
                    help="Tokens per context window (default 256)")
    ap.add_argument("--out", default="kv_sweep_results.json")
    args = ap.parse_args()

    # Dataset configs: (label, path, offsets)
    data_root = os.path.join(SCRIPT_DIR, "../../research/data")
    configs = [
        ("wiki2",   os.path.join(data_root, "wikitext2_test.txt"),    [0, 10_000, 50_000]),
        ("wiki103", os.path.join(data_root, "wikitext103_test.txt"),   [0, 20_000, 100_000]),
        ("c4",      os.path.join(data_root, "c4_val.txt"),             [0, 30_000, 150_000]),
    ]

    # Filter to datasets that actually exist
    configs = [(lbl, p, offs) for lbl, p, offs in configs if os.path.exists(p)]
    if not configs:
        print("No corpus files found in research/data/. Aborting.")
        sys.exit(1)

    # Resolve layer index
    layer = args.layer
    if layer < 0:
        # Quick peek
        import ctypes, sys as _sys
        _sys.path.insert(0, SCRIPT_DIR)
        import llama_bindings as llama
        lib = llama.load_lib()
        lib.llama_backend_init()
        mparams = lib.llama_model_default_params()
        mparams.n_gpu_layers = 0
        model_ptr = lib.llama_model_load_from_file(args.model.encode(), mparams)
        n_layer = lib.llama_model_n_layer(model_ptr)
        lib.llama_model_free(model_ptr)
        lib.llama_backend_free()
        layer = n_layer // 2
        print(f"Auto-selected layer {layer}", flush=True)

    total_runs = sum(len(offs) for _, _, offs in configs)
    print(f"Running {total_runs} configs × {len(QUANTS)} quants | n_tokens={args.n_tokens} | layer={layer}\n")

    all_records = []
    run_log = []
    with tempfile.TemporaryDirectory() as tmp:
        run_idx = 0
        for lbl, corpus_path, offsets in configs:
            for offset in offsets:
                run_idx += 1
                tag = f"{lbl}_off{offset//1000}k"
                print(f"[{run_idx}/{total_runs}] {tag}", flush=True)
                rec = run_one(
                    model=args.model,
                    corpus=corpus_path,
                    offset=offset,
                    n_tokens=args.n_tokens,
                    layer=layer,
                    n_threads=args.n_threads,
                    tmp_dir=tmp,
                    label=tag,
                )
                if rec is not None:
                    all_records.append(rec)
                    run_log.append({"tag": tag, "offset": offset, "corpus": lbl})

    if not all_records:
        print("No successful runs. Aborting.")
        sys.exit(1)

    print(f"\nAggregating {len(all_records)} successful runs …", flush=True)
    agg = aggregate(all_records)

    # ── Print summary table ──────────────────────────────────────────────────
    print()
    hdr = f"{'quant':<22}  {'K_RMS':>7}  {'V_RMS':>7}  {'KL':>9}  {'out_err':>8}  {'K%':>5}  {'early':>7}  {'late':>7}  {'growth':>7}"
    print(hdr)
    print("─" * len(hdr))
    for q in QUANTS:
        if q not in agg:
            continue
        d = agg[q]
        print(
            f"  {q:<20}  {d['k_rms_mean']:>7.4f}  {d['v_rms_mean']:>7.4f}"
            f"  {d['mean_kl']:>9.5f}  {d['mean_out']:>8.4f}"
            f"  {d['k_pct']:>4.0f}%"
            f"  {d['early_out']:>7.4f}  {d['late_out']:>7.4f}  {d['growth']:>6.2f}x"
        )

    print()
    print("=== ch vs tok at same bit depth (K_RMS, aggregated) ===")
    for bits in [8, 4, 3]:
        ch = f"int{bits}_ch"; tok = f"int{bits}_tok"
        if ch in agg and tok in agg:
            ratio_k = agg[tok]['k_rms_mean'] / agg[ch]['k_rms_mean']
            ratio_v = agg[tok]['v_rms_mean'] / agg[ch]['v_rms_mean']
            print(f"  {bits}-bit  K: ch={agg[ch]['k_rms_mean']:.4f}  tok={agg[tok]['k_rms_mean']:.4f}  tok/ch={ratio_k:.2f}x  |  "
                  f"V: ch={agg[ch]['v_rms_mean']:.4f}  tok={agg[tok]['v_rms_mean']:.4f}  tok/ch={ratio_v:.2f}x")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"runs": run_log, "aggregated": agg}, f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
