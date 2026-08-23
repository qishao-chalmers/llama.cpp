#!/usr/bin/env python3
"""Plot KV profile JSON from run_sweep.py --profile-kv-out.

Usage:
  python3 plot_kv_profile.py results_kv_profile.json -o profile.png

Requires matplotlib.
"""

import argparse
import json
import sys

SCRIPT_DIR = __import__("os").path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)


def main():
    ap = argparse.ArgumentParser(description="Visualize KV timing profile JSON")
    ap.add_argument("profile_json", help="JSON from run_sweep --profile-kv-out")
    ap.add_argument("-o", "--out", default="kv_profile.png", help="Output PNG path")
    ap.add_argument("--quant", default=None,
                    help="Single quant name to plot (default: first non-fp16 key)")
    args = ap.parse_args()

    with open(args.profile_json, encoding="utf-8") as f:
        data = json.load(f)

    if args.quant:
        key = args.quant
        if key not in data:
            sys.exit(f"Unknown quant '{key}'. Keys: {list(data.keys())}")
        rows = [data[key]]
        title = key
    else:
        keys = [k for k in data if k != "fp16" and not k.startswith("_")]
        if not keys:
            keys = list(data.keys())
        key = keys[0]
        rows = [data[key]]
        title = key

    row = rows[0]
    decode = row.get("decode_s", 0.0)
    gpu_kv = row.get("gpu_kv_s", 0.0)
    parts = [
        ("decode (llama)", decode),
        ("kv_get", row.get("kv_get_s", 0)),
        ("kv_parse", row.get("kv_parse_s", 0)),
        ("kv_quant", row.get("kv_quant_s", 0)),
        ("kv_pack", row.get("kv_pack_s", 0)),
        ("kv_set", row.get("kv_set_s", 0)),
        ("kv_gpu (CuPy)", gpu_kv),
    ]
    labels = [p[0] for p in parts]
    vals = [p[1] for p in parts]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: stacked bar — decode vs CPU KV vs GPU KV
    kv_cpu = sum(vals[1:6])
    ax = axes[0]
    ax.barh([0], [decode], left=0, height=0.5, label="decode", color="#2ecc71")
    left = decode
    if kv_cpu > 0:
        ax.barh([0], [kv_cpu], left=left, height=0.5, label="KV CPU", color="#e74c3c")
        left += kv_cpu
    if gpu_kv > 0:
        ax.barh([0], [gpu_kv], left=left, height=0.5, label="KV GPU (CuPy)", color="#9b59b6")
    ax.set_yticks([0])
    ax.set_yticklabels([title])
    ax.set_xlabel("seconds")
    ax.set_title("Decode vs KV quant (CPU serialize path vs GPU in-place)")
    ax.legend(loc="lower right")

    # Right: pie of KV sub-components (CPU slices + gpu bucket)
    ax = axes[1]
    kv_labels = labels[1:]
    kv_vals = vals[1:]
    if sum(kv_vals) <= 0:
        ax.text(0.5, 0.5, "No KV hook time", ha="center", va="center")
    else:
        ax.pie(kv_vals, labels=kv_labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("KV breakdown (CPU phases + GPU)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
