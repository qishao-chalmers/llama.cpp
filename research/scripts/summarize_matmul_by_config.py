#!/usr/bin/env python3
"""summarize_matmul_by_config.py — Per-model matmul duration by projection role.

Reads layer-stage rows from parse_nsys_decode_kernels.py and writes one CSV
per model. First column is the projection name; remaining columns are
{quant}_npl{N}_mean_us for each weight quant × batch size.

Role assignment (layer CUDA order from the decoder structure):
  pre-FA:  k_proj, v_proj
  post-FA: o_proj, ffn_up, ffn_gate, ffn_down, q_proj
q_proj often appears after the FFN in the recorded order (attn_prep split +
CUDA schedule); it is still labeled q_proj. When only one n_ff matmul is
present (typical npl=1), it is labeled ffn_up and ffn_gate is left blank.

Usage:
    python3 research/scripts/summarize_matmul_by_config.py \\
        --in-dir research/results/nsys/decode_kernels \\
        --out-dir research/results/nsys/decode_kernels
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(
    r"^(?P<model>qwen3-\d+b|gemma3-\d+b)_(?P<quant>Q2_K|Q3_K_M|Q8_0)"
    r"_npp(?P<npp>\d+)_ntg(?P<ntg>\d+)_npl(?P<npl>\d+)"
)

PROBLEM_RE = re.compile(r"^(?P<M>\d+)x(?P<N>\d+)/(?P<type>\S+)$")

QUANTS = ("Q2_K", "Q3_K_M", "Q8_0")
NPLS = (1, 4, 8, 16)

# Logical display order in the CSV
ROLE_ORDER = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)

# Observed dims (from npl=8 problem shapes). Used only as a sanity check.
MODEL_DIMS = {
    "qwen3-8b":   {"n_embd": 4096, "n_q": 4096, "n_kv": 1024, "n_ff": 12288},
    "qwen3-14b":  {"n_embd": 5120, "n_q": 5120, "n_kv": 1024, "n_ff": 17408},
    "gemma3-12b": {"n_embd": 3840, "n_q": 4096, "n_kv": 2048, "n_ff": 15360},
    "gemma3-27b": {"n_embd": 5376, "n_q": 4096, "n_kv": 2048, "n_ff": 21504},
}


def is_matmul(kernel: str) -> bool:
    return kernel.startswith("mul_mat") and "fixup" not in kernel


def is_fa(kernel: str) -> bool:
    return kernel.startswith("flash_attn_ext_")


def assign_roles(matmuls: list[dict]) -> dict[str, dict]:
    """Map role → matmul row from layer-ordered matmul list (FA already removed).

    Expected counts:
      7 → k, v, o, up, gate, down, q
      6 → k, v, o, up, down, q   (gate absent / fused at small npl)
    """
    n = len(matmuls)
    if n == 7:
        names = [
            "k_proj", "v_proj", "o_proj",
            "ffn_up", "ffn_gate", "ffn_down", "q_proj",
        ]
    elif n == 6:
        names = [
            "k_proj", "v_proj", "o_proj",
            "ffn_up", "ffn_down", "q_proj",
        ]
    elif n == 5:
        # rare: no separate q in structure
        names = ["k_proj", "v_proj", "o_proj", "ffn_up", "ffn_down"]
    else:
        # fallback: label by ordinal
        return {f"matmul_{i}": m for i, m in enumerate(matmuls)}

    return {name: matmuls[i] for i, name in enumerate(names)}


def parse_report(path: Path) -> tuple[dict, dict[str, dict]] | None:
    """Return (meta, {role: matmul_row}) or None."""
    text = path.read_text()
    lines = text.splitlines()
    nm = NAME_RE.search(path.name)
    if not nm:
        return None
    meta = nm.groupdict()
    meta["report"] = path.stem.replace("_decode_kernels", "")
    meta["npl"] = int(meta["npl"])

    body = "\n".join(ln for ln in lines if not ln.startswith("#"))
    layer = [
        r for r in csv.DictReader(io.StringIO(body)) if r.get("stage") == "layer"
    ]

    # Matmuls in layer order; FA is the split between attn prep and post/FFN.
    # The recorded order is already attn→FFN (see parse_nsys_decode_kernels).
    matmuls: list[dict] = []
    for r in layer:
        if not is_matmul(r["kernel"]):
            continue
        M = N = typ = ""
        pm = PROBLEM_RE.match((r.get("problem") or "").strip())
        if pm:
            M, N, typ = pm.group("M"), pm.group("N"), pm.group("type")
        matmuls.append(
            {
                "kernel": r["kernel"],
                "launch": r.get("launch") or "",
                "problem": r.get("problem") or "",
                "M": M,
                "N": N,
                "type": typ,
                "slot": int(r["slot"]),
                "count": int(r["count"]),
                "mean_us": float(r["mean_us"]),
                "sum_us": float(r["sum_us"]),
            }
        )

    roles = assign_roles(matmuls)
    return meta, roles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # model → role → (quant, npl) → mean_us
    data: dict[str, dict[str, dict[tuple[str, int], float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    # also keep problem/kernel annotations per role for a sidecar column set
    meta_info: dict[str, dict[str, dict[tuple[str, int], str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    configs: set[tuple[str, int]] = set()
    models: set[str] = set()

    for path in sorted(args.in_dir.glob("*_decode_kernels.csv")):
        parsed = parse_report(path)
        if not parsed:
            continue
        meta, roles = parsed
        model, quant, npl = meta["model"], meta["quant"], meta["npl"]
        models.add(model)
        configs.add((quant, npl))
        for role, row in roles.items():
            data[model][role][(quant, npl)] = round(row["mean_us"], 4)
            hint = row["problem"] or row["kernel"]
            meta_info[model][role][(quant, npl)] = hint

    quant_npls = sorted(
        configs, key=lambda x: (QUANTS.index(x[0]) if x[0] in QUANTS else 99, x[1])
    )
    mean_cols = [f"{q}_npl{n}_mean_us" for q, n in quant_npls]

    for model in sorted(models):
        out = args.out_dir / f"matmul_by_role_{model}.csv"
        # stable role list: known order first, then any extras
        roles_present = set(data[model].keys())
        ordered = [r for r in ROLE_ORDER if r in roles_present]
        ordered += sorted(roles_present - set(ROLE_ORDER))

        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["role", *mean_cols])
            w.writeheader()
            for role in ordered:
                row = {"role": role}
                for q, n in quant_npls:
                    key = f"{q}_npl{n}_mean_us"
                    val = data[model][role].get((q, n), "")
                    row[key] = val
                w.writerow(row)
        print(f"wrote {out}")

        # long form for this model (optional detail)
        out_long = args.out_dir / f"matmul_by_role_{model}_long.csv"
        with out_long.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["role", "quant", "npl", "mean_us", "shape_or_kernel"],
            )
            w.writeheader()
            for role in ordered:
                for q, n in quant_npls:
                    if (q, n) not in data[model][role]:
                        continue
                    w.writerow(
                        {
                            "role": role,
                            "quant": q,
                            "npl": n,
                            "mean_us": data[model][role][(q, n)],
                            "shape_or_kernel": meta_info[model][role].get((q, n), ""),
                        }
                    )
        print(f"wrote {out_long}")


if __name__ == "__main__":
    main()
