#!/usr/bin/env python3
"""parse_nsys_decode_kernels.py — Decode-stage CUDA kernel timings from an nsys report.

Finds decode via flash-attn anchors (one per layer per token):
  - npl=1 often uses flash_attn_ext_vec
  - larger npl often uses flash_attn_ext_f16 with a *different demangled template*
    than prefill (same shortName/shape is possible — demangled ncols differs).
Auto-picks the latest major flash_attn_ext_{vec,f16} demangled group as decode.

Reports:
  1. Structure  — one layer in attn→FFN order (mean over layers×tokens),
                  plus per-token input / output.
                  Columns: launch (CUDA grid/block) and problem (GEMV MxN/type, FA template).
  2. Flat       — all decode kernels aggregated by (name, launch, problem), sorted by total time.
  3. Throughput — decode tok/s from n_tg / decode_wall.

Usage:
    python3 research/scripts/parse_nsys_decode_kernels.py \\
        research/results/nsys/qwen3-8b_Q2_K_npp4096_ntg64_npl1_nograph.nsys-rep \\
        --keep-sqlite

    python3 research/scripts/parse_nsys_decode_kernels.py \\
        research/results/nsys/*.nsys-rep --keep-sqlite \\
        --out-dir research/results/nsys/decode_kernels

Requires: nsys (for .nsys-rep), Python stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


# Main attention kernels used as decode anchors (not combine/fixup/mask helpers).
FA_MAIN_RE = re.compile(r"^flash_attn_ext_(vec|f16|tile|wmma|mma)")
TMPL_RE = re.compile(
    r"(flash_attn_ext_\w+)<((?:\(int\)\d+|\(bool\)\d+)(?:,\s*(?:\(int\)\d+|\(bool\)\d+))*)>"
)
# mul_mat_vec_q<(ggml_type)T, (int)ncols_dst, ...>
MMVQ_RE = re.compile(
    r"mul_mat_vec_q<\(ggml_type\)(\d+),\s*\(int\)(\d+)"
)

# ggml_type ids commonly seen in demangled mmvq (subset; unknown → tN)
GGML_TYPE_NAME = {
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "IQ2_XXS", 16: "IQ2_XS", 17: "IQ3_XXS", 18: "IQ1_S", 19: "IQ4_NL",
    20: "IQ3_S", 21: "IQ2_S", 22: "IQ4_XS",
}


def fmt_launch(gx, gy, gz, bx, by, bz) -> str:
    """CUDA launch grid/block."""
    return f"{gx}x{gy}x{gz}/{bx}x{by}x{bz}"


def tmpl_tag(demangled: str) -> str:
    """Short FA template tag, e.g. f16<128,128,2,4>."""
    m = TMPL_RE.search(demangled or "")
    if not m:
        return ""
    name = m.group(1).replace("flash_attn_ext_", "")
    nums = re.findall(r"\((?:int|bool)\)(\d+)", m.group(2))
    return f"{name}<{','.join(nums)}>"


def mmvq_rows_per_block(ncols_dst: int) -> int:
    """Must match calc_rows_per_block() in mmvq.cu for GENERIC/GCN tables."""
    return 1 if ncols_dst == 1 else 2


def problem_shape(name: str, demangled: str, gx, gy, gz) -> str:
    """
    Logical problem shape when recoverable; empty string otherwise.

    mul_mat_vec_q: dst [M x ncols_dst]; gridX = ceil(M / rows_per_block(ncols_dst)).
    flash_attn_ext_*: demangled template tag.
    """
    if name == "mul_mat_vec_q":
        m = MMVQ_RE.search(demangled or "")
        if m:
            ggml_t = int(m.group(1))
            ncols_dst = int(m.group(2))
            M = int(gx) * mmvq_rows_per_block(ncols_dst)
            N = ncols_dst
            tname = GGML_TYPE_NAME.get(ggml_t, f"t{ggml_t}")
            if int(gy) > 1 or int(gz) > 1:
                return f"{M}x{N}x{gy}x{gz}/{tname}"
            return f"{M}x{N}/{tname}"

    if is_fa_main(name):
        return tmpl_tag(demangled) or ""

    return ""


def is_fa_main(name: str) -> bool:
    return bool(FA_MAIN_RE.match(name or ""))


# ── nsys I/O ─────────────────────────────────────────────────────────────────

def export_sqlite(nsys_rep: Path, sqlite_path: Path, nsys_bin: str) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nsys_bin, "export", "--type=sqlite", "--force-overwrite=true",
        "-o", str(sqlite_path), str(nsys_rep),
    ]
    print(f"[export] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def load_kernels(conn: sqlite3.Connection):
    """Return list of (start, end, dur_ns, name, launch, problem, demangled)."""
    rows = conn.execute(
        """
        SELECT k.start, k.end, sn.value,
               k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
               dn.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds sn ON sn.id = k.shortName
        LEFT JOIN StringIds dn ON dn.id = k.demangledName
        ORDER BY k.start
        """
    ).fetchall()
    out = []
    for s, e, name, gx, gy, gz, bx, by, bz, dem in rows:
        dem = dem or ""
        launch = fmt_launch(gx, gy, gz, bx, by, bz)
        problem = problem_shape(name, dem, gx, gy, gz)
        out.append((int(s), int(e), int(e) - int(s), name, launch, problem, dem))
    return out


def resolve_sqlite(path: Path, nsys_bin: str, keep_sqlite: bool, sqlite_dir: Path | None):
    if path.suffix.lower() in (".sqlite", ".db"):
        return path, False
    if path.suffix.lower() not in (".nsys-rep", ".qdrep"):
        raise SystemExit(f"Unsupported input: {path}")

    if sqlite_dir is not None:
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path = sqlite_dir / (path.stem + ".sqlite")
        if not sqlite_path.is_file():
            export_sqlite(path, sqlite_path, nsys_bin)
        return sqlite_path, False

    if keep_sqlite:
        sqlite_path = path.with_suffix(".sqlite")
        if not sqlite_path.is_file():
            export_sqlite(path, sqlite_path, nsys_bin)
        return sqlite_path, False

    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".sqlite")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        export_sqlite(path, tmp_path, nsys_bin)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path, True


# ── decode anchor selection ──────────────────────────────────────────────────

def detect_decode_anchor(kernels, anchor_filter: str | None = None):
    """
    Pick the decode flash-attn demangled specialization.

    Prefill and decode may both use flash_attn_ext_f16 with the same grid shape;
    they differ in demangled template (ncols) and timeline (decode starts later).

    Returns dict: name, demangled, tmpl, launch (majority), problem, n, t0, t1
    """
    groups: dict[str, dict] = {}
    for s, e, d, name, launch, problem, dem in kernels:
        if not is_fa_main(name):
            continue
        if anchor_filter:
            tag = tmpl_tag(dem)
            if (
                anchor_filter not in name
                and anchor_filter not in dem
                and anchor_filter not in tag
            ):
                continue
        key = dem if dem else f"{name}|{launch}"
        g = groups.get(key)
        if g is None:
            g = {
                "name": name,
                "demangled": dem,
                "tmpl": tmpl_tag(dem),
                "launches": Counter(),
                "problems": Counter(),
                "n": 0,
                "t0": s,
                "t1": e,
                "durs": [],
            }
            groups[key] = g
        g["n"] += 1
        g["t0"] = min(g["t0"], s)
        g["t1"] = max(g["t1"], e)
        g["launches"][launch] += 1
        g["problems"][problem] += 1
        g["durs"].append(d)

    if not groups:
        raise SystemExit(
            "No flash_attn_ext_{vec,f16,...} kernels found"
            + (f" matching filter '{anchor_filter}'" if anchor_filter else "")
        )

    max_n = max(g["n"] for g in groups.values())
    majors = [g for g in groups.values() if g["n"] >= max(32, max_n // 4)]
    if not majors:
        majors = list(groups.values())
    chosen = max(majors, key=lambda g: g["t0"])
    chosen["launch"] = chosen["launches"].most_common(1)[0][0]
    chosen["problem"] = chosen["problems"].most_common(1)[0][0]
    # keep "shape" alias for older meta prints
    chosen["shape"] = chosen["problem"] or chosen["launch"]
    return chosen


def is_anchor_kernel(name: str, demangled: str, anchor: dict) -> bool:
    if anchor["demangled"]:
        return demangled == anchor["demangled"]
    return name == anchor["name"]


# ── decode segmentation ──────────────────────────────────────────────────────

def decode_window(kernels, anchor: dict):
    """Kernels inside [first_anchor_start, last_anchor_end]."""
    anchors = [
        (s, e)
        for s, e, _, name, _launch, _prob, dem in kernels
        if is_anchor_kernel(name, dem, anchor)
    ]
    if not anchors:
        return None, [], []
    t0 = min(s for s, _ in anchors)
    t1 = max(e for _, e in anchors)
    window = [k for k in kernels if t0 <= k[0] <= t1]
    fa_idx = [
        i for i, k in enumerate(window) if is_anchor_kernel(k[3], k[6], anchor)
    ]
    return (t0, t1, len(anchors)), window, fa_idx


def inter_fa_segments(window, fa_idx):
    """
    segs[k] = (fa_k_key, fa_k_dur, keys_between, durs_between)
    key = (name, launch, problem)
    """
    segs = []
    for i in range(len(fa_idx) - 1):
        a, b = fa_idx[i], fa_idx[i + 1]
        fa = window[a]
        fa_key = (fa[3], fa[4], fa[5])
        fa_dur = fa[2]
        keys = [(window[j][3], window[j][4], window[j][5]) for j in range(a + 1, b)]
        durs = [window[j][2] for j in range(a + 1, b)]
        segs.append((fa_key, fa_dur, keys, durs))
    return segs


def detect_n_layers(seg_lens: list[int]) -> tuple[int, int, int]:
    counts = Counter(seg_lens)
    mode_len = counts.most_common(1)[0][0]
    boundary_len = max(counts.keys())
    idxs = [i for i, L in enumerate(seg_lens) if L == boundary_len]
    if not idxs:
        raise SystemExit("Could not find token-boundary inter-FA segments")
    n_layers = idxs[0] + 1
    if len(idxs) >= 2:
        gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
        if len(set(gaps)) == 1:
            n_layers = gaps[0]
    return n_layers, mode_len, boundary_len


def attn_prep_start(keys: list[tuple]) -> int:
    """Index where next-layer attn prep begins inside an inter-FA key list."""
    ropes = [i for i, k in enumerate(keys) if "rope" in k[0]]
    if not ropes:
        norms = [i for i, k in enumerate(keys) if "rms_norm" in k[0]]
        norms = [i for i in norms if i >= len(keys) // 2]
        return norms[0] if norms else len(keys)
    trailing = [i for i in ropes if i >= len(keys) // 2] or [ropes[-1]]
    r0 = trailing[0]
    for j in range(r0 - 1, -1, -1):
        if "rms_norm" in keys[j][0]:
            return j
    return r0


def stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"count": 0, "mean_us": 0.0, "std_us": 0.0, "min_us": 0.0, "max_us": 0.0}
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {
        "count": n,
        "mean_us": mean / 1e3,
        "std_us": std / 1e3,
        "min_us": min(vals) / 1e3,
        "max_us": max(vals) / 1e3,
    }


def row_from_key(stage: str, slot: int, key: tuple, durs: list[int]) -> dict:
    name, launch, problem = key
    return {
        "stage": stage,
        "slot": slot,
        "kernel": name,
        "launch": launch,
        "problem": problem,
        **stats(durs),
        "sum_us": sum(durs) / 1e3 if durs else 0.0,
    }


def build_layer_means(segs, mode_len: int, S: int, mode_keys: list[tuple]):
    """
    Layer i = attn_prep(segs[i-1][S:]) + FA_i + post_ffn(segs[i][:S]).
    segs[k] = (fa_k_key, fa_k_dur, keys_between, durs_between).
    """
    # slot keys for a full layer in attn→FFN order
    # FA key taken from first matching instance
    n_slots = (len(mode_keys) - S) + 1 + S
    buckets: list[list[int]] = [[] for _ in range(n_slots)]
    slot_keys: list[tuple | None] = [None] * n_slots

    # preset non-FA slots from mode template
    for k, key in enumerate(mode_keys[S:]):
        slot_keys[k] = key
    for k, key in enumerate(mode_keys[:S]):
        slot_keys[(len(mode_keys) - S) + 1 + k] = key

    n_used = 0
    for i in range(1, len(segs)):
        prev = segs[i - 1]
        cur = segs[i]
        if len(prev[2]) != mode_len or len(cur[2]) != mode_len:
            continue
        if prev[2] != mode_keys or cur[2] != mode_keys:
            continue
        fa_key, fa_dur = cur[0], cur[1]
        slot_durs = prev[3][S:] + [fa_dur] + cur[3][:S]
        if len(slot_durs) != n_slots:
            continue
        fa_slot = len(mode_keys) - S
        slot_keys[fa_slot] = fa_key
        for k, d in enumerate(slot_durs):
            buckets[k].append(d)
        n_used += 1

    rows = []
    for k in range(n_slots):
        key = slot_keys[k] or ("?", "?")
        rows.append(row_from_key("layer", k, key, buckets[k]))
    return rows, n_used


def extract_io_means(segs, mode_keys: list[tuple], S: int, boundary_len: int):
    """Input/output kernels from token-boundary segments (shape-aware match)."""
    prep = mode_keys[S:]
    post = mode_keys[:S]
    input_buckets: dict[tuple, list[int]] = defaultdict(list)
    output_buckets: dict[tuple, list[int]] = defaultdict(list)
    input_order: list[tuple] = []
    output_order: list[tuple] = []
    n_tok = 0

    for fa_key, fa_dur, keys, durs in segs:
        if len(keys) != boundary_len:
            continue
        n = list(keys)
        d = list(durs)

        # trailing cpy* → input (before next FA)
        while n and "cpy" in n[-1][0]:
            key = n.pop()
            dur = d.pop()
            if key not in input_buckets:
                input_order.append(key)
            input_buckets[key].append(dur)

        if len(n) < len(prep) or n[-len(prep) :] != prep:
            continue
        n = n[: -len(prep)]
        d = d[: -len(prep)]

        # shape-aware align to last-layer post/FFN; leftovers → output
        # (lm_head is same name as ffn GEMV but different grid → auto-separates)
        pi = 0
        for key, dur in zip(n, d):
            if pi < len(post) and key == post[pi]:
                pi += 1
            else:
                if key not in output_buckets:
                    output_order.append(key)
                output_buckets[key].append(dur)
        n_tok += 1

    input_rows = [
        row_from_key("input", i, k, input_buckets[k]) for i, k in enumerate(input_order)
    ]
    output_rows = [
        row_from_key("output", i, k, output_buckets[k]) for i, k in enumerate(output_order)
    ]
    return input_rows, output_rows, n_tok


def summarize_flat(window) -> list[dict]:
    """Aggregate all decode-window kernels by (name, launch, problem), sorted by sum time."""
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for _s, _e, dur, name, launch, problem, _dem in window:
        by_key[(name, launch, problem)].append(dur)
    rows = []
    for key, durs in by_key.items():
        r = row_from_key("flat", 0, key, durs)
        rows.append(r)
    total = sum(r["sum_us"] for r in rows) or 1.0
    for r in rows:
        r["pct_of_decode"] = 100.0 * r["sum_us"] / total
    rows.sort(key=lambda r: -r["sum_us"])
    for i, r in enumerate(rows):
        r["slot"] = i
    return rows


def analyze(kernels, anchor_filter: str | None = None):
    anchor = detect_decode_anchor(kernels, anchor_filter)
    meta_w, window, fa_idx = decode_window(kernels, anchor)
    if meta_w is None:
        raise SystemExit(f"No launches for chosen decode anchor {anchor['tmpl'] or anchor['name']}")
    t0, t1, n_anchors = meta_w
    segs = inter_fa_segments(window, fa_idx)
    if len(segs) < 2:
        raise SystemExit("Need at least 2 flash_attn anchors to form a layer")

    seg_lens = [len(keys) for _fk, _fd, keys, _dd in segs]
    n_layers, mode_len, boundary_len = detect_n_layers(seg_lens)

    mode_seqs = [tuple(keys) for _fk, _fd, keys, _dd in segs if len(keys) == mode_len]
    if not mode_seqs:
        raise SystemExit("No mode-length inter-FA segments")
    mode_keys = list(Counter(mode_seqs).most_common(1)[0][0])
    S = attn_prep_start(mode_keys)

    layer_rows, n_layer_inst = build_layer_means(segs, mode_len, S, mode_keys)
    input_rows, output_rows, n_tok = extract_io_means(segs, mode_keys, S, boundary_len)
    flat_rows = summarize_flat(window)

    n_tg = n_anchors // n_layers if n_layers else 0
    decode_wall_ms = (t1 - t0) / 1e6
    decode_wall_s = decode_wall_ms / 1e3
    tok_s = (n_tg / decode_wall_s) if decode_wall_s > 0 and n_tg > 0 else 0.0

    meta = {
        "n_anchors": n_anchors,
        "n_layers": n_layers,
        "n_tg_est": n_tg,
        "mode_inter_fa_len": mode_len,
        "boundary_inter_fa_len": boundary_len,
        "attn_prep_start": S,
        "n_layer_instances": n_layer_inst,
        "n_token_io": n_tok,
        "decode_wall_ms": decode_wall_ms,
        "decode_tok_s": tok_s,
        "anchor_name": anchor["name"],
        "anchor_tmpl": anchor["tmpl"],
        "anchor_launch": anchor["launch"],
        "anchor_problem": anchor["problem"],
        "anchor_shape": anchor["shape"],
        "anchor_n": anchor["n"],
        "t0": t0,
        "t1": t1,
    }
    return meta, input_rows, layer_rows, output_rows, flat_rows


# ── printing / CSV ───────────────────────────────────────────────────────────

def print_structure(title: str, rows: list[dict], file=sys.stderr):
    print(f"\n=== {title} ===", file=file)
    if not rows:
        print("  (none)", file=file)
        return
    print(
        f"  {'slot':>4}  {'kernel':28s}  {'launch':22s}  {'problem':18s}  "
        f"{'count':>7}  {'mean_us':>10}  {'std_us':>10}",
        file=file,
    )
    for r in rows:
        print(
            f"  {r['slot']:4d}  {r['kernel'][:28]:28s}  {r['launch'][:22]:22s}  "
            f"{(r['problem'] or '-')[:18]:18s}  {r['count']:7d}  "
            f"{r['mean_us']:10.3f}  {r['std_us']:10.3f}",
            file=file,
        )
    total = sum(r["mean_us"] for r in rows)
    print(
        f"  {'':4}  {'TOTAL (sum of means)':28s}  {'':22s}  {'':18s}  {'':7}  {total:10.3f}",
        file=file,
    )


def print_flat(rows: list[dict], top: int, file=sys.stderr):
    print(f"\n=== Flat decode kernels (by total time, top {top}) ===", file=file)
    print(
        f"  {'#':>4}  {'kernel':28s}  {'launch':22s}  {'problem':18s}  "
        f"{'count':>7}  {'mean_us':>10}  {'sum_us':>12}  {'pct':>7}",
        file=file,
    )
    for r in rows[:top]:
        print(
            f"  {r['slot']:4d}  {r['kernel'][:28]:28s}  {r['launch'][:22]:22s}  "
            f"{(r['problem'] or '-')[:18]:18s}  {r['count']:7d}  "
            f"{r['mean_us']:10.3f}  {r['sum_us']:12.1f}  {r['pct_of_decode']:6.2f}%",
            file=file,
        )


def print_throughput(meta: dict, file=sys.stderr):
    print("\n=== Decode throughput ===", file=file)
    print(
        f"  anchor            = {meta['anchor_name']} {meta['anchor_tmpl']} "
        f"launch={meta['anchor_launch']} problem={meta['anchor_problem'] or '-'} "
        f"(n={meta['anchor_n']})",
        file=file,
    )
    print(f"  n_layers          = {meta['n_layers']}", file=file)
    print(f"  n_tg (est)        = {meta['n_tg_est']}", file=file)
    print(f"  decode_wall_ms    = {meta['decode_wall_ms']:.3f}", file=file)
    print(f"  decode_tok_s      = {meta['decode_tok_s']:.2f} tok/s", file=file)
    if meta["n_tg_est"] > 0:
        print(
            f"  ms/token          = {meta['decode_wall_ms'] / meta['n_tg_est']:.3f}",
            file=file,
        )


def write_csv(path: Path | None, meta: dict, sections: dict, source: str):
    """Write all sections into one CSV with a stage column."""
    fieldnames = [
        "stage", "slot", "kernel", "launch", "problem", "count",
        "mean_us", "std_us", "min_us", "max_us", "sum_us", "pct_of_decode",
    ]
    out = open(path, "w", newline="") if path else sys.stdout
    close = path is not None
    try:
        out.write(
            f"# source={source} anchor={meta['anchor_name']} tmpl={meta['anchor_tmpl']} "
            f"launch={meta['anchor_launch']} problem={meta['anchor_problem']} "
            f"n_layers={meta['n_layers']} n_tg_est={meta['n_tg_est']} "
            f"decode_wall_ms={meta['decode_wall_ms']:.3f} "
            f"decode_tok_s={meta['decode_tok_s']:.2f} "
            f"n_layer_instances={meta['n_layer_instances']} n_token_io={meta['n_token_io']}\n"
        )
        w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for stage in ("input", "layer", "output", "flat"):
            for r in sections[stage]:
                w.writerow(
                    {
                        **r,
                        "mean_us": f"{r['mean_us']:.4f}",
                        "std_us": f"{r['std_us']:.4f}",
                        "min_us": f"{r['min_us']:.4f}",
                        "max_us": f"{r['max_us']:.4f}",
                        "sum_us": f"{r.get('sum_us', 0.0):.4f}",
                        "pct_of_decode": f"{r.get('pct_of_decode', 0.0):.3f}",
                    }
                )
    finally:
        if close:
            out.close()


def analyze_one(
    path: Path,
    anchor_filter: str | None,
    nsys_bin: str,
    keep_sqlite: bool,
    sqlite_dir: Path | None,
):
    sqlite_path, is_temp = resolve_sqlite(path, nsys_bin, keep_sqlite, sqlite_dir)
    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        kernels = load_kernels(conn)
        conn.close()
    finally:
        if is_temp:
            sqlite_path.unlink(missing_ok=True)
    if not kernels:
        raise SystemExit(f"No CUPTI kernels in {path}")
    return analyze(kernels, anchor_filter)


def main():
    p = argparse.ArgumentParser(
        description="Decode kernel timings: layer structure + flat + tok/s."
    )
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument(
        "--anchor",
        default="auto",
        help="Decode FA filter: 'auto' (default) picks latest major "
             "flash_attn_ext_{vec,f16} demangled group; or a substring "
             "e.g. flash_attn_ext_vec / flash_attn_ext_f16 / f16<128,128,2",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--keep-sqlite", action="store_true")
    p.add_argument("--sqlite-dir", type=Path, default=None)
    p.add_argument("--nsys", default="nsys")
    p.add_argument("--top", type=int, default=30, help="Top-N flat kernels to print")
    args = p.parse_args()

    inputs = [i for i in args.inputs if i.is_file()]
    for m in args.inputs:
        if not m.is_file():
            print(f"WARN: skip missing {m}", file=sys.stderr)
    if not inputs:
        sys.exit("No input files found")
    if args.out_dir is None and len(inputs) > 1:
        sys.exit("Multiple inputs require --out-dir")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    anchor_filter = None if args.anchor in ("auto", "", "none") else args.anchor

    ok = 0
    failed = 0
    for inp in inputs:
        try:
            meta, input_rows, layer_rows, output_rows, flat_rows = analyze_one(
                inp, anchor_filter, args.nsys, args.keep_sqlite, args.sqlite_dir
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {inp.name}: {exc}", file=sys.stderr)
            continue

        print(
            f"[decode] {inp.name}: anchor={meta['anchor_name']} {meta['anchor_tmpl'] or ''} "
            f"launch={meta['anchor_launch']} problem={meta['anchor_problem'] or '-'} "
            f"n={meta['anchor_n']} | "
            f"layers={meta['n_layers']} tg≈{meta['n_tg_est']} "
            f"tok/s={meta['decode_tok_s']:.1f} wall={meta['decode_wall_ms']:.1f} ms "
            f"layer_inst={meta['n_layer_instances']} token_io={meta['n_token_io']}",
            file=sys.stderr,
        )
        print_throughput(meta)
        print_structure(
            f"Input (mean over {meta['n_token_io']} token boundaries)",
            input_rows,
        )
        print_structure(
            f"Layer attn→FFN (mean over {meta['n_layer_instances']} layer instances)",
            layer_rows,
        )
        print_structure(
            f"Output (mean over {meta['n_token_io']} token boundaries)",
            output_rows,
        )
        print_flat(flat_rows, args.top)

        sections = {
            "input": input_rows,
            "layer": layer_rows,
            "output": output_rows,
            "flat": flat_rows,
        }
        if args.out_dir is not None:
            out_csv = args.out_dir / (inp.stem + "_decode_kernels.csv")
            write_csv(out_csv, meta, sections, str(inp))
            print(f"[wrote] {out_csv}", file=sys.stderr)
        else:
            write_csv(None, meta, sections, str(inp))
        ok += 1

    if len(inputs) > 1:
        print(f"[done] ok={ok} failed={failed}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
