#!/usr/bin/env python3
"""benchmark_kv_timing.py — Measure real decode ms/token with native llama.cpp KV types.

Uses llama.cpp's native --cache-type-k support (Q8_0, Q4_0, Q5_0, Q4_1) which stores
KV in genuinely compressed GPU memory with real attention bandwidth savings.
Compares measured ms/token against roofline predictions from perf_model.py.

Weight quantization (GGUF filenames)
--------------------------------------
You can pass **any** GGUF; pass several paths to compare weight quants in one run.
The **weight_tag** and effective **weight_bits** (for roofline) are inferred from the
filename (last -Q... suffix), including Q2_K, Q3_K_M, Q4_K_M, Q8_0, etc.
See _infer_weight_tag / _infer_weight_bits. K-quants use approximate
bits-per-weight (e.g. Q2_K≈3.1, Q3_K*≈4.0, Q4_K*≈4.9) for roofline only — measured
timings always use the real loaded model.

Workload model
--------------
  Prefill  prompt_len tokens  (simulates the input prompt)
  Warmup   n_warmup steps     (GPU pipeline warm-up, discarded)
  Decode   decode_len steps   (timed autoregressive generation)

The KV cache grows from prompt_len to prompt_len + decode_len during the timed window.
Roofline is evaluated at the midpoint (prompt_len + decode_len // 2) so it represents
the average KV bandwidth load over the full decode trajectory.

Set decode_len to match realistic generation lengths (512 for short tasks,
4096–8192 for long-form generation) so the timing captures the full KV growth.

Use --decode-bucket-size 64 (for example) to print ms/tok and tok/s for each 64-token
slice of the timed decode, so you can see throughput vs growing KV length.

Use --prefill-bucket-size 64 to aggregate wall time from each native prefill chunk (512-token
llama_decode steps) into N-token buckets, so you can see prefill tok/s vs prompt position.

Usage:
    # GSM8K-like: short prompt, short decode
    python3 research/scripts/benchmark_kv_timing.py \\
        models/Qwen3-8B-Q8_0.gguf \\
        --hw gh200 --model-preset qwen3-8b --n-gpu-layers 99 --flash-attn \\
        --prompt-lens 512 --decode-len 512 --batch-sizes 1 4 16

    # Long-form generation: long prompt, long decode (sweep batch sizes)
    python3 research/scripts/benchmark_kv_timing.py \\
        models/Qwen3-8B-Q8_0.gguf \\
        models/Qwen3-8B-Q8_0-Q2_K.gguf \\
        models/Qwen3-8B-Q8_0-Q4_K_M.gguf \\
        --hw gh200 --model-preset qwen3-8b --n-gpu-layers 99 --flash-attn \\
        --prompt-lens 1024 4096 --decode-len 512 2048 4096 \\
        --batch-sizes 1 4 16 32 --kv-types f16 q8_0 q4_0 \\
        --out calib_qwen3-8b.json

Output:
    Table: (weight, kv, prompt, decode_len, B) -> mean ms/tok | tok/s | roofline | calib
    JSON:  rows[] for every combination of models × kv-types × prompt-lens × decode-lens × batch;
            bootstrap_ms_quant keys include d{decode_len}; metadata lists decode_lens.
"""

import argparse
import contextlib
import ctypes
import json
import math
import os
import re
import sys
import time
from typing import Any, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import llama_bindings as llama

# Try to import roofline model for comparison
try:
    from perf_model import MODEL_PRESETS, HARDWARE_PRESETS, QUANT_CONFIGS
    from perf_model import kv_bytes_per_token, weight_bytes, decode_time_for_tokens, model_step_flops
    HAS_PERF_MODEL = True
except ImportError:
    HAS_PERF_MODEL = False
    print("[warn] perf_model.py not found — roofline comparison disabled", file=sys.stderr)

# ── Native KV type registry ────────────────────────────────────────────────────
# Only types supported by llama.cpp's kv_cache_types list in common/arg.cpp
KV_TYPE_MAP = {
    "f16":   1,   # GGML_TYPE_F16  — 2 bytes/elem
    "q8_0":  8,   # GGML_TYPE_Q8_0 — 8-bit block quant: ~1 byte/elem
    "q5_0":  6,   # GGML_TYPE_Q5_0 — 5-bit block quant: ~0.625 byte/elem
    "q4_0":  2,   # GGML_TYPE_Q4_0 — 4-bit block quant: ~0.5 byte/elem
    "q4_1":  3,   # GGML_TYPE_Q4_1 — 4-bit block quant (with min): ~0.5 byte/elem
}

# Effective bytes per element for roofline comparison
# (data + scale overhead, 32-element blocks for Qx_0 types)
KV_EFFECTIVE_BPE = {
    "f16":   2.0,
    "q8_0":  1.0 + 2.0/32,    # 8 bits data + fp16 scale per 32 elems
    "q5_0":  0.625 + 2.0/32,  # 5 bits data + fp16 scale per 32 elems
    "q4_0":  0.5   + 2.0/32,  # 4 bits data + fp16 scale per 32 elems
    "q4_1":  0.5   + 4.0/32,  # 4 bits data + fp16 scale + fp16 min per 32 elems
}


def _infer_weight_tag(path: str) -> str:
    """Extract full weight quantization name from model filename (last -Q{N}... suffix).

    Examples:
        Qwen3-8B-Q8_0.gguf          → Q8_0
        Qwen3-8B-Q8_0-Q4_K_M.gguf  → Q4_K_M   (last suffix wins)
        Qwen3-8B-Q8_0-Q2_K.gguf    → Q2_K
    """
    name = os.path.basename(path)
    # Match -Q<digit><word_chars>  e.g. -Q8_0, -Q4_K_M, -Q2_K
    # \w covers A-Z, 0-9, _ so stops cleanly at '.' before extension
    matches = re.findall(r'-([Qq]\d\w*)', name)
    if matches:
        return matches[-1].upper()
    return "Q8_0"   # fallback


def _infer_weight_bits(path: str) -> float:
    """Effective bits per weight element from model filename (roofline comparison only)."""
    name = os.path.basename(path).upper()
    if "Q2_K" in name:  return 3.1   # Q2_K / Q2_K_M
    if "Q3_K" in name:  return 4.0   # Q3_K_M, Q3_K_S, etc.
    if "Q4_K" in name:  return 4.9   # Q4_K_M, Q4_K_S, etc.
    if "Q8_0" in name:  return 8.5   # Q8_0 effective bpw
    if "F16"  in name:  return 16.0
    return 8.5   # default: assume Q8_0


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect fd 2 to /dev/null to suppress llama.cpp loading output."""
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


PREFILL_CHUNK = 512  # native chunk size for llama_decode during prefill (matches _prefill)


def _prefill(lib, ctx, n_tokens: int, seq_id: int = 0):
    """Fill KV cache with n_tokens for one sequence."""
    pos = 0
    while pos < n_tokens:
        chunk = min(PREFILL_CHUNK, n_tokens - pos)
        batch = lib.llama_batch_init(chunk, 0, 1)
        for i in range(chunk):
            batch.token[i]     = 1
            batch.pos[i]       = pos + i
            batch.n_seq_id[i]  = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i]    = 0
        batch.n_tokens = chunk
        lib.llama_decode(ctx, batch)
        lib.llama_batch_free(batch)
        pos += chunk


def _prefill_sequence_chunk_times(lib, ctx, n_tokens: int, seq_id: int = 0) -> list:
    """Time each native prefill chunk (PREFILL_CHUNK tokens per llama_decode)."""
    pos = 0
    chunks = []
    while pos < n_tokens:
        chunk = min(PREFILL_CHUNK, n_tokens - pos)
        batch = lib.llama_batch_init(chunk, 0, 1)
        for i in range(chunk):
            batch.token[i]     = 1
            batch.pos[i]       = pos + i
            batch.n_seq_id[i]  = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i]    = 0
        batch.n_tokens = chunk
        t0 = time.perf_counter()
        lib.llama_decode(ctx, batch)
        t1 = time.perf_counter()
        lib.llama_batch_free(batch)
        chunks.append({
            "from": pos, "to": pos + chunk - 1, "n": chunk,
            "seconds": float(t1 - t0),
        })
        pos += chunk
    return chunks


def _prefill_multi_chunk_times(lib, ctx, n_tokens: int, n_seqs: int) -> list:
    """Chunk times for prefill; if n_seqs>1, average wall time per chunk index across sequences."""
    if n_tokens <= 0:
        return []
    per_seq = [_prefill_sequence_chunk_times(lib, ctx, n_tokens, seq_id=s)
               for s in range(n_seqs)]
    if n_seqs == 1:
        return per_seq[0]
    nchunks = len(per_seq[0])
    merged = []
    for i in range(nchunks):
        avg_s = float(np.mean([per_seq[s][i]["seconds"] for s in range(n_seqs)]))
        row = dict(per_seq[0][i])
        row["seconds"] = avg_s
        merged.append(row)
    return merged


def _prefill_buckets_from_chunks(chunks: list, prompt_len: int, bucket_size: int) -> Optional[list]:
    """Split chunk times into prompt token buckets (proportional overlap per bucket)."""
    if not chunks or prompt_len <= 0 or bucket_size <= 0:
        return None
    buckets = []
    for b0 in range(0, prompt_len, bucket_size):
        b1 = min(b0 + bucket_size, prompt_len)
        acc_s = 0.0
        for ch in chunks:
            a, cend = ch["from"], ch["to"] + 1
            overlap = max(0, min(cend, b1) - max(a, b0))
            if overlap <= 0:
                continue
            acc_s += ch["seconds"] * (overlap / ch["n"])
        n_tok = b1 - b0
        if n_tok <= 0:
            continue
        ms_per_token = (acc_s / n_tok) * 1000.0
        ctx_mid = (b0 + b1 - 1) // 2
        buckets.append({
            "prompt_from": b0,
            "prompt_to":   b1 - 1,
            "n_tokens":    n_tok,
            "ctx_mid":     ctx_mid,
            "ms_per_token": ms_per_token,
            "tok_per_s":   1000.0 / ms_per_token if ms_per_token > 0 else 0.0,
        })
    return buckets


def _prefill_multi(lib, ctx, n_tokens: int, n_seqs: int):
    """Prefill n_tokens for each of n_seqs sequences (seq_ids 0..n_seqs-1).

    Done sequentially per sequence so each has its own KV region.
    """
    for seq_id in range(n_seqs):
        _prefill(lib, ctx, n_tokens, seq_id=seq_id)


def _decode_batch(lib, ctx, pos: int, n_seqs: int) -> float:
    """Decode one new token simultaneously for n_seqs sequences.

    Each sequence i is at the same position `pos` in its own KV slot.
    Returns wall-clock time in seconds for the entire step.
    """
    batch = lib.llama_batch_init(n_seqs, 0, 1)
    batch.n_tokens = n_seqs
    for i in range(n_seqs):
        batch.token[i]     = 1
        batch.pos[i]       = pos
        batch.n_seq_id[i]  = 1
        batch.seq_id[i][0] = i
        batch.logits[i]    = 0
    batch.logits[n_seqs - 1] = 1   # need at least one logit
    t0 = time.perf_counter()
    lib.llama_decode(ctx, batch)
    t1 = time.perf_counter()
    lib.llama_batch_free(batch)
    return t1 - t0


def measure_decode_ms(lib, ctx, prefill_len: int,
                       n_warmup: int, decode_len: int,
                       batch_size: int = 1,
                       decode_bucket_size: int = 0,
                       prefill_bucket_size: int = 0):
    """Prefill prefill_len tokens for batch_size sequences, then time decode_len steps.

    Returns (mean_ms_per_token, decode_buckets, prefill_buckets).

    mean_ms_per_token: mean over all timed decode steps (wall-clock per step / batch_size).

    decode_buckets: if decode_bucket_size > 0, per-bucket ms/tok over decode token indices.
    prefill_buckets: if prefill_bucket_size > 0 and prefill_len > 0, per-bucket prefill
    stats (time from native PREFILL_CHUNK steps aggregated into N-token buckets).

    Physics: each step reads weights once (amortised over batch_size) but reads
    KV for all batch_size sequences → larger batch = more KV-bound.
    """
    prefill_buckets = None
    if prefill_bucket_size and prefill_bucket_size > 0 and prefill_len > 0:
        chunk_times = _prefill_multi_chunk_times(lib, ctx, prefill_len, batch_size)
        prefill_buckets = _prefill_buckets_from_chunks(
            chunk_times, prefill_len, prefill_bucket_size)
    else:
        _prefill_multi(lib, ctx, prefill_len, n_seqs=batch_size)

    pos = prefill_len

    for _ in range(n_warmup):
        _decode_batch(lib, ctx, pos, n_seqs=batch_size)
        pos += 1

    times = []
    for _ in range(decode_len):
        t = _decode_batch(lib, ctx, pos, n_seqs=batch_size)
        times.append(t)
        pos += 1

    step_ms = float(np.mean(times)) * 1000.0   # ms per step
    mean_ms = step_ms / batch_size

    decode_buckets = None
    if decode_bucket_size and decode_bucket_size > 0 and decode_len > 0:
        decode_buckets = []
        for b0 in range(0, decode_len, decode_bucket_size):
            b1 = min(b0 + decode_bucket_size, decode_len)
            chunk = times[b0:b1]
            chunk_mean_s = float(np.mean(chunk))
            mms = chunk_mean_s * 1000.0 / batch_size
            ctx_lo = prefill_len + b0
            ctx_hi = prefill_len + b1 - 1
            ctx_mid = (ctx_lo + ctx_hi) // 2
            decode_buckets.append({
                "decode_from": b0,
                "decode_to":   b1 - 1,
                "n_steps":     b1 - b0,
                "ctx_mid":     ctx_mid,
                "ms_per_token": mms,
                "tok_per_s":   1000.0 / mms if mms > 0 else 0.0,
            })

    return mean_ms, decode_buckets, prefill_buckets


def _print_prefill_bucket_lines(buckets: list, indent: str = "      "):
    """Pretty-print prefill buckets (prompt token range → ms/tok, tok/s)."""
    if not buckets:
        return
    print(f"{indent}prefill buckets (prompt positions → ms/tok, tok/s; ctx_mid ≈ mid of range):")
    for b in buckets:
        lo, hi = b["prompt_from"], b["prompt_to"]
        print(f"{indent}  [{lo:5d},{hi:5d}]  ctx_mid={b['ctx_mid']:5d}  "
              f"{b['ms_per_token']:.3f} ms/tok  ({b['tok_per_s']:.1f} tok/s)")


def _print_decode_bucket_lines(buckets: list, indent: str = "      "):
    """Pretty-print per-bucket decode stats (one line per bucket)."""
    if not buckets:
        return
    print(f"{indent}decode buckets (wall time / steps in range → ms/tok, tok/s; ctx_mid ≈ KV len):")
    for b in buckets:
        lo, hi = b["decode_from"], b["decode_to"]
        print(f"{indent}  [{lo:5d},{hi:5d}]  ctx_mid={b['ctx_mid']:5d}  "
              f"{b['ms_per_token']:.3f} ms/tok  ({b['tok_per_s']:.1f} tok/s)")


def roofline_ms(model_name: str, hw_name: str, weight_bits: float,
                kv_type: str, ctx_len: int, batch_size: int = 1) -> float:
    """Roofline prediction for ms/token (wall-clock per step / batch_size).

    ctx_len: effective KV context length for this comparison — use the midpoint
    prompt_len + decode_len // 2 so KV traffic matches the mean over a long decode.

    Batch-aware BW model:
      - Weights are read once per step regardless of batch_size → amortised.
      - KV is read for every sequence → scales with batch_size.
      - Compute scales linearly with batch_size.

    Returns NaN if perf_model unavailable or model/hw unknown.
    """
    if not HAS_PERF_MODEL:
        return float("nan")

    model = dict(MODEL_PRESETS.get(model_name, {}))
    if not model:
        return float("nan")
    hw = HARDWARE_PRESETS.get(hw_name)
    if hw is None:
        return float("nan")

    model["weight_bits"] = weight_bits

    bpe      = KV_EFFECTIVE_BPE.get(kv_type, 2.0)
    bw_ratio = bpe / 2.0

    fp16_kv_bpt = kv_bytes_per_token("fp16", model)
    kv_bpt   = fp16_kv_bpt * bw_ratio          # bytes per token per sequence

    w_bytes  = weight_bytes(model)
    sflops   = model_step_flops(model, ctx_len)  # per-token FLOPs
    bw_eff   = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"]
    comp_eff = hw["compute_tflops"] * 1e12 * hw["efficiency"]

    # Per-step wall-clock:
    #   weight BW: read once, shared across batch
    #   KV BW:     read for every sequence in the batch
    #   compute:   scales with batch_size
    t_mem_step     = (w_bytes + kv_bpt * ctx_len * batch_size) / bw_eff
    t_compute_step = 2.0 * sflops * batch_size / comp_eff
    t_step         = max(t_mem_step, t_compute_step)   # seconds

    return t_step / batch_size * 1000.0   # ms per token


# Native GPU KV quants (q8_0, q4_0) are *slower* than fp16 despite fewer bytes
# because flash-attention must dequantize compressed KV tiles before use.  The
# dequant latency is approximately per-kernel-call (fixed), not per-byte, so
# q4_0 ≈ q8_0 in wall time.  Applying a bytes-only model badly mispredicts KV
# latency for these types.  Use fp16 bytes × overhead as effective KV bytes.
#
# Overhead values from roofline_calibration.md (B=1 low end; B=32 high end):
#   f16:  1.00 (no dequant)
#   q8_0: 1.08 (dequant adds ~8–22% vs fp16 depending on B)
#   q4_0: 1.08 (same kernel path as q8_0 — different table size, same overhead)
KV_DEQUANT_OVERHEAD = {
    "f16":  1.00,
    "q8_0": 1.08,
    "q4_0": 1.08,
}


def roofline_decompose_ms(model_name: str, hw_name: str, weight_bits: float,
                           kv_type: str, ctx_len: int, batch_size: int = 1,
                           ) -> tuple[float, float]:
    """Weight-only and KV-only roofline components, each in ms per token.

    The standard roofline_ms() uses a single bandwidth pool:
        roof_ms = (w_bytes + kv_bpt*ctx*B) / (bw_eff * B) * 1000
                = w_bytes/(bw_eff*B)*1000 + kv_bpt*ctx/bw_eff*1000

    This function returns those two additive terms separately so a three-parameter
    calibration can fit independent correction factors (alpha, beta):
        predicted_ms = T_floor/B + alpha * wt_ms + beta * kv_ms

    KV component uses *effective* bytes, not storage bytes:
        - fp16: fp16 bytes (no dequant)
        - q8_0/q4_0: fp16 bytes × KV_DEQUANT_OVERHEAD (dequant costs ≈ reading fp16)

    Using storage bytes (as roofline_ms() does) gives accurate results only per-slice
    because each slice has one kv_type and absorbs the overhead into its own scale.
    The effective-byte model is needed for global multi-quant pooled fits.

    Returns (wt_ms_per_tok, kv_ms_per_tok), or (nan, nan) on unknown model/hw.
    """
    if not HAS_PERF_MODEL:
        return float("nan"), float("nan")
    model = dict(MODEL_PRESETS.get(model_name, {}))
    if not model:
        return float("nan"), float("nan")
    hw = HARDWARE_PRESETS.get(hw_name)
    if hw is None:
        return float("nan"), float("nan")

    model["weight_bits"] = weight_bits
    fp16_kv_bpt = kv_bytes_per_token("fp16", model)
    w_bytes     = weight_bytes(model)
    bw_eff      = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"]

    # Effective KV bytes: fp16 × overhead (dequant cost dominates byte savings for GPU quants)
    overhead = KV_DEQUANT_OVERHEAD.get(kv_type.lower(), 1.0)
    kv_bpt_eff = fp16_kv_bpt * overhead

    # Weight: read once per step, amortised over batch_size tokens
    wt_ms = w_bytes / bw_eff / batch_size * 1000.0
    # KV: read for every sequence; per-token = kv_bpt_eff * ctx_len (independent of B)
    kv_ms = kv_bpt_eff * ctx_len / bw_eff * 1000.0

    return wt_ms, kv_ms


def load_roofline_calibration_json(path: str) -> Optional[dict[str, Any]]:
    """Load JSON from fit_roofline_calibration.py (t_floor_ms + scale)."""

    with open(os.path.expanduser(path), encoding="utf-8") as f:
        cal = json.load(f)
    for k in ("t_floor_ms", "scale", "model_preset", "hw"):
        if k not in cal:
            print(f"[warn] calibration JSON missing {k!r}; ignoring {path}", file=sys.stderr)
            return None
    return cal


def calibrated_decode_ms_per_tok(roof_ms: float, batch_size: int, cal: dict) -> float:
    """measured_ms ≈ t_floor_ms / batch_size + scale * roofline_ms (fit from cluster)."""

    return float(cal["t_floor_ms"]) / float(batch_size) + float(cal["scale"]) * float(roof_ms)


def row_matches_roofline_calibration(
    cal: dict,
    *,
    model_preset: str,
    hw: str,
    weight_tag: str,
    kv_type: str,
    prompt_len: int,
) -> bool:
    if str(cal.get("model_preset")) != str(model_preset):
        return False
    if str(cal.get("hw")) != str(hw):
        return False
    wt = cal.get("weight_tag")
    if wt is not None and str(wt) != str(weight_tag):
        return False
    kv = cal.get("kv_type")
    if kv is not None and str(kv).lower() != str(kv_type).lower():
        return False
    if cal.get("prompt_len_filter") is not None:
        if int(prompt_len) != int(cal["prompt_len_filter"]):
            return False
    return True


def print_table(rows: list, title: str):
    """Print a formatted results table.

    rows: list of dicts with keys: weight_tag, kv_type, prompt_len, decode_len,
          batch_size, measured_ms, roofline_ms, calib_factor, speedup_vs_f16, tok_per_s
          optional: roofline_calibrated_ms
    """
    def _sf(x) -> float:
        if x is None:
            return float("nan")
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    W = 120
    has_cal = any(not math.isnan(_sf(r.get("roofline_calibrated_ms"))) for r in rows)
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")
    hdr = (f"  {'weight':>8} {'kv':>6} {'prompt+dec':>12} {'B':>4} "
           f"{'ms/tok':>9} {'tok/s':>8} {'roof ms/tok':>12} {'calib':>7} ")
    if has_cal:
        hdr += f"{'roof cal':>10} {'c2':>6} "
    hdr += f"{'speedup':>8}  note"
    print(hdr)
    print(f"{'─'*W}")

    # Baseline: f16, same (weight_tag, prompt_len, decode_len, batch_size)
    f16_baseline = {}
    for r in rows:
        if r["kv_type"] == "f16":
            f16_baseline[(r["weight_tag"], r["prompt_len"],
                          r["decode_len"], r["batch_size"])] = r["measured_ms"]

    prev_group = None

    for r in rows:
        meas    = r["measured_ms"]
        roof    = r["roofline_ms"]
        tps     = 1000.0 / meas if meas > 0 else 0.0
        calib   = meas / roof if (roof > 0 and not math.isnan(roof)) else float("nan")
        base    = f16_baseline.get((r["weight_tag"], r["prompt_len"],
                                    r["decode_len"], r["batch_size"]), float("nan"))
        speedup = base / meas if (meas > 0 and not math.isnan(base)) else float("nan")

        r["calib_factor"]   = calib
        r["speedup_vs_f16"] = speedup
        r["tok_per_s"]      = tps

        roof_cal = _sf(r.get("roofline_calibrated_ms"))
        calib2 = (
            meas / roof_cal
            if (roof_cal > 0 and not math.isnan(roof_cal))
            else float("nan")
        )
        r["calib_factor_vs_calibrated"] = calib2

        cur_group = (r["weight_tag"], r["prompt_len"], r["decode_len"])
        if prev_group is not None and cur_group != prev_group:
            print()
        prev_group = cur_group

        roof_s  = f"{roof:.3f}" if not math.isnan(roof) else "   n/a "
        calib_s = f"{calib:.3f}×" if not math.isnan(calib) else "   n/a"
        spd_s   = f"{speedup:.3f}×" if not math.isnan(speedup) else "  base"
        note    = ("← base" if r["kv_type"] == "f16"
                   else f"bw_ratio={KV_EFFECTIVE_BPE[r['kv_type']]/2:.3f}")

        line = (
            f"  {r['weight_tag']:>8} {r['kv_type']:>6}"
            f" {r['prompt_len']:>6}+{r['decode_len']:<5}"
            f" {r['batch_size']:>4} "
            f"{meas:>9.3f} {tps:>8.1f} {roof_s:>12} {calib_s:>7} "
        )
        if has_cal:
            rc = _sf(r.get("roofline_calibrated_ms"))
            rc_s = f"{rc:.3f}" if not math.isnan(rc) else "   n/a "
            c2_s = f"{calib2:.2f}×" if not math.isnan(calib2) else "  n/a "
            line += f"{rc_s:>10} {c2_s:>6} "
        line += f"{spd_s:>8}  {note}"
        print(line)

    print(f"\n{'═'*W}")

    # Summary: calibration factor consistency
    valid_calib = [r["calib_factor"] for r in rows
                   if not math.isnan(r.get("calib_factor", float("nan")))]
    if valid_calib:
        print(f"\n  Calibration factor stats (measured / roofline):")
        print(f"    mean={np.mean(valid_calib):.3f}  "
              f"std={np.std(valid_calib):.3f}  "
              f"min={np.min(valid_calib):.3f}  "
              f"max={np.max(valid_calib):.3f}")
        print(f"  → If std is small the roofline model scales correctly; "
              f"apply mean calib to int2/int3 projections.")


def extrapolate_int_quants(rows: list, model_name: str, hw_name: str,
                            prompt_len: int, decode_len: int,
                            weight_bits: float, weight_tag: str,
                            batch_size: int = 1):
    """Use calibration factor to project int2/int3/int4 channel-wise speedup.

    Roofline uses mid_ctx = prompt_len + decode_len // 2 to match benchmark_kv_timing.
    """
    if not HAS_PERF_MODEL:
        return

    mid_ctx = prompt_len + decode_len // 2

    calib_vals = [r["calib_factor"] for r in rows
                  if (r["weight_tag"] == weight_tag
                      and r["prompt_len"] == prompt_len
                      and r["decode_len"] == decode_len
                      and r["batch_size"] == batch_size
                      and not math.isnan(r.get("calib_factor", float("nan"))))]
    if not calib_vals:
        return
    calib = float(np.mean(calib_vals))

    print(f"\n  Projected ms/token ({weight_tag}, prompt={prompt_len}, decode={decode_len}, "
          f"mid_ctx={mid_ctx}, B={batch_size}) "
          f"using calib={calib:.3f}×:")
    print(f"  {'quant':<14} {'bpe':>5} {'bw_ratio':>9} "
          f"{'roof ms/tok':>12} {'calib ms/tok':>13} {'tok/s':>8} {'speedup vs f16':>15}")
    print(f"  {'─'*82}")

    f16_meas = next((r["measured_ms"] for r in rows
                     if r["weight_tag"] == weight_tag
                     and r["kv_type"] == "f16"
                     and r["prompt_len"] == prompt_len
                     and r["decode_len"] == decode_len
                     and r["batch_size"] == batch_size), None)

    model = dict(MODEL_PRESETS.get(model_name, {}))
    if not model:
        return
    model["weight_bits"] = weight_bits
    hw = HARDWARE_PRESETS.get(hw_name)
    if not hw:
        return

    fp16_kv_bpt = kv_bytes_per_token("fp16", model)
    w_bytes  = weight_bytes(model)
    sflops   = model_step_flops(model, mid_ctx)
    bw_eff   = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"]
    comp_eff = hw["compute_tflops"] * 1e12 * hw["efficiency"]

    # bpe in bytes per element (both K and V included in kv_bytes_per_token)
    quants_to_project = [
        ("int8_ch",  1.0 + 2.0/8/64),   # 8-bit data + fp16 scale per 64-token group
        ("int4_ch",  0.5 + 2.0/8/64),
        ("int3_ch",  0.375 + 2.0/8/64),
        ("int2_ch",  0.25  + 2.0/8/64),
    ]

    for qname, bpe in quants_to_project:
        bw_ratio = bpe / 2.0
        kv_bpt   = fp16_kv_bpt * bw_ratio
        # Batch-aware roofline (same formula as roofline_ms)
        t_mem_step     = (w_bytes + kv_bpt * mid_ctx * batch_size) / bw_eff
        t_compute_step = 2.0 * sflops * batch_size / comp_eff
        roof_ms        = max(t_mem_step, t_compute_step) / batch_size * 1000.0
        calib_ms       = roof_ms * calib
        tps            = 1000.0 / calib_ms if calib_ms > 0 else float("nan")
        speedup        = f16_meas / calib_ms if f16_meas else float("nan")
        spd_s          = f"{speedup:.3f}×" if not math.isnan(speedup) else "  n/a"
        tps_s          = f"{tps:.1f}" if not math.isnan(tps) else "  n/a"
        print(f"  {qname:<14} {bpe:>5.3f} {bw_ratio:>9.4f} "
              f"{roof_ms:>12.3f} {calib_ms:>13.3f} {tps_s:>8} {spd_s:>15}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("models", nargs="+",
                   help="Model GGUF path(s). Multiple = sweep weight quantizations.")

    # Hardware
    p.add_argument("--n-gpu-layers", type=int, default=99)
    p.add_argument("--flash-attn",   action="store_true", default=False)
    p.add_argument("--n-threads",    type=int, default=4)
    p.add_argument("--hw",           default="h100-sxm",
                   help="Hardware preset name for roofline (from perf_model.py)")
    p.add_argument("--model-preset", default=None,
                   help="Model preset name for roofline (auto-detected from filename if None)")

    # KV types to test
    p.add_argument("--kv-types", nargs="+",
                   default=["f16", "q8_0", "q4_0"],
                   choices=list(KV_TYPE_MAP),
                   help="KV cache types to benchmark (default: f16 q8_0 q4_0)")

    # Workload
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[1024],
                   help="Prompt (prefill) lengths in tokens (default: 1024). "
                        "Simulates the input prompt; KV starts here before timed decode.")
    p.add_argument("--decode-len", nargs="+", type=int, default=[512], dest="decode_lens",
                   metavar="N",
                   help="Decode length(s): tokens generated in the timed window (default: 512). "
                        "Pass multiple values to sweep in one run, e.g. 512 2048 4096. "
                        "Roofline uses mid_ctx = prompt_len + decode_len // 2 per row.")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1],
                   help="Batch sizes (concurrent sequences) to sweep (default: 1). "
                        "Larger batch → more KV-bound → KV quant speedup more visible. "
                        "Example: --batch-sizes 1 4 16 32")
    p.add_argument("--n-warmup",  type=int, default=20,
                   help="Warmup decode steps before timing, not included in decode_len (default: 20)")
    p.add_argument("--decode-bucket-size", type=int, default=0, metavar="N",
                   help="If N>0, report ms/tok and tok/s for each N-token slice of timed "
                        "decode (trend as KV grows). 0 = summary mean only (default: 0).")
    p.add_argument("--prefill-bucket-size", type=int, default=0, metavar="N",
                   help="If N>0, time each native prefill chunk (%d tok llama_decode), "
                        "aggregate into N-token prompt buckets (prefill tok/s vs position). "
                        "0 = off (default: 0)." % PREFILL_CHUNK)

    # Output
    p.add_argument("--out", default=None,
                   help="Save calibration JSON (also updates bootstrap_ms_quant_example.json entries)")
    p.add_argument("--lib", default=None,
                   help="Path to libllama.so (default: build_release/bin/libllama.so)")
    p.add_argument(
        "--roofline-calibration-json",
        default=None,
        metavar="PATH",
        help="JSON from fit_roofline_calibration.py: adds roofline_calibrated_ms = "
             "t_floor_ms/B + scale*roofline_ms when model/hw/weight/kv match",
    )

    args = p.parse_args()

    roofline_calib: Optional[dict] = None
    if args.roofline_calibration_json:
        roofline_calib = load_roofline_calibration_json(args.roofline_calibration_json)
        if roofline_calib:
            print(
                f"[info] loaded roofline calibration: "
                f"t_floor_ms={roofline_calib['t_floor_ms']}  scale={roofline_calib['scale']} "
                f"({args.roofline_calibration_json})",
                file=sys.stderr,
            )

    # Load library
    lib_path = args.lib or os.path.join(_SCRIPT_DIR, "../../build_release/bin/libllama.so")
    lib = llama.load_lib(lib_path)
    lib.llama_backend_init()
    # Keep load_lib()'s log callback — do not llama_log_set(None): that restores
    # ggml_log_callback_default and floods stderr with GGML_LOG_DEBUG (e.g. CUDA graph warmup).

    all_rows = []

    for model_path in args.models:
        weight_tag  = _infer_weight_tag(model_path)
        weight_bits = _infer_weight_bits(model_path)

        # Auto-detect model preset from filename
        model_preset = args.model_preset
        if model_preset is None:
            fname = os.path.basename(model_path).lower()
            for name in (MODEL_PRESETS if HAS_PERF_MODEL else {}).keys():
                if name.replace("-", "") in fname.replace("-", "").replace("_", ""):
                    model_preset = name
                    break

        print(f"\nLoading model: {model_path}")
        print(f"  weight_tag={weight_tag}  weight_bits={weight_bits:.1f}"
              f"  preset={model_preset or 'unknown'}")

        mparams = lib.llama_model_default_params()
        mparams.n_gpu_layers = args.n_gpu_layers
        model = lib.llama_model_load_from_file(model_path.encode(), mparams)
        if not model:
            print(f"  ERROR: failed to load model", file=sys.stderr)
            continue

        for kv_type_name in args.kv_types:
            kv_type_id = KV_TYPE_MAP[kv_type_name]

            for prompt_len in args.prompt_lens:
                for decode_len in args.decode_lens:
                    for batch_size in args.batch_sizes:
                        # n_ctx per sequence: prompt + warmup + full decode length
                        seq_slots = prompt_len + args.n_warmup + decode_len + 16
                        cparams = lib.llama_context_default_params()
                        cparams.n_ctx           = batch_size * seq_slots
                        cparams.n_seq_max       = batch_size   # allow seq_ids 0..batch_size-1
                        cparams.n_batch         = max(512, batch_size)
                        cparams.n_ubatch        = max(512, batch_size)
                        cparams.n_threads       = args.n_threads
                        cparams.n_threads_batch = args.n_threads
                        cparams.no_perf         = True
                        cparams.type_k          = kv_type_id
                        cparams.type_v          = kv_type_id
                        if args.flash_attn:
                            cparams.flash_attn_type = 1

                        with _suppress_c_stderr():
                            ctx = lib.llama_init_from_model(model, cparams)
                        if not ctx:
                            print(f"  ERROR: failed to create context "
                                  f"(kv={kv_type_name}, prompt={prompt_len}, "
                                  f"decode={decode_len}, B={batch_size})",
                                  file=sys.stderr)
                            continue

                        print(f"  kv={kv_type_name:6s}  prompt={prompt_len:5d}  "
                              f"decode={decode_len:5d}  B={batch_size:3d}  ... ",
                              end="", flush=True)

                        try:
                            ms, dec_buckets, pre_buckets = measure_decode_ms(
                                lib, ctx, prompt_len,
                                args.n_warmup, decode_len,
                                batch_size=batch_size,
                                decode_bucket_size=args.decode_bucket_size,
                                prefill_bucket_size=args.prefill_bucket_size)
                        except Exception as e:
                            print(f"ERROR: {e}")
                            lib.llama_free(ctx)
                            continue

                        # Roofline at midpoint of the decode trajectory
                        mid_ctx = prompt_len + decode_len // 2
                        roof = (roofline_ms(model_preset, args.hw, weight_bits,
                                            kv_type_name, mid_ctx, batch_size=batch_size)
                                if model_preset else float("nan"))

                        roof_cal = float("nan")
                        if (
                            roofline_calib is not None
                            and model_preset
                            and not math.isnan(roof)
                            and row_matches_roofline_calibration(
                                roofline_calib,
                                model_preset=model_preset,
                                hw=args.hw,
                                weight_tag=weight_tag,
                                kv_type=kv_type_name,
                                prompt_len=prompt_len,
                            )
                        ):
                            roof_cal = calibrated_decode_ms_per_tok(
                                roof, batch_size, roofline_calib)

                        tps = 1000.0 / ms if ms > 0 else 0.0
                        print(f"{ms:.3f} ms/tok  ({tps:.1f} tok/s)  "
                              + (f"roofline={roof:.3f}" if not math.isnan(roof) else ""))
                        if pre_buckets:
                            _print_prefill_bucket_lines(pre_buckets)
                        if dec_buckets:
                            _print_decode_bucket_lines(dec_buckets)

                        all_rows.append({
                            "model_path":    model_path,
                            "weight_tag":    weight_tag,
                            "weight_bits":   weight_bits,
                            "model_preset":  model_preset,
                            "kv_type":       kv_type_name,
                            "prompt_len":    prompt_len,
                            "decode_len":    decode_len,
                            "mid_ctx":       mid_ctx,
                            "batch_size":    batch_size,
                            "measured_ms":   ms,
                            "tok_per_s":     tps,
                            "roofline_ms":   roof,
                            "roofline_calibrated_ms": roof_cal,
                            "calib_factor":  float("nan"),
                            "speedup_vs_f16": float("nan"),
                            "prefill_buckets": pre_buckets,
                            "decode_buckets": dec_buckets,
                        })

                        lib.llama_free(ctx)

        lib.llama_model_free(model)

    if not all_rows:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    print_table(all_rows, f"Decode ms/token: measured vs roofline  "
                f"[hw={args.hw}  warmup={args.n_warmup}  decode_lens={args.decode_lens}]")

    # Extrapolate int2/int3/int4 for each (weight_tag, prompt_len, decode_len, batch_size) combo
    if HAS_PERF_MODEL:
        seen = set()
        for r in all_rows:
            key = (r["weight_tag"], r["prompt_len"], r["decode_len"],
                   r["batch_size"], r.get("model_preset"))
            if key in seen or r.get("model_preset") is None:
                continue
            seen.add(key)
            print(f"\n{'─'*100}")
            extrapolate_int_quants(all_rows, r["model_preset"], args.hw,
                                   r["prompt_len"], r["decode_len"],
                                   r["weight_bits"], r["weight_tag"],
                                   batch_size=r["batch_size"])

    # Save JSON output
    if args.out:
        f16_base = {}
        for r in all_rows:
            if r["kv_type"] == "f16":
                f16_base[(r["weight_tag"], r["prompt_len"], r["decode_len"],
                          r["batch_size"])] = r["measured_ms"]

        out_data = {
            "_comment": (
                "Calibration data from benchmark_kv_timing.py. "
                "calib_factor = measured_ms / roofline_ms (roofline at mid_ctx). "
                "roofline_calibrated_ms = t_floor_ms/B + scale*roofline_ms when "
                "--roofline-calibration-json matches. "
                "Use mean_calib to scale roofline projections for int2/int3/int4. "
                "Top-level decode_lens lists all timed decode lengths in this file; "
                "each row has its own decode_len."
            ),
            "hardware": args.hw,
            "n_warmup": args.n_warmup,
            "decode_lens": list(args.decode_lens),
            "prompt_lens": list(args.prompt_lens),
            "batch_sizes": list(args.batch_sizes),
            "kv_types": list(args.kv_types),
            "decode_bucket_size": args.decode_bucket_size,
            "prefill_bucket_size": args.prefill_bucket_size,
            "rows": [],
            # bootstrap_ms_quant: keyed by scenario (prompt, decode, mid_ctx in label)
            "bootstrap_ms_quant": {}
        }
        if args.roofline_calibration_json:
            out_data["roofline_calibration_file"] = args.roofline_calibration_json
            if roofline_calib:
                out_data["roofline_calibration"] = {
                    "t_floor_ms": roofline_calib["t_floor_ms"],
                    "scale": roofline_calib["scale"],
                    "model_preset": roofline_calib["model_preset"],
                    "hw": roofline_calib["hw"],
                }

        for r in all_rows:
            base    = f16_base.get((r["weight_tag"], r["prompt_len"], r["decode_len"],
                                    r["batch_size"]),
                                   float("nan"))
            speedup = base / r["measured_ms"] if r["measured_ms"] > 0 else float("nan")
            calib   = (r["measured_ms"] / r["roofline_ms"]
                       if r["roofline_ms"] > 0 and not math.isnan(r["roofline_ms"])
                       else float("nan"))
            row = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                   for k, v in r.items()}
            row["calib_factor"]   = None if math.isnan(calib)   else round(calib, 4)
            row["speedup_vs_f16"] = None if math.isnan(speedup) else round(speedup, 4)
            out_data["rows"].append(row)

            key = (f"{r['weight_tag']} {r['kv_type']} "
                   f"B{r['batch_size']} p{r['prompt_len']} d{r['decode_len']}")
            if key not in out_data["bootstrap_ms_quant"]:
                out_data["bootstrap_ms_quant"][key] = round(r["measured_ms"], 4)

        # Calibration factor stats
        calib_vals = [r["calib_factor"] for r in out_data["rows"]
                      if r["calib_factor"] is not None]
        if calib_vals:
            out_data["calib_mean"] = round(float(np.mean(calib_vals)), 4)
            out_data["calib_std"]  = round(float(np.std(calib_vals)), 4)

        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nSaved calibration data → {args.out}")
        print(f"bootstrap_ms_quant entries:")
        for k, v in out_data["bootstrap_ms_quant"].items():
            print(f"  {k!r}: {v}")

    lib.llama_backend_free()


if __name__ == "__main__":
    main()
