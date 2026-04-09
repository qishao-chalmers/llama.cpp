#!/usr/bin/env python3
"""run_sweep.py — KV cache quantization perplexity sweep.

Usage:
    python3 run_sweep.py models/Qwen3-8B-Q8_0.gguf research/data/wikitext2_test.txt \
        --n-ctx 128 --n-chunks 20 --n-threads 8 --out results.json
"""

import argparse, html, json, math, os, re, sys, time
from collections import Counter
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import llama_bindings as llama
import parse_state
import quant as quant_mod
import strategies
import kv_profile

try:
    import gpu_quant
    HAS_GPU_QUANT = gpu_quant.HAS_CUPY
except ImportError:
    HAS_GPU_QUANT = False

try:
    import gpu_kv_shadow
    HAS_GPU_KV_SHADOW = gpu_kv_shadow.HAS_CUPY
except ImportError:
    HAS_GPU_KV_SHADOW = False


def _adaptive_verify_label(args) -> str:
    if args.adaptive_verify_top_p is not None:
        return f"top_p={args.adaptive_verify_top_p}"
    if args.adaptive_verify_top_k is not None:
        return f"top_k={args.adaptive_verify_top_k}"
    return "greedy"


# Adaptive-gen segment viz: up to 8 rows × width cols (~ n_tokens/width tok/column).
# Glyphs: int2..fp16 use ':' stacks (height varies); int3 adds a top row '.' over ':'.
# Tuple rows are top→bottom within the stack (aligns with printed qviz top→bottom).
# Segments name draft quant vs fp16 only — verifier (e.g. int8_ch) is not duplicated in segment labels.
_QVIZ_HEIGHT = 8
_QVIZ_WIDTH = 64


def _quant_viz_stack(qname: str) -> tuple[int, str | tuple[str, ...]]:
    """Height (1–8) and one char (repeated) or tuple of chars top→bottom for the stack."""
    if not qname or qname == "fp16":
        return (_QVIZ_HEIGHT, ":")
    q = qname.lower()
    # int8 before int4 before int3 before int2 (substring order).
    if "int8" in q:
        return (4, ":")
    if "int4" in q:
        return (2, (":", ":"))
    if "int3" in q:
        return (2, (".", ":"))
    if "int2" in q:
        return (1, ":")
    if "nf4" in q or "fp8" in q:
        return (1, "~")
    if "bf16" in q:
        return (1, "b")
    return (1, (qname[0] if qname else "?").lower())


def _segment_quant_viz_rows(segments, n_tokens, width=_QVIZ_WIDTH):
    """Return 1–8 strings (top first): proportional draft/fp16 timeline; trim all-space top rows."""
    if width <= 0 or n_tokens <= 0 or not segments:
        return []
    total = sum(n for _, n in segments)
    if total <= 0:
        return []
    nrows = _QVIZ_HEIGHT
    rows = [[" "] * width for _ in range(nrows)]
    for k in range(width):
        mid = (k + 0.5) * total / width
        acc = 0.0
        qn = segments[-1][0]
        for q, n in segments:
            if acc + n > mid:
                qn = q
                break
            acc += n
        h, ch_spec = _quant_viz_stack(qn)
        base = nrows - h
        if isinstance(ch_spec, tuple):
            for i in range(h):
                rows[base + i][k] = ch_spec[i]
        else:
            for r in range(base, nrows):
                rows[r][k] = ch_spec
    while len(rows) > 1 and all(rows[0][j] == " " for j in range(width)):
        rows.pop(0)
    return ["".join(r) for r in rows]


def _quant_viz_color_hex(qname: str) -> str:
    """Background color for segment column: light gray (int2) → black (fp16)."""
    if not qname or qname == "fp16":
        return "#000000"
    q = qname.lower()
    if "int8" in q:
        return "#2a2a2a"
    if "int4" in q:
        return "#555555"
    if "int3" in q:
        return "#909090"
    if "int2" in q:
        return "#c8c8c8"
    if "nf4" in q or "fp8" in q:
        return "#707070"
    if "bf16" in q:
        return "#404040"
    return "#808080"


def _hex_to_rgb_triple(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _segment_quant_viz_color_rows(segments, n_tokens, width=_QVIZ_WIDTH):
    """Same geometry as ASCII qviz: list of rows (top first), each cell None or #RGB hex."""
    if width <= 0 or n_tokens <= 0 or not segments:
        return []
    total = sum(n for _, n in segments)
    if total <= 0:
        return []
    nrows = _QVIZ_HEIGHT
    rows = [[None] * width for _ in range(nrows)]
    for k in range(width):
        mid = (k + 0.5) * total / width
        acc = 0.0
        qn = segments[-1][0]
        for q, n in segments:
            if acc + n > mid:
                qn = q
                break
            acc += n
        col = _quant_viz_color_hex(qn)
        h, _ = _quant_viz_stack(qn)
        base = nrows - h
        for r in range(base, nrows):
            rows[r][k] = col
    while len(rows) > 1 and all(rows[0][j] is None for j in range(width)):
        rows.pop(0)
    return rows


def _segment_quant_viz_html(segments, n_tokens, width=_QVIZ_WIDTH, title=""):
    """HTML table: one cell per token column; height = stack; color = quant (gray scale)."""
    color_rows = _segment_quant_viz_color_rows(segments, n_tokens, width=width)
    if not color_rows:
        return ""
    parts = []
    if title:
        parts.append(f'<div class="qviz-title">{html.escape(title)}</div>')
    parts.append('<table class="qviz-grid" style="border-collapse:collapse;border-spacing:0;margin:4px 0;">')
    for row in color_rows:
        parts.append("<tr>")
        for cell in row:
            if cell is None:
                parts.append("""<td style="width:10px;height:10px;padding:0;background:#ffffff;"></td>""")
            else:
                parts.append(
                    f'<td style="width:10px;height:10px;padding:0;background:{cell};"></td>')
        parts.append("</tr>")
    parts.append("</table>")
    parts.append(
        '<p class="qviz-legend" style="font-size:11px;color:#444;margin:0">'
        "Taller = higher precision (fp16); darker = higher precision; "
        "int2 light gray → fp16 black.</p>")
    return "".join(parts)


def _segment_quant_viz_ansi_lines(segments, n_tokens, width=_QVIZ_WIDTH):
    """Return lines for terminal: 24-bit background color per cell, two spaces per column."""
    color_rows = _segment_quant_viz_color_rows(segments, n_tokens, width=width)
    if not color_rows:
        return []
    reset = "\033[0m"
    white = "\033[48;2;255;255;255m"
    lines = []
    for row in color_rows:
        buf = []
        for cell in row:
            if cell is None:
                buf.append(f"{white}  ")
            else:
                r, g, b = _hex_to_rgb_triple(cell)
                buf.append(f"\033[48;2;{r};{g};{b}m  ")
        buf.append(reset)
        lines.append("".join(buf))
    return lines


_QVIZ_LEGEND = (
    "qviz: draft/fp16 (~n_tok/64/col); fp16=:::::::: int8=:::: int4=:: int3=.: int2=:; "
    "verifier quant not in segment names"
)



def _quant_bits_estimate(qspec: str) -> int:
    """Best-effort bit-width estimate for ordering candidate quant specs."""
    if not qspec:
        return 16
    s = qspec.lower()
    # Split k:v and layer-range segments; take the maximum bit-width seen.
    segs = s.replace(":", "/").split("/")
    bits = 0
    for seg in segs:
        if "int2" in seg:
            bits = max(bits, 2)
        elif "int3" in seg:
            bits = max(bits, 3)
        elif "int4" in seg or "nf4" in seg:
            bits = max(bits, 4)
        elif "int8" in seg or "fp8" in seg:
            bits = max(bits, 8)
        elif "bf16" in seg:
            bits = max(bits, 16)
        elif "fp16" in seg:
            bits = max(bits, 16)
    return bits or 16


def _make_cpu_zone_hook(lib, k_names, v_names, n_pos_per_embd, asym=False,
                        quant_fn_factory=None, profile=None):
    """Build a bare CPU hook (no window logic) for a specific quant type.
    Used for sink_hook and recent_hook in _apply_window zone quantization.
    Accepts start_k/start_v for absolute-offset quantization."""
    _fn = quant_fn_factory if quant_fn_factory is not None \
          else (lambda n: quant_mod.get_quant_fn(n, asym=asym))
    k_fns = [_fn(n) for n in k_names]
    v_fns = [_fn(n) for n in v_names]
    def _zone_hook(ctx, n_new_k=None, n_new_v=None,
                   start_k=None, start_v=None, **_):
        parse_state.apply_kv_hook(lib, ctx, llama.ContextPtr,
                                  k_fn=k_fns, v_fn=v_fns,
                                  seq_id=0, n_pos_per_embd=n_pos_per_embd,
                                  n_new_k=n_new_k, n_new_v=n_new_v,
                                  start_k=start_k, start_v=start_v,
                                  profile=profile)
    return _zone_hook


def _apply_window(main_hook, n_sink, n_recent,
                  sink_hook=None, recent_hook=None):
    """Wrap main_hook with three-zone KIVI-style quantization.

    Zones (positions in the KV cache at time T):
      [0,          n_sink)      → sink_hook   (or fp16 if sink_hook is None)
      [n_sink,     T-n_recent)  → main_hook   (stale tokens, heaviest compression)
      [T-n_recent, T)           → recent_hook (or fp16 if recent_hook is None)

    Token lifecycle during decode:
      1. New token added → quantized with recent_hook (if set), else stays fp16
      2. After n_recent more tokens arrive, token falls out of recent window
         → re-quantized with main_hook (from its recent_hook state or fp16)

    When group_size > n_recent some new tokens bypass the recent zone entirely
    (they immediately become stale in the same batch they were added).

    Prefill: all three zones quantized in one shot.
    Caller must pass n_prompt= on the prefill call so zone boundaries are known.

    Adaptive-gen / KV-restore bulk quantize: pass n_prompt=0 and n_seq_len= (number of
    tokens currently in the restored cache). That calls main_hook(n_new_k=None) once,
    which resets per-layer pending and quantizes all cells — required for mixed
    group-size hooks. Do *not* pass a fake n_prompt=prime_pos here: the zone splitter
    mis-partitions the cache and passing stale sizes as n_new_k corrupts pending.
    """
    if (n_sink <= 0 and n_recent <= 0
            and sink_hook is None and recent_hook is None):
        return main_hook

    n_done = [0]  # K and V fire together after the group-size equality fix

    def window_hook(ctx, n_new_k=None, n_new_v=None, n_prompt=0, n_seq_len=None, **_):
        G = n_new_k  # == n_new_v in practice; use a single counter

        if G is None:
            # ── Full-cache bulk (adaptive-gen after restore): no zone split ─────
            if n_prompt == 0:
                main_hook(ctx, n_new_k=None, n_new_v=None)
                if n_seq_len is not None:
                    n_done[0] = n_seq_len
                return
            # ── Prefill with zones (PPL / prompt load) ─────────────────────────
            n_done[0] = n_prompt
            sink_end     = min(n_sink, n_prompt)
            mid_start    = sink_end
            mid_end      = max(mid_start, n_prompt - n_recent) if n_recent > 0 else n_prompt
            recent_start = mid_end

            if sink_hook and sink_end > 0:
                sink_hook(ctx, n_new_k=sink_end, n_new_v=sink_end,
                          start_k=0, start_v=0)
            if mid_end > mid_start:
                main_hook(ctx, n_new_k=mid_end - mid_start, n_new_v=mid_end - mid_start,
                          start_k=mid_start, start_v=mid_start)
            if recent_hook and n_prompt > recent_start:
                count = n_prompt - recent_start
                recent_hook(ctx, n_new_k=count, n_new_v=count,
                            start_k=recent_start, start_v=recent_start)

        else:
            # ── Decode fire: G new tokens ────────────────────────────────────
            n_before  = n_done[0]
            n_done[0] += G

            # Stale zone: tokens that fell out of (or bypassed) the recent window.
            # Old recent window: [n_before - n_recent, n_before)
            # New recent window: [n_before + G - n_recent, n_before + G)
            # Fell out (= should be quantized with main_hook now):
            #   [n_before - n_recent, n_before + G - n_recent) — G tokens
            # Clip against [n_sink, ∞) and against [0, ∞).
            stale_raw_start = n_before - n_recent   # may be negative when window not full
            stale_raw_end   = stale_raw_start + G   # = n_before + G - n_recent
            stale_start = max(stale_raw_start, n_sink, 0)
            stale_count = max(0, stale_raw_end - stale_start)
            if stale_count > 0:
                main_hook(ctx, n_new_k=stale_count, n_new_v=stale_count,
                          start_k=stale_start, start_v=stale_start)

            # Recent zone: tokens that land in the NEW recent window.
            # = last min(G, n_recent) tokens (no start_k needed — default "last N").
            if recent_hook and n_recent > 0:
                recent_count = min(G, n_recent)
                recent_hook(ctx, n_new_k=recent_count, n_new_v=recent_count)

            # No-recent-window fallback: if n_recent==0 stale logic above covers main
            # but stale_raw_start = n_before, stale_count = G (minus sink clip) ✓

    return window_hook


def make_kv_hook(lib, k_names, v_names, n_pos_per_embd=1,
                 use_gpu=False, n_layer=0, ctx_ptr=None,
                 default_group_size=64, n_sink=0, n_recent=0,
                 k_sink_names=None, v_sink_names=None,
                 k_recent_names=None, v_recent_names=None,
                 asym=False, quant_fn_factory=None, profile=None):
    """Create a KV hook for the given K/V quant name lists.

    k_names / v_names: list[str] of length n_layer.
    Returns None if all layers are fp16 (no-op).

    Per-layer group sizes are computed from each layer's quant type:
    - per-token quants → group_size=default_group_size (same firing cadence as per-channel)
    - per-channel _ch_g{N} quants → group_size=N (from the named variant)
    - per-channel quants → group_size=default_group_size
    When layers on the same side have mixed group sizes (e.g. _ch_g32 on layers
    0-15 and _ch_g64 on layers 16-31), a stateful closure tracks each layer's
    pending count independently so the tighter group fires more often.

    Zone quant parameters (k_sink_names etc.): if provided, the sink / recent
    zones use a separate quant type instead of fp16.  These always use the CPU
    path (parse_state) regardless of use_gpu.
    """
    if all(n == "fp16" for n in k_names) and all(n == "fp16" for n in v_names):
        return None

    # Build CPU zone hooks for sink / recent zones (if zone quants requested).
    _fn = quant_fn_factory if quant_fn_factory is not None \
          else (lambda n: quant_mod.get_quant_fn(n, asym=asym))
    sink_hook   = (_make_cpu_zone_hook(lib, k_sink_names,   v_sink_names,   n_pos_per_embd,
                                       quant_fn_factory=_fn, profile=profile)
                   if k_sink_names   is not None else None)
    recent_hook = (_make_cpu_zone_hook(lib, k_recent_names, v_recent_names, n_pos_per_embd,
                                       quant_fn_factory=_fn, profile=profile)
                   if k_recent_names is not None else None)

    def _layer_gs(names):
        gs = []
        for n in names:
            if n in quant_mod.CH_QUANT_GROUP_SIZE:
                gs.append(quant_mod.CH_QUANT_GROUP_SIZE[n])
            else:
                gs.append(default_group_size)
        return gs

    k_layer_gs = _layer_gs(k_names)
    v_layer_gs = _layer_gs(v_names)
    k_uniform  = len(set(k_layer_gs)) == 1
    v_uniform  = len(set(v_layer_gs)) == 1

    if k_uniform and v_uniform:
        # Simple path: no per-layer tracking needed; hook fires at the right cadence
        if use_gpu and HAS_GPU_QUANT:
            return _apply_window(
                gpu_quant.make_kv_hook_gpu(lib, ctx_ptr, k_names, v_names, n_layer,
                                           profile=profile),
                n_sink, n_recent, sink_hook=sink_hook, recent_hook=recent_hook)
        k_fns = [_fn(n) for n in k_names]
        v_fns = [_fn(n) for n in v_names]
        def hook_simple(ctx, n_new_k=None, n_new_v=None,
                        start_k=None, start_v=None, _kf=k_fns, _vf=v_fns, **_):
            parse_state.apply_kv_hook(lib, ctx, llama.ContextPtr,
                                      k_fn=_kf, v_fn=_vf,
                                      seq_id=0, n_pos_per_embd=n_pos_per_embd,
                                      n_new_k=n_new_k, n_new_v=n_new_v,
                                      start_k=start_k, start_v=start_v,
                                      profile=profile)
        return _apply_window(hook_simple, n_sink, n_recent,
                             sink_hook=sink_hook, recent_hook=recent_hook)

    # Mixed granularities: stateful per-layer pending counters.
    # Each layer fires independently when its own threshold is met.
    k_pending = [0] * n_layer
    v_pending = [0] * n_layer

    def _per_layer_new(n_new, pending, layer_gs):
        """Accumulate n_new into pending; return per-layer list.
        None = quantize all (prefill), 0 = skip, int = quantize last N cells."""
        result = []
        for i in range(n_layer):
            if n_new is None:           # prefill: reset and quantize all
                pending[i] = 0
                result.append(None)
            else:
                pending[i] += n_new
                if pending[i] >= layer_gs[i]:
                    result.append(pending[i])
                    pending[i] = 0
                else:
                    result.append(0)    # not enough accumulated yet
        return result

    if use_gpu and HAS_GPU_QUANT:
        def hook_gpu_mixed(ctx, n_new_k=None, n_new_v=None, **_):
            k_per = _per_layer_new(n_new_k, k_pending, k_layer_gs)
            v_per = _per_layer_new(n_new_v, v_pending, v_layer_gs)
            if n_new_k is None or any(k_per) or any(v_per):
                gpu_quant.apply_kv_hook_gpu(
                    lib, ctx, n_layer, k_names, v_names,
                    n_new_k=None if n_new_k is None else k_per,
                    n_new_v=None if n_new_v is None else v_per,
                    profile=profile)
        return _apply_window(hook_gpu_mixed, n_sink, n_recent,
                             sink_hook=sink_hook, recent_hook=recent_hook)

    k_fns = [_fn(n) for n in k_names]
    v_fns = [_fn(n) for n in v_names]
    def hook_cpu_mixed(ctx, n_new_k=None, n_new_v=None, **_):
        k_per = _per_layer_new(n_new_k, k_pending, k_layer_gs)
        v_per = _per_layer_new(n_new_v, v_pending, v_layer_gs)
        if n_new_k is None or any(k_per) or any(v_per):
            parse_state.apply_kv_hook(lib, ctx, llama.ContextPtr,
                                      k_fn=k_fns, v_fn=v_fns,
                                      seq_id=0, n_pos_per_embd=n_pos_per_embd,
                                      n_new_k=None if n_new_k is None else k_per,
                                      n_new_v=None if n_new_v is None else v_per,
                                      profile=profile)
    return _apply_window(hook_cpu_mixed, n_sink, n_recent,
                         sink_hook=sink_hook, recent_hook=recent_hook)


def get_kv_group_sizes(k_names, v_names, default_group_size):
    """Return (k_group_size, v_group_size) for the given K/V quant name lists.

    Priority order for each side:
      1. Named _ch_g{N} quant present → group_size = N (from CH_QUANT_GROUP_SIZE)
      2. Otherwise (including per-token quants) → default_group_size

    Per-token quants fire at the same cadence as per-channel (every
    default_group_size tokens) for a fair quality comparison.  The quant
    function itself still operates per-token (axis=1); only the hook firing
    frequency changes.

    For mixed-layer specs (e.g. int4_ch_g32@0-15/int4_ch_g64@16-31) the
    minimum group size across layers is used so the hook fires often enough
    for the tightest group.
    """
    def _resolve(names):
        named_gs = [quant_mod.CH_QUANT_GROUP_SIZE[n]
                    for n in names if n in quant_mod.CH_QUANT_GROUP_SIZE]
        if named_gs:
            return min(named_gs)
        return default_group_size

    return _resolve(k_names), _resolve(v_names)


# Task-specific stop strings keyed by corpus filename keywords.
# These are safety-net stops for pathological outputs (fake follow-on problems,
# role leakage, verbosity past the answer). Appended automatically; never override
# user-specified stops.
_TASK_STOP_RULES = [
    # AIME / math competitions: stop at fake follow-on problem.
    # \n\n--- intentionally omitted: gpt-oss uses --- as markdown HR in solutions,
    # and 0-shot AIME has no --- separator in the prompt to teach fake continuation.
    (["aime"],                      ["\n\nQuestion:"]),
    # GSM8K: stop at fake follow-on question
    (["gsm8k"],                     ["\n\nQuestion:"]),
    # NIAH: answer is a single compound word — stop at first newline
    (["niah"],                      ["\n"]),
    # HumanEval / code: stop at next top-level function or __main__ guard
    (["humaneval", "human_eval"],   ["\n\ndef ", "\n\nif __name__"]),
]


def _failure_summary(gl: dict, n_total: int) -> str:
    """Build a compact failure-mode string from gen_len_stats for the summary print line.

    Shows non-zero failure counts only. For exact-match tasks:
      correct=7  wrong=3  trunc=2  no_match=1
    For f1/code tasks (counters are 0): shows trunc count only if any.
    """
    parts = []
    if gl.get("n_correct") or gl.get("n_wrong") or gl.get("n_trunc_fail") or gl.get("n_no_match"):
        parts.append(f"correct={gl['n_correct']}")
        if gl["n_wrong"]:      parts.append(f"wrong={gl['n_wrong']}")
        if gl["n_trunc_fail"]: parts.append(f"trunc={gl['n_trunc_fail']}")
        if gl["n_no_match"]:   parts.append(f"no_match={gl['n_no_match']}")
        if gl.get("n_inconclusive"): parts.append(f"inconclusive={gl['n_inconclusive']}")
    elif gl.get("n_truncated"):
        parts.append(f"trunc={gl['n_truncated']}/{n_total}")
    return ("  " + "  ".join(parts)) if parts else ""


def _infer_task_stops(filename: str) -> list:
    """Return task-specific stop strings inferred from the corpus filename."""
    name = filename.lower()
    for keywords, stops in _TASK_STOP_RULES:
        if any(kw in name for kw in keywords):
            return stops
    return []


def main():
    parser = argparse.ArgumentParser(description="KV cache quantization perplexity sweep")
    parser.add_argument("model")
    parser.add_argument("corpus_file",
                        help=".txt for flat mode, .jsonl for structured mode")
    parser.add_argument("--corpus-mode",    default="flat",
                        choices=["flat", "structured"],
                        help="flat: sliding-window over .txt (default). "
                             "structured: per-example (prompt,completion) from .jsonl")
    parser.add_argument("--n-ctx",          type=int, default=128,
                        help="Context window size. In structured mode: cap on "
                             "prompt+completion tokens per example (excess is truncated).")
    parser.add_argument("--n-chunks",       type=int, default=20,
                        help="Flat mode: number of non-overlapping chunks. "
                             "Structured mode: max examples to evaluate (0=all).")
    parser.add_argument("--n-threads",      type=int, default=8)
    parser.add_argument("--n-gpu-layers",   type=int, default=0)
    parser.add_argument("--flash-attn",     action="store_true",
                        help="Enable Flash Attention (required for GPU-side KV quantization).")
    parser.add_argument("--n-pos-per-embd", type=int, default=1,
                        help="Set to 4 for M-RoPE models (e.g. Qwen-VL)")
    parser.add_argument("--n-prompt",       type=int, default=0,
                        help="Flat mode only: leading tokens to treat as prompt. "
                             "In structured mode the prompt length comes from the JSONL.")
    parser.add_argument("--prefill-tokens", type=int, default=0,
                        help="Fixed prefill size for multi-window mode. "
                             "Use with --score-windows. Sets n_ctx = prefill + max(windows).")
    parser.add_argument("--score-windows",  type=int, nargs="+", default=None,
                        help="Multi-window scoring: decode max(windows) tokens once, "
                             "compute PPL at each cutoff (e.g. 512 1024 2048). "
                             "Requires --prefill-tokens. Replaces --n-prompt/--n-ctx.")
    parser.add_argument("--prefill-batch",  action="store_true",
                        help="Flat mode: batch-prefill the prompt. "
                             "Structured mode always uses batch-prefill.")
    parser.add_argument("--quantize-prompt-only", action="store_true",
                        help="Strategy D: quantize prompt KV only, decode stays FP16.")
    parser.add_argument("--quant-group-size",   type=int, default=64,
                        help="Tokens per quantization group during decode (default 128). "
                             "Scales are shared across the group, matching real KV quant systems. "
                             "Larger = more compression error but fewer scales stored.")
    parser.add_argument("--sink-tokens",        type=int, default=0,
                        help="Leave the first N tokens unquantized (attention sinks). "
                             "Papers show initial tokens absorb disproportionate attention mass; "
                             "keeping them in fp16 often recovers quality at minimal memory cost. "
                             "With --adaptive-gen/--adaptive-sim, zone hooks track a global "
                             "decode counter that is not reset on KV restore — use 0 unless you "
                             "understand the interaction.")
    parser.add_argument("--recent-tokens",      type=int, default=0,
                        help="Keep the most recent N tokens in fp16 at all times (KIVI-style). "
                             "When new tokens arrive and older ones fall out of this window, "
                             "they get quantized. Combined with --sink-tokens this gives three "
                             "zones: sinks (fp16) | stale (quantized) | recent (fp16). "
                             "Same KV-restore caveat as --sink-tokens for adaptive modes.")
    parser.add_argument("--quant-sink",         default=None,
                        help="Quant type for the sink zone instead of fp16 "
                             "(e.g. 'int8_ch'). Supports K:V split and layer-range syntax. "
                             "Has no effect when --sink-tokens is 0.")
    parser.add_argument("--quant-recent",       default=None,
                        help="Quant type for the recent-window zone instead of fp16 "
                             "(e.g. 'int8_ch'). Tokens are quantized immediately when added, "
                             "then re-quantized with the main quant when they become stale. "
                             "Has no effect when --recent-tokens is 0.")
    parser.add_argument("--show-text",      action="store_true",
                        help="Print detokenized prompt/completion and fp16 top-1 predictions.")
    parser.add_argument("--show-gen",       action="store_true",
                        help="Print the full generated text for each accuracy example (raw, not repr). "
                             "Useful for inspecting CoT reasoning without the prompt noise.")
    parser.add_argument("--show-prompt",    action="store_true",
                        help="Print the detokenized prompt for each accuracy example (raw). "
                             "Combine with --show-gen to see prompt + generation together.")
    parser.add_argument("--save-diags", default=None, metavar="FILE",
                        help="Save per-token diagnostics (H, p_max, self_surp, lp) to JSON. "
                             "Default: <out>_diags.json (auto-derived from --out).")
    parser.add_argument("--save-per-example", default=None, metavar="FILE",
                        help="Save per-example accuracy scores and gen lengths to JSON. "
                             "Default: <out>_per_example.json (auto-derived from --out).")
    parser.add_argument("--show-text-chunk", type=int, default=None,
                        help="Which chunk/example index to show token predictions for. "
                             "Default: show all. Pass a non-negative int to show only that index.")
    parser.add_argument("--eval-accuracy",  action="store_true",
                        help="Structured mode: run greedy generation and compare extracted "
                             "answer against gold. Reports accuracy alongside PPL/KL.")
    parser.add_argument("--save-bins",      default=None, metavar="FILE",
                        help="Save per-bin hit counts to FILE (JSON). "
                             "Default: <out>_bins.json (auto-derived from --out). "
                             "Use plot_bins.py to visualize.")
    parser.add_argument("--asym",           action="store_true",
                        help="Use asymmetric (affine) quantization: stores min+scale per group "
                             "instead of just scale. Gives 2^bits levels instead of 2^(bits-1). "
                             "For int2: 4 bins {min,…,max} vs 3 bins {-s,0,+s}. "
                             "Has no effect on fp16/bf16/fp8/nf4.")
    parser.add_argument("--skip-ppl",       action="store_true",
                        help="Skip teacher-forced PPL and KL computation. Only runs "
                             "--eval-accuracy generation. Auto-enabled when --corpus-mode "
                             "structured + --eval-accuracy are both set (use --no-skip-ppl "
                             "to override). Requires --eval-accuracy.")
    parser.add_argument("--no-skip-ppl",    action="store_true",
                        help="Force PPL computation even in structured+eval-accuracy mode "
                             "(overrides the auto-enable of --skip-ppl).")
    parser.add_argument("--max-gen-tokens", type=int, default=512,
                        help="Max new tokens to generate per example for --eval-accuracy (default 512).")
    parser.add_argument("--answer-regex",
                        default=r"(?:####|[Tt]he answer is)\s*\$?\s*\*{0,2}\s*([\d,]+)\s*\*{0,2}|\\boxed\{([\d,]+)\}",
                        help="Regex with capture group(s) to extract the answer from generated "
                             "text. Matches '#### 42', 'The answer is 42', 'The answer is **42**', "
                             "and '\\boxed{42}' formats. Ignored when --eval-metric f1.")
    parser.add_argument("--eval-metric",   default="exact", choices=["exact", "f1", "code"],
                        help="How to compare generated answer to gold. "
                             "'exact': regex extraction + string match (default, good for GSM8K). "
                             "'f1': token-overlap F1 against all gold answers in jsonl 'answers' field "
                             "(better for LongBench QA where multiple valid answers exist). "
                             "'code': save full generated text as pred, score=0 placeholder; "
                             "run eval_code.py afterwards for pass@1 (HumanEval).")
    parser.add_argument("--stop-strings",  nargs="*", default=["\n\nQuestion:", "\nassistant", "\n\n\n"],
                        help="Stop generation when any of these strings appear in the output. "
                             "Default includes '\\nassistant' to prevent chat role token leakage.")
    parser.add_argument("--no-task-stops", action="store_true", default=False,
                        help="Disable auto-inferred task-specific stop strings (useful for debugging).")
    parser.add_argument("--prompt-prefix", default=None,
                        help="String prepended to every example prompt before tokenization. "
                             "If omitted, auto-detected from the model vocabulary when --eval-accuracy "
                             "is set. Pass an empty string ('') to suppress auto-detection.")
    parser.add_argument("--prompt-suffix", default=None,
                        help="String appended to every example prompt before tokenization. "
                             "If omitted, auto-detected from the model vocabulary when --eval-accuracy "
                             "is set. Pass an empty string ('') to suppress auto-detection.")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high"],
                        help="GPT-OSS reasoning effort injected into the system message "
                             "(low/medium/high). Only applies when auto-detecting gpt-oss format "
                             "(i.e. --prompt-prefix not set explicitly). Ignored for other models.")
    parser.add_argument("--adaptive-sim",   action="store_true",
                        help="Simulate adaptive quantization: run draft quant (--quants entry) and "
                             "--verifier-quant in parallel, compare window-by-window. "
                             "Records acceptance_rate, first_fail_window, draft_fraction per example. "
                             "Requires --eval-accuracy --skip-ppl --corpus-mode structured.")
    parser.add_argument("--verifier-quant", default="int8_ch",
                        help="Verifier KV quant: --adaptive-sim (draft vs verifier windows) and "
                             "--adaptive-gen (replay each quant rollout window under this quant). "
                             "Default int8_ch. Use fp16 for strict fp16-teacher verification. "
                             "Draft sweep uses each --quants entry (excluding fp16 and this verifier).")
    parser.add_argument("--adaptive-gen",    action="store_true",
                        help="Real adaptive generation: bootstrap window (fp16 generates W tokens); "
                             "then quant rollout windows (draft quant generates W tokens, verifier "
                             "replays; accept or fp16 fallback). Requires --eval-accuracy --skip-ppl.")
    parser.add_argument("--adaptive-window", type=int, default=8,
                        help="Rollout window size W (tokens) for each quant draft / fp16 verify "
                             "cycle in --adaptive-sim/--adaptive-gen (default: 8).")
    parser.add_argument("--bootstrap-window", type=int, default=32,
                        help="Bootstrap window size W_b (tokens): fp16 generates W_b tokens in "
                             "window 0, candidates verify all W_b for bootstrap pass/fail "
                             "(default: 32). Separate from --adaptive-window.")
    parser.add_argument("--adaptive-verify-top-k", type=int, default=None, metavar="K",
                        help="For --adaptive-sim/--adaptive-gen: accept draft tokens that appear "
                             "in the verifier's top-K (per position). Mutually exclusive with "
                             "--adaptive-verify-top-p. Default: greedy (argmax) match.")
    parser.add_argument("--adaptive-verify-top-p", type=float, default=None, metavar="P",
                        help="For --adaptive-sim/--adaptive-gen: nucleus top-p acceptance on "
                             "verifier logits at each position (0<P<=1). Mutually exclusive with "
                             "--adaptive-verify-top-k.")
    parser.add_argument("--no-adaptive-gen-gpu-shadow", action="store_true",
                        help="For --adaptive-gen: use CPU KV blob save/restore (PCIe on GPU). "
                             "Default: GPU device-to-device shadow checkpoint via CuPy when "
                             "GPU KV is active.")
    parser.add_argument("--qviz-html", default=None, metavar="FILE",
                        help="With --adaptive-gen: write grayscale HTML segment qviz (tall=dark, "
                             "short=light) to FILE.")
    parser.add_argument("--qviz-ansi", action="store_true",
                        help="With --adaptive-gen: print qviz using ANSI 24-bit background colors.")
    parser.add_argument("--profile-kv", action="store_true",
                        help="Time llama_decode vs KV quant hook. CPU path: get/parse/quant/pack/set "
                             "(parse_state). GPU path (--flash-attn + CuPy): gpu_kv_s includes "
                             "in-place kernels + cuda synchronize. Per-quant summary; --profile-kv-out JSON.")
    parser.add_argument("--profile-kv-out", default=None, metavar="FILE",
                        help="Write per-quant profile dicts as JSON (quant name -> timings).")
    parser.add_argument("--out",            default="results.json")
    parser.add_argument("--quants",    nargs="+",
                        default=["fp16","bf16","fp8_e4m3","fp8_e5m2","int8","int8_ch","int4","int4_ch","nf4","int2"])
    parser.add_argument("--quant-k",   default=None,
                        help="K-cache quant spec added as one extra sweep entry alongside --quants. "
                             "Supports layer ranges: 'int8_ch@0-15/int4_ch@16-31'. "
                             "Combined with --quant-v (defaults to fp16 if omitted).")
    parser.add_argument("--quant-v",   default=None,
                        help="V-cache quant spec added as one extra sweep entry alongside --quants. "
                             "Supports layer ranges: 'int8_tok@0-15/int4_tok@16-31'. "
                             "Combined with --quant-k (defaults to fp16 if omitted).")
    args = parser.parse_args()

    if args.adaptive_verify_top_k is not None and args.adaptive_verify_top_k < 1:
        parser.error("--adaptive-verify-top-k must be >= 1")
    if args.adaptive_verify_top_p is not None:
        if not (0.0 < args.adaptive_verify_top_p <= 1.0):
            parser.error("--adaptive-verify-top-p must be in (0, 1]")
    if args.adaptive_verify_top_k is not None and args.adaptive_verify_top_p is not None:
        parser.error("use only one of --adaptive-verify-top-k and --adaptive-verify-top-p")
    # Merge --quant-k / --quant-v into the quants list as one extra entry
    if args.quant_k is not None or args.quant_v is not None:
        k_spec = args.quant_k or "fp16"
        v_spec = args.quant_v or "fp16"
        extra  = f"{k_spec}:{v_spec}"
        if extra not in args.quants:
            args.quants = list(args.quants) + [extra]

    # ── Auto-derive sibling output paths from --out ───────────────────────────
    def _sibling(suffix):
        base = args.out[:-5] if args.out.endswith(".json") else args.out
        return base + suffix

    if args.save_bins        is None: args.save_bins        = _sibling("_bins.json")
    if args.save_per_example is None: args.save_per_example = _sibling("_per_example.json")
    if args.save_diags       is None: args.save_diags       = _sibling("_diags.json")
    if args.profile_kv and args.profile_kv_out is None:
        args.profile_kv_out = _sibling("_kv_profile.json")

    # ── Structured corpus implies skip-ppl by default ────────────────────────
    # Teacher-forced PPL is meaningless for QA/reasoning tasks (GSM8K, NIAH,
    # LongBench): the model copies reference answers token-by-token regardless
    # of whether it can actually generate them. Auto-enable --skip-ppl unless
    # the user explicitly passed --no-skip-ppl to override.
    if args.corpus_mode == "structured" and args.eval_accuracy and not args.no_skip_ppl:
        args.skip_ppl = True

    # ── Tee stdout+stderr to a .log file alongside the .json output ──────────
    log_path = _sibling(".log")
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, stream, f): self._s, self._f = stream, f
        def write(self, d):  self._s.write(d);  self._f.write(d)
        def flush(self):     self._s.flush();   self._f.flush()
        def fileno(self):    return self._s.fileno()

    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    print(f"command : {' '.join(sys.argv)}", flush=True)
    print(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"log     : {log_path}", flush=True)

    import atexit
    def _close_log():
        print(f"\nfinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()
    atexit.register(_close_log)

    lib = llama.load_lib()
    kv_prof = None
    profile_agg = {}
    if args.profile_kv:
        kv_prof = kv_profile.KvProfile()
        kv_profile.wrap_llama_decode(lib, kv_prof)
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)
    n_layer = lib.llama_model_n_layer(model)

    # End-of-generation check — stops generation when the model emits any token
    # that the model author marked as EOG (EOS, EOT, or model-specific variants).
    # These tokens typically detokenize to empty string, so stop-string matching
    # would never catch them. llama_vocab_is_eog covers all cases in one call.
    def _is_eog(token_id: int) -> bool:
        return bool(lib.llama_vocab_is_eog(vocab, token_id))

    # Auto-detect chat format and apply to --prompt-prefix / --prompt-suffix
    # when neither was set explicitly (both are None) and --eval-accuracy is active.
    # Passing --prompt-prefix "" explicitly suppresses auto-detection.
    _fmt_name = "unknown"   # captured for post-processing below
    if args.eval_accuracy and args.prompt_prefix is None and args.prompt_suffix is None:
        auto_prefix, auto_suffix, auto_stop, fmt_name = llama.detect_chat_format(lib, vocab)
        _fmt_name = fmt_name
        if auto_prefix is not None:
            args.prompt_prefix = auto_prefix
            args.prompt_suffix = auto_suffix
            if auto_stop and auto_stop not in args.stop_strings:
                args.stop_strings = list(args.stop_strings) + [auto_stop]
            # gpt-oss: replace bare user prefix with full system message when --effort is set.
            if fmt_name == "gpt-oss" and args.effort is not None:
                args.prompt_prefix = (
                    f"<|start|>system<|message|>Reasoning: {args.effort}\n\n"
                    f"# Valid channels: analysis, commentary, final. "
                    f"Channel must be included for every message.<|end|>\n"
                    f"<|start|>user<|message|>"
                )
            elif args.effort is not None:
                print(f"[auto] --effort ignored: only applies to gpt-oss (detected: {fmt_name})", flush=True)
            print(f"[auto] detected chat format: {fmt_name}", flush=True)
            if args.effort is not None and fmt_name == "gpt-oss":
                print(f"[auto] gpt-oss reasoning effort: {args.effort}", flush=True)
            print(f"[auto] prompt-prefix: {repr(args.prompt_prefix)}", flush=True)
            print(f"[auto] prompt-suffix: {repr(args.prompt_suffix)}", flush=True)
            print(f"[auto] stop-string added: {repr(auto_stop)}", flush=True)
        else:
            print("[auto] chat format: unknown — prompt-prefix/suffix left empty", flush=True)

    # Normalize: ensure prefix/suffix are strings (not None) for all downstream code.
    if args.prompt_prefix is None:
        args.prompt_prefix = ""
    if args.prompt_suffix is None:
        args.prompt_suffix = ""

    # Auto-append task-specific stop strings inferred from the corpus filename.
    # Always appended (never removes user-specified stops); only when --eval-accuracy.
    if args.eval_accuracy and not args.no_task_stops:
        _task_stops = _infer_task_stops(os.path.basename(args.corpus_file))
        _added = [s for s in _task_stops if s not in args.stop_strings]
        if _added:
            args.stop_strings = list(args.stop_strings) + _added
            print(f"[auto] task stops added: {[repr(s) for s in _added]}", flush=True)

    # Qwen3 thinking models: \n stop fires inside <think> before the answer is emitted.
    # When auto-detected as qwen/chatml AND <think> is a special token AND \n is a stop,
    # append an empty think block to the suffix so the model skips thinking entirely.
    # Only applies to auto-detected format; explicit --prompt-suffix overrides this.
    if (_fmt_name == "qwen/chatml"
            and "\n" in args.stop_strings
            and llama._probe_special(lib, vocab, "<think>")):
        args.prompt_suffix = args.prompt_suffix.rstrip("\n") + "<think>\n\n</think>\n"
        print("[auto] qwen3 thinking suppressed: empty think block injected into suffix", flush=True)

    # Resolve zone quant names once (same n_layer for all sweep entries).
    k_sink_names, v_sink_names = (
        quant_mod.resolve_quant_layers(args.quant_sink, n_layer)
        if args.quant_sink else (None, None))
    k_recent_names, v_recent_names = (
        quant_mod.resolve_quant_layers(args.quant_recent, n_layer)
        if args.quant_recent else (None, None))

    def show_chunk_text(prompt_tokens, decode_tokens, label=""):
        """Print full detokenized prompt and decode target."""
        prompt_text = llama.detokenize(lib, vocab, prompt_tokens)
        decode_text = llama.detokenize(lib, vocab, decode_tokens)
        print(f"  [text{' ' + label if label else ''}]", flush=True)
        print(f"    prompt ({len(prompt_tokens)} tokens):", flush=True)
        print(prompt_text, flush=True)
        print(f"    decode target ({len(decode_tokens)} tokens):", flush=True)
        print(decode_text, flush=True)

    # decode_tokens: the full decode window token ids (chunk[n_prefill:] or ct)
    # actual[t] = decode_tokens[t+1]  (model reads decode_tokens[t], predicts decode_tokens[t+1])
    # Per-chunk/example token prediction data.
    # _text_per_chunk[idx] = {"actual_ids": [...], "top1s": {q: [...]},
    #                          "log_probs": {q: [...]}, "diags": {q: {...}}}
    _text_per_chunk = {}

    # Per-example accuracy results for --save-per-example (NIAH / position analysis).
    # _per_example_results[quant] = [{"label", "score", "gold", "pred", "gen_len"}, ...]
    _per_example_results = {}

    def _show_this(idx, total):
        """Return True if this chunk/example index should have its data collected."""
        if args.show_text_chunk is not None:
            target = args.show_text_chunk % total if args.show_text_chunk < 0 else args.show_text_chunk
            return idx == target
        return True  # default: collect all

    def _want_diags(idx, total):
        """Return True if diagnostics should be collected for this chunk/example index."""
        return (args.show_text or bool(args.save_diags)) and _show_this(idx, total)

    def show_predictions():
        """Print a table of actual vs predicted tokens for all collected quants.

        Each quant has five sub-columns (all computed without needing ground truth
        except lp):
          token : '-' when prediction matches actual; predicted piece when wrong
          lp    : log-prob of the correct token (needs ground truth; near 0 = confident)
          H     : output entropy in nats (low = sharp/confident, high = confused)
          p     : probability of top-1 prediction (model's own confidence)
          ss    : self-surprisal — log-prob of the previous step's top-1 under the
                  current distribution; drops when model's own continuations become
                  incoherent (NaN for step 0)
        """
        if not _text_per_chunk:
            return

        def _f(v, w, d):
            return f"{'—':>{w}}" if v != v else f"{v:>{w}.{d}f}"

        def _avg(lst):
            return sum(lst) / len(lst) if lst else float("nan")

        tok_w = 12
        lp_w  =  6
        h_w   =  5
        pm_w  =  5
        ss_w  =  6
        act_w = 12
        grp_w = tok_w + 1 + lp_w + 1 + h_w + 1 + pm_w + 1 + ss_w

        for chunk_idx in sorted(_text_per_chunk.keys()):
            cdata       = _text_per_chunk[chunk_idx]
            actual_ids  = cdata["actual_ids"]
            top1s_map   = cdata["top1s"]
            lp_map      = cdata["log_probs"]
            diag_map    = cdata["diags"]
            quant_names = list(top1s_map.keys())

            h1 = f"  {'step':>4}  {'actual':{act_w}}"
            h2 = f"  {'':>4}  {'':>{act_w}}"
            for q in quant_names:
                h1 += f"  {q:{grp_w}}"
                h2 += f"  {'token':{tok_w}} {'lp':>{lp_w}} {'H':>{h_w}} {'p':>{pm_w}} {'ss':>{ss_w}}"
            sep = "  " + "-" * (len(h1) - 2)

            print(f"\n  [token predictions — chunk/ex {chunk_idx}, {len(actual_ids)} steps]", flush=True)
            print(h1, flush=True)
            print(h2, flush=True)
            print(sep, flush=True)

            buckets = {q: {"match": {"lp":[],"H":[],"p_max":[],"ss":[]},
                            "miss":  {"lp":[],"H":[],"p_max":[],"ss":[]}}
                       for q in quant_names}

            for t, actual_id in enumerate(actual_ids):
                actual_piece = llama.token_to_piece(lib, vocab, actual_id).replace("\n", "↵")
                row = f"  {t:>4}  {repr(actual_piece):{act_w}}"
                for q in quant_names:
                    top1s = top1s_map.get(q, [])
                    lps   = lp_map.get(q, [])
                    diags = diag_map.get(q, {})
                    if t < len(top1s):
                        lp    = lps[t]   if t < len(lps)                        else float("nan")
                        H     = diags["H"][t]         if "H"         in diags and t < len(diags["H"])         else float("nan")
                        p_max = diags["p_max"][t]     if "p_max"     in diags and t < len(diags["p_max"])     else float("nan")
                        ss    = diags["self_surp"][t] if "self_surp" in diags and t < len(diags["self_surp"]) else float("nan")
                        matched  = top1s[t] == actual_id
                        tok_cell = "-" if matched else repr(
                            llama.token_to_piece(lib, vocab, top1s[t]).replace("\n", "↵"))
                        row += (f"  {tok_cell:{tok_w}} {_f(lp,lp_w,1)}"
                                f" {_f(H,h_w,2)} {_f(p_max,pm_w,3)} {_f(ss,ss_w,1)}")
                        bkt = buckets[q]["match" if matched else "miss"]
                        for key, val in (("lp", lp), ("H", H), ("p_max", p_max), ("ss", ss)):
                            if val == val:
                                bkt[key].append(val)
                    else:
                        row += f"  {'—':{tok_w}} {'':>{lp_w}} {'':>{h_w}} {'':>{pm_w}} {'':>{ss_w}}"
                print(row, flush=True)

            q_w   = max(len(q) for q in quant_names)
            n_max = max((len(buckets[q][b]["lp"])
                         for q in quant_names for b in ("match", "miss")), default=0)
            n_w   = len(str(max(n_max, 1)))
            lbl_w = 9 + 3 + n_w
            sh = (f"  {'quant':{q_w}}  {'bucket':{lbl_w}}"
                  f"  {'lp':>{lp_w}} {'H':>{h_w}} {'p':>{pm_w}} {'ss':>{ss_w}}")
            print(f"\n  [summary — averages by match/miss]", flush=True)
            print(sh, flush=True)
            print("  " + "-" * (len(sh) - 2), flush=True)
            for q in quant_names:
                for bucket_name, word in (("match", "matched"), ("miss", "unmatched")):
                    bkt = buckets[q][bucket_name]
                    n   = len(bkt["lp"])
                    tag = f"{word:{9}} N={n:{n_w}}"
                    print(f"  {q:{q_w}}  {tag:{lbl_w}}"
                          f"  {_f(_avg(bkt['lp']),  lp_w, 2)}"
                          f" {_f(_avg(bkt['H']),     h_w,  2)}"
                          f" {_f(_avg(bkt['p_max']), pm_w, 3)}"
                          f" {_f(_avg(bkt['ss']),    ss_w, 2)}", flush=True)

    def save_results(results):
        """Write results JSON, appending incrementally after each quant."""
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (saved → {args.out})", flush=True)

    def _f1_score(prediction, gold_list):
        """Token-overlap F1 between prediction and best-matching gold answer (0.0–1.0).
        Standard LongBench evaluation metric.
        """
        import collections
        def _toks(s):
            s = s.lower()
            s = re.sub(r'[^\w\s]', ' ', s)
            return s.split()
        pred_toks = _toks(prediction)
        if not pred_toks:
            return 0.0
        best = 0.0
        for gold in gold_list:
            gold_toks = _toks(gold)
            if not gold_toks:
                continue
            common    = collections.Counter(pred_toks) & collections.Counter(gold_toks)
            n_same    = sum(common.values())
            if n_same == 0:
                continue
            precision = n_same / len(pred_toks)
            recall    = n_same / len(gold_toks)
            f1        = 2 * precision * recall / (precision + recall)
            best      = max(best, f1)
        return best

    _HEDGE_RE = re.compile(
        r'\b(wait|actually|hold on|let me (?:re)?check|let me recalculate|'
        r'but (?:wait|actually)|hmm+|i need to (?:re)?check|that\'?s not right|'
        r'let me redo|i made an error|correction|i(?:\'m| am) not sure|'
        r'let me (?:try|re-?do|go back|re-?compute|re-?calculate))\b',
        re.IGNORECASE
    )

    # Fallback: last standalone number (possibly with commas/dollar sign).
    # Used when the structured answer regex finds no match.
    _LAST_NUM_RE = re.compile(r'[\$]?\s*\*{0,2}([\d,]+)\*{0,2}')

    def run_adaptive_sim(examples, draft_hook, verifier_hook,
                         draft_k_gs, draft_v_gs, ver_k_gs, ver_v_gs):
        """Simulate the adaptive KV quantization scheme window-by-window.

        Scheme (per example):
          Phase 1 — fp16 generates all tokens autoregressively, saving the KV
            state (and last token as a re-prime key) at every W-token boundary.
          Phase 2 — for each window w:
            a. Restore the fp16 KV state at window start (simulates KV overwrite).
            b. Re-prime: feed the last token before the window to recover logits
               for the first token of the window (fp16 KV, no hook).
            c. Check: does int4 (draft_hook) also predict fp16_tokens[w_start]?
            d. Feed fp16_tokens[w_start .. w_end-2] with draft_hook (verify_window),
               check each greedy matches the next fp16 token.
            e. Window accepted iff ALL predictions match.

        The state restore in (a) simulates the KV overwrite after each fp16 prefill
        verification: int4 always starts from fp16-corrected KV, never from drifting
        int4 KV.  Real acceptance rates are therefore higher than the old post-hoc
        comparison (which let both runs drift independently).

        Returns a list of per-example dicts with keys:
          n_tokens_fp16, n_windows, n_accepted_windows,
          first_fail_window, first_fail_pos,
          acceptance_rate, draft_fraction.
        """
        W = args.adaptive_window
        n_prompt_base = 0   # will be set per example
        sim_results = []

        for ei, (pt, ct, label) in enumerate(examples):
            max_ctx_avail = args.n_ctx - len(pt) - 1
            max_gen = min(args.max_gen_tokens, max_ctx_avail)
            if max_gen <= 0:
                sim_results.append(None)
                continue

            if args.show_prompt and _show_this(ei, len(examples)):
                prompt_text = llama.detokenize(lib, vocab, pt)
                print(f"\n--- prompt [{label}] ({len(pt)} tokens) ---", flush=True)
                print(prompt_text, flush=True)
                print("---", flush=True)

            # ── Phase 1: fp16 generation with KV-state checkpoints ───────────
            mem = lib.llama_get_memory(ctx)
            lib.llama_memory_clear(mem, True)

            # Batch-prefill prompt[:-1] only; save kv0 BEFORE the last prompt
            # token so that re-prime can decode at position n_pt-1 (X+1 rule).
            n_pt = len(pt)
            if n_pt > 1:
                batch = lib.llama_batch_init(n_pt - 1, 0, 1)
                batch.n_tokens = n_pt - 1
                for i, tok in enumerate(pt[:-1]):
                    batch.token[i]     = tok
                    batch.pos[i]       = i
                    batch.n_seq_id[i]  = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i]    = 0
                if lib.llama_decode(ctx, batch) != 0:
                    lib.llama_batch_free(batch)
                    sim_results.append(None)
                    continue
                lib.llama_batch_free(batch)

            # boundary_states[w] = (kv_blob, prime_token, prime_pos)
            # prime_pos is NOT yet in KV; re-prime decodes it at X+1.
            # Window 0: KV has 0..n_pt-2; prime decodes pt[-1] at n_pt-1.
            kv0 = strategies.save_kv_state(lib, ctx)
            boundary_states = [(kv0, pt[-1], n_pt - 1)]

            # Decode last prompt token to get logits for first generated token.
            ret = strategies._single_decode(lib, ctx, pt[-1], n_pt - 1)
            if ret != 0:
                sim_results.append(None)
                continue

            # Generate fp16 tokens, save boundary state every W tokens.
            ptr    = lib.llama_get_logits_ith(ctx, 0)
            logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
            token  = int(np.argmax(logits))
            fp16_tokens = [token]
            pos = n_pt

            for _ in range(max_gen - 1):
                if _is_eog(token):
                    break
                if args.stop_strings:
                    tail = fp16_tokens[-128:]
                    tail_text = llama.detokenize(lib, vocab, tail, remove_special=False)
                    if any(s in tail_text for s in args.stop_strings):
                        break
                # Save boundary state at the start of each new window.
                # KV has 0..pos-1; token will decode at pos → prime_pos=pos not in KV.
                if len(fp16_tokens) % W == 0:
                    kv_blob = strategies.save_kv_state(lib, ctx)
                    boundary_states.append((kv_blob, token, pos))
                ret = strategies._single_decode(lib, ctx, token, pos)
                if ret != 0:
                    break
                ptr    = lib.llama_get_logits_ith(ctx, 0)
                logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
                token  = int(np.argmax(logits))
                fp16_tokens.append(token)
                pos   += 1

            N = len(fp16_tokens)
            if N == 0:
                sim_results.append(None)
                continue

            if args.show_gen and _show_this(ei, len(examples)):
                gen_text = llama.detokenize(lib, vocab, fp16_tokens, remove_special=False)
                print(f"\n--- fp16 gen [{label}] ({N} tokens) ---", flush=True)
                print(gen_text.strip(), flush=True)
                print("---", flush=True)

            n_full_windows = N // W          # only verify complete windows
            n_accepted        = 0
            first_fail_window = None
            first_fail_pos    = None

            # ── Phase 2: window-by-window int4 verification ──────────────────
            for w in range(min(n_full_windows, len(boundary_states))):
                w_start = w * W
                w_end   = w_start + W          # always a complete window here
                kv_blob, prime_tok, prime_pos = boundary_states[w]

                # (a) Restore fp16 KV state → simulates KV overwrite after verify
                strategies.restore_kv_state(lib, ctx, kv_blob)

                # (b) Re-prime with fp16 (no hook): feed prime_tok to get logits
                #     for fp16_tokens[w_start].
                ret = strategies._single_decode(lib, ctx, prime_tok, prime_pos)
                if ret != 0:
                    break
                ptr         = lib.llama_get_logits_ith(ctx, 0)
                prime_logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()

                # (c) Check first-token prediction (fp16 KV, no draft hook yet)
                av_k, av_p = args.adaptive_verify_top_k, args.adaptive_verify_top_p
                need_av = av_k is not None or av_p is not None
                if not strategies.adaptive_verify_accept(
                        prime_logits, fp16_tokens[w_start], av_k, av_p):
                    if first_fail_window is None:
                        first_fail_window = w
                        first_fail_pos    = w_start
                    continue

                # (d) Feed fp16_tokens[w_start .. w_end-2] with draft hook.
                window_slice = fp16_tokens[w_start:w_end - 1]   # W-1 tokens
                if need_av:
                    greedy, log_rows = strategies.verify_window(
                        lib, ctx, n_vocab,
                        window_slice,
                        pos_start=n_pt + w_start,
                        kv_hook=draft_hook,
                        k_group_size=draft_k_gs,
                        v_group_size=draft_v_gs,
                        return_logits=True)
                    window_ok = all(
                        strategies.adaptive_verify_accept(
                            log_rows[i], fp16_tokens[w_start + i + 1], av_k, av_p)
                        for i in range(len(greedy)))
                else:
                    greedy = strategies.verify_window(
                        lib, ctx, n_vocab,
                        window_slice,
                        pos_start=n_pt + w_start,
                        kv_hook=draft_hook,
                        k_group_size=draft_k_gs,
                        v_group_size=draft_v_gs)
                    window_ok = all(greedy[i] == fp16_tokens[w_start + i + 1]
                                    for i in range(len(greedy)))

                if window_ok:
                    n_accepted += 1
                elif first_fail_window is None:
                    first_fail_window = w
                    for i, g in enumerate(greedy):
                        ok_i = (
                            strategies.adaptive_verify_accept(
                                log_rows[i], fp16_tokens[w_start + i + 1], av_k, av_p)
                            if need_av else (g == fp16_tokens[w_start + i + 1]))
                        if not ok_i:
                            first_fail_pos = w_start + i + 1
                            break

            acceptance_rate = n_accepted / max(n_full_windows, 1)
            draft_fraction  = n_accepted * W / max(N, 1)

            entry = {
                "label":               label,
                "n_tokens_fp16":       N,
                "n_windows":           n_full_windows,
                "n_accepted_windows":  n_accepted,
                "first_fail_window":   first_fail_window,
                "first_fail_pos":      first_fail_pos,
                "acceptance_rate":     acceptance_rate,
                "draft_fraction":      draft_fraction,
                "adaptive_verify":     _adaptive_verify_label(args),
            }
            sim_results.append(entry)
            status = ("all_ok" if first_fail_window is None
                      else f"fail@w{first_fail_window}/tok{first_fail_pos}")
            print(f"  adaptive ex {ei+1}/{len(examples)}: {label} | "
                  f"acc={acceptance_rate:.2f}  draft_frac={draft_fraction:.2f}  "
                  f"n_tok={N}  {status}",
                  flush=True)
        return sim_results

    def run_adaptive_gen(examples, draft_hook, draft_k_gs, draft_v_gs,
                         use_gpu_shadow=False, n_layer=0, draft_quant_name="draft",
                         ver_hook=None, ver_k_gs=64, ver_v_gs=64,
                         verifier_quant_name="fp16",
                         draft_quant_candidates=None):
        """Real adaptive generation: bootstrap window + quant rollout windows + verify.

        Bootstrap window (first segment): fp16 generates W_b tokens (kv_hook=None).
        Bootstrap candidate check replays the fp16 reference in steps of --adaptive-window
        (same restore → bulk quant → decode prime → draft W → verify as post-bootstrap
        rollout), not one continuous quant verify over the full bootstrap length.

        Quant rollout windows (each later segment while generating):
          a. Draft quant generates W tokens from restored fp16 boundary (`draft_hook`).
          b. Verifier replays those tokens (`ver_hook` or fp16 if None).
          c. Verifier acceptance (--adaptive-verify-top-k / --adaptive-verify-top-p or greedy);
           on accept keep quant rollout; else fp16 regenerates this window (fallback).

        KV save/restore:
        - With --adaptive-gen-gpu-shadow: each checkpoint saves GPU D2D tensors plus a CPU
          seq blob; restore uses the CPU path so metadata matches llama (D2D-only restore
          can desync positions vs tensor bytes). PCIe cost is like the no-CuPy case.
        - Without gpu-shadow: save_kv_state / restore_kv_state only.
        Hot path per quant rollout window: typically two restores (before draft quant,
        before verifier) plus one save after accept/reject — three checkpoint operations per
        window attempt.

        Why not strategies.trim_kv_seq here: draft_hook(ctx, n_new_k=None, n_new_v=None)
        bulk-quantizes the entire KV cache in place, including the prompt region, so fp16
        values are overwritten. Restoring fp16 requires the saved blob; trimming tail cells
        only recovers length metadata, not corrupted prompt cells.

        KV invariant after a successful verify: positions 0..prime_pos-1 hold fp16-derived
        values; the next draft bulk-quantizes from that fp16 snapshot again.

        Step B (verify, per quant rollout window): ver_hook=None when --verifier-quant fp16
        (teacher) — verify_window uses one batched fp16 prefill for the replayed tokens.
        With a quant verifier (ver_hook set), verify_window stays token-by-token so each
        logits step sees quantized KV; there is no correct single-batch shortcut for that.

        Returns list of dicts per example: gen_ids, acceptance_rate, draft_fraction, segments.
        """
        W   = args.adaptive_window    # rollout window size (int2 draft / fp16 verify cycles)
        W_b = args.bootstrap_window   # bootstrap window size (window 0 fp16 + candidate verify)
        results = []
        av_k = args.adaptive_verify_top_k
        av_p = args.adaptive_verify_top_p
        need_av = av_k is not None or av_p is not None
        qviz_legend_printed = False

        cand_specs = draft_quant_candidates if draft_quant_candidates is not None else [draft_quant_name]
        cand_specs = [c for c in cand_specs if c and c != "fp16"]
        _ord = {c: i for i, c in enumerate(cand_specs)}
        cand_specs = sorted(cand_specs, key=lambda c: (_quant_bits_estimate(c), _ord.get(c, 10**9)))

        def _emit_segment(buf: list, name: str, n: int) -> None:
            """Merge consecutive runs with the same quant label."""
            if n <= 0:
                return
            if buf and buf[-1][0] == name:
                buf[-1] = (name, buf[-1][1] + n)
            else:
                buf.append((name, n))

        def _adjust_segments_for_trim(buf: list, gen_len: int, trimmed_len: int) -> list:
            """Drop trailing token counts when trimming at EOG (gen_ids longer than trimmed)."""
            if gen_len <= trimmed_len or not buf:
                return buf
            drop = gen_len - trimmed_len
            out = list(buf)
            while drop > 0 and out:
                name, n = out[-1]
                if n > drop:
                    out[-1] = (name, n - drop)
                    break
                drop -= n
                out.pop()
            return out

        def _save_ckpt_pair():
            """GPU D2D snapshot plus CPU seq bytes.

            Restore always uses the CPU blob when present: D2D memcpy alone does not
            reliably resync llama KV metadata (v_cells / positions) with tensor bytes,
            which breaks bootstrap verify under --adaptive-gen-gpu-shadow; CPU restore
            matches the no-CuPy path. GPU copies are kept for debugging / future use.
            """
            if use_gpu_shadow:
                g = gpu_kv_shadow.GpuKvShadowCheckpoint()
                g.save(lib, ctx, n_layer)
                c = strategies.save_kv_state(lib, ctx)
                return (g, c)
            return (None, strategies.save_kv_state(lib, ctx))

        def _restore_ckpt_pair(pair):
            g, c = pair
            if c is not None:
                strategies.restore_kv_state(lib, ctx, c)
                return
            if g is not None:
                g.restore(lib, ctx, n_layer)
                return
            raise RuntimeError("restore_ckpt_pair: empty checkpoint")

        for ei, (pt, ct, label) in enumerate(examples):
            max_ctx_avail = args.n_ctx - len(pt) - 1
            max_gen = min(args.max_gen_tokens, max_ctx_avail)
            if max_gen <= 0:
                results.append(None)
                continue

            mem = lib.llama_get_memory(ctx)
            lib.llama_memory_clear(mem, True)
            n_pt = len(pt)

            # ── Prompt prefill ────────────────────────────────────────────
            # We split the prefill into two parts so that we can save a blob
            # at KV = 0..n_pt-2 (before the last prompt token).  This blob
            # is used as the "prime boundary" for window 0 int2 verification,
            # giving a clean position for re-decoding pt[-1] in int2 context.
            if n_pt > 1:
                # Part 1: batch-prefill pt[0..n_pt-2] (no logits needed)
                batch = lib.llama_batch_init(n_pt - 1, 0, 1)
                batch.n_tokens = n_pt - 1
                for i, tok in enumerate(pt[:-1]):
                    batch.token[i]     = tok
                    batch.pos[i]       = i
                    batch.n_seq_id[i]  = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i]    = 0
                if lib.llama_decode(ctx, batch) != 0:
                    lib.llama_batch_free(batch)
                    results.append(None)
                    continue
                lib.llama_batch_free(batch)
                # Save KV = 0..n_pt-2 (fp16); prime boundary for window 0
                pre_prime_blob = _save_ckpt_pair()
                # Part 2: decode last prompt token (logits for bootstrap window)
                ret = strategies._single_decode(lib, ctx, pt[-1], n_pt - 1)
                if ret != 0:
                    results.append(None)
                    continue
                ptr = lib.llama_get_logits_ith(ctx, 0)
            else:
                # n_pt==1: still need a checkpoint *before* decoding the sole prompt token,
                # so bootstrap candidate selection can restore → quantize → decode pt[0]
                # (same roles as pre_prime_blob for n_pt>1). Without this, pre_prime_blob is
                # None and we skip draft auto-pick, leaving draft_quant_name (often int2_ch).
                pre_prime_blob = _save_ckpt_pair()
                ret = strategies._single_decode(lib, ctx, pt[0], 0)
                if ret != 0:
                    results.append(None)
                    continue
                ptr = lib.llama_get_logits_ith(ctx, 0)

            logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()

            # ── Bootstrap window: fp16 generates W_b tokens ─────────────────
            bootstrap_tokens = strategies.generate_window(
                lib, ctx, n_vocab, int(np.argmax(logits)),
                pos_start=n_pt, W=W_b,
                kv_hook=None,
                stop_fn=_is_eog)

            gen_ids = list(bootstrap_tokens)
            path_per_tok = ["fp16"] * len(gen_ids)
            segments: list[tuple[str, int]] = []
            _emit_segment(segments, "fp16", len(bootstrap_tokens))
            # KV = 0..n_pt+len(bootstrap_tokens)-2 (all fp16)
            kv_boundary = _save_ckpt_pair()
            prime_tok = bootstrap_tokens[-1]
            prime_pos = n_pt + len(bootstrap_tokens) - 1

            ex_draft_hook = draft_hook
            ex_draft_k_gs = draft_k_gs
            ex_draft_v_gs = draft_v_gs
            ex_draft_name = draft_quant_name

            # ── Bootstrap pick: try each --quants candidate; score probe acceptance;
            # then pick max (probe_rate, bit_width) so int4 can win over int2 on ties.
            # Windowed bootstrap verify: same restore → bulk quant → prime decode → draft W
            # → verifier replay as post-bootstrap rollout, repeated over bootstrap_tokens.
            n_accepted  = 0
            n_attempted = 0
            if pre_prime_blob is not None:
                picked = None
                scored = []  # bootstrap+probe results per candidate (for final argmax)
                failed = []  # candidates that failed bootstrap, with n_match score
                tried = cand_specs or [draft_quant_name]
                for cand in tried:
                    cand_k_names, cand_v_names = quant_mod.resolve_quant_layers(cand, n_layer)
                    cand_k_gs, cand_v_gs = get_kv_group_sizes(
                        cand_k_names, cand_v_names, args.quant_group_size)
                    cand_hook = make_kv_hook(
                        lib, cand_k_names, cand_v_names, args.n_pos_per_embd,
                        use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                        default_group_size=args.quant_group_size,
                        n_sink=args.sink_tokens,
                        n_recent=args.recent_tokens,
                        k_sink_names=k_sink_names,
                        v_sink_names=v_sink_names,
                        k_recent_names=k_recent_names,
                        v_recent_names=v_recent_names,
                        asym=args.asym, quant_fn_factory=None,
                        profile=kv_prof)

                    kv_roll = None  # fp16 teacher after each chunk (pair or gpu/cpu ckpt)
                    prime_tok_bs = pt[-1]
                    prime_pos_bs = n_pt - 1
                    off = 0
                    w0_ok = True
                    n_match = 0

                    if not bootstrap_tokens:
                        w0_ok = False

                    while w0_ok and off < len(bootstrap_tokens):
                        w_chunk = min(W, len(bootstrap_tokens) - off)
                        pos_start = prime_pos_bs + 1
                        ref_chunk = bootstrap_tokens[off:off + w_chunk]

                        # First chunk: same as legacy bootstrap (pre_prime → quant prefill →
                        # decode pt[-1]). Do *not* use rollout bulk+decode(prime) here: the
                        # fp16 prompt checkpoint already has pt[-1] at n_pt-1, which would
                        # duplicate position (llama requires Y = X+1).
                        if off == 0:
                            _restore_ckpt_pair(pre_prime_blob)
                            cand_hook(ctx, n_new_k=None, n_new_v=None,
                                      n_prompt=n_pt - 1)
                            ret = strategies._single_decode(lib, ctx, pt[-1], n_pt - 1)
                            if ret != 0:
                                w0_ok = False
                                n_match = off
                                break
                            cand_hook(ctx, n_new_k=1, n_new_v=1)
                        else:
                            _restore_ckpt_pair(kv_roll)
                            cand_hook(ctx, n_new_k=None, n_new_v=None,
                                      n_prompt=0, n_seq_len=prime_pos_bs + 1)
                            ret = strategies._single_decode(lib, ctx, prime_tok_bs,
                                                            prime_pos_bs)
                            if ret != 0:
                                w0_ok = False
                                n_match = off
                                break
                            cand_hook(ctx, n_new_k=1, n_new_v=1)
                        ptr_d = lib.llama_get_logits_ith(ctx, 0)
                        d_logits = np.ctypeslib.as_array(ptr_d, shape=(n_vocab,)).copy()
                        draft_tokens = strategies.generate_window(
                            lib, ctx, n_vocab, int(np.argmax(d_logits)),
                            pos_start=pos_start, W=w_chunk,
                            kv_hook=cand_hook,
                            k_group_size=cand_k_gs, v_group_size=cand_v_gs,
                            stop_fn=_is_eog)

                        if off == 0:
                            _restore_ckpt_pair(pre_prime_blob)
                            if ver_hook is not None:
                                ver_hook(ctx, n_new_k=None, n_new_v=None,
                                         n_prompt=n_pt - 1)
                            ret_v = strategies._single_decode(lib, ctx, pt[-1], n_pt - 1)
                            if ret_v != 0:
                                w0_ok = False
                                n_match = off
                                break
                            if ver_hook is not None:
                                ver_hook(ctx, n_new_k=1, n_new_v=1)
                        else:
                            _restore_ckpt_pair(kv_roll)
                            if ver_hook is not None:
                                ver_hook(ctx, n_new_k=None, n_new_v=None,
                                         n_prompt=0, n_seq_len=prime_pos_bs + 1)
                            ret_v = strategies._single_decode(lib, ctx, prime_tok_bs,
                                                              prime_pos_bs)
                            if ret_v != 0:
                                w0_ok = False
                                n_match = off
                                break
                            if ver_hook is not None:
                                ver_hook(ctx, n_new_k=1, n_new_v=1)
                        ptr_v = lib.llama_get_logits_ith(ctx, 0)
                        v_logits = np.ctypeslib.as_array(ptr_v, shape=(n_vocab,)).copy()

                        if len(draft_tokens) > 1:
                            if need_av:
                                v_greedy, v_log_rows = strategies.verify_window(
                                    lib, ctx, n_vocab,
                                    draft_tokens[:-1],
                                    pos_start=pos_start,
                                    kv_hook=ver_hook,
                                    k_group_size=ver_k_gs,
                                    v_group_size=ver_v_gs,
                                    return_logits=True)
                                window_ok = (
                                    strategies.adaptive_verify_accept(
                                        v_logits, draft_tokens[0], av_k, av_p)
                                    and all(
                                        strategies.adaptive_verify_accept(
                                            v_log_rows[i], draft_tokens[i + 1], av_k, av_p)
                                        for i in range(len(v_greedy))))
                            else:
                                v_pred0 = int(np.argmax(v_logits))
                                v_greedy = strategies.verify_window(
                                    lib, ctx, n_vocab,
                                    draft_tokens[:-1],
                                    pos_start=pos_start,
                                    kv_hook=ver_hook,
                                    k_group_size=ver_k_gs,
                                    v_group_size=ver_v_gs)
                                window_ok = (v_pred0 == draft_tokens[0] and
                                             all(v_greedy[i] == draft_tokens[i + 1]
                                                 for i in range(len(v_greedy))))
                        else:
                            window_ok = strategies.adaptive_verify_accept(
                                v_logits, draft_tokens[0], av_k, av_p)

                        ref_ok = (len(draft_tokens) == len(ref_chunk) and
                                  all(draft_tokens[i] == ref_chunk[i]
                                      for i in range(len(ref_chunk))))

                        if not (window_ok and ref_ok):
                            w0_ok = False
                            n_match = off
                            for i in range(min(len(draft_tokens), len(ref_chunk))):
                                if draft_tokens[i] != ref_chunk[i]:
                                    break
                                n_match += 1
                            break

                        kv_roll = _save_ckpt_pair()
                        prime_tok_bs = ref_chunk[-1]
                        prime_pos_bs = pos_start + len(ref_chunk) - 1
                        off += len(ref_chunk)

                    if w0_ok and off != len(bootstrap_tokens):
                        w0_ok = False

                    # ── Per-candidate bootstrap diagnostic ───────────────
                    if not w0_ok:
                        if len(bootstrap_tokens) > 1 and n_match >= 0:
                            print(f"      {cand}: bootstrap FAIL "
                                  f"({n_match}/{len(bootstrap_tokens)} tokens matched)",
                                  flush=True)
                        else:
                            print(f"      {cand}: bootstrap FAIL", flush=True)
                        failed.append({
                            "name":  cand,
                            "match": n_match if len(bootstrap_tokens) > 1 else -1,
                            "bits":  _quant_bits_estimate(cand),
                            "hook":  cand_hook,
                            "k_gs":  cand_k_gs,
                            "v_gs":  cand_v_gs,
                        })
                        continue

                    # Probe: estimate early acceptance rate over a few rollout windows.
                    # This avoids picking a low-bit quant that passes bootstrap but is
                    # rejected frequently once rollout starts.
                    probe_windows = 4
                    probe_min_rate = 0.90
                    probe_ckpt = kv_boundary
                    probe_prime_tok = prime_tok
                    probe_prime_pos = prime_pos
                    probe_attempted = 0
                    probe_accepted = 0
                    for _ in range(probe_windows):
                        probe_w = min(W, max_gen - len(gen_ids))
                        if probe_w <= 0 or _is_eog(probe_prime_tok):
                            break
                        probe_attempted += 1

                        # Draft window under candidate
                        _restore_ckpt_pair(probe_ckpt)
                        cand_hook(ctx, n_new_k=None, n_new_v=None,
                                  n_prompt=0, n_seq_len=probe_prime_pos + 1)
                        ret_p = strategies._single_decode(lib, ctx, probe_prime_tok, probe_prime_pos)
                        if ret_p != 0:
                            break
                        cand_hook(ctx, n_new_k=1, n_new_v=1)
                        ptr_p = lib.llama_get_logits_ith(ctx, 0)
                        cand_draft_logits = np.ctypeslib.as_array(ptr_p, shape=(n_vocab,)).copy()
                        draft_tokens = strategies.generate_window(
                            lib, ctx, n_vocab, int(np.argmax(cand_draft_logits)),
                            pos_start=probe_prime_pos + 1, W=probe_w,
                            kv_hook=cand_hook,
                            k_group_size=cand_k_gs, v_group_size=cand_v_gs,
                            stop_fn=_is_eog)

                        # Verifier replay for acceptance
                        _restore_ckpt_pair(probe_ckpt)
                        if ver_hook is not None:
                            ver_hook(ctx, n_new_k=None, n_new_v=None,
                                     n_prompt=0, n_seq_len=probe_prime_pos + 1)
                        ret_vp = strategies._single_decode(lib, ctx, probe_prime_tok, probe_prime_pos)
                        if ret_vp != 0:
                            break
                        if ver_hook is not None:
                            ver_hook(ctx, n_new_k=1, n_new_v=1)
                        ptr_vp = lib.llama_get_logits_ith(ctx, 0)
                        vp_logits = np.ctypeslib.as_array(ptr_vp, shape=(n_vocab,)).copy()
                        if len(draft_tokens) > 1:
                            if need_av:
                                vp_greedy, vp_rows = strategies.verify_window(
                                    lib, ctx, n_vocab,
                                    draft_tokens[:-1],
                                    pos_start=probe_prime_pos + 1,
                                    kv_hook=ver_hook,
                                    k_group_size=ver_k_gs,
                                    v_group_size=ver_v_gs,
                                    return_logits=True)
                                window_ok = (
                                    strategies.adaptive_verify_accept(
                                        vp_logits, draft_tokens[0], av_k, av_p)
                                    and all(
                                        strategies.adaptive_verify_accept(
                                            vp_rows[i], draft_tokens[i + 1], av_k, av_p)
                                        for i in range(len(vp_greedy))))
                            else:
                                vp_pred0 = int(np.argmax(vp_logits))
                                vp_greedy = strategies.verify_window(
                                    lib, ctx, n_vocab,
                                    draft_tokens[:-1],
                                    pos_start=probe_prime_pos + 1,
                                    kv_hook=ver_hook,
                                    k_group_size=ver_k_gs,
                                    v_group_size=ver_v_gs)
                                window_ok = (vp_pred0 == draft_tokens[0] and
                                             all(vp_greedy[i] == draft_tokens[i + 1]
                                                 for i in range(len(vp_greedy))))
                        else:
                            window_ok = strategies.adaptive_verify_accept(
                                vp_logits, draft_tokens[0], av_k, av_p)

                        if window_ok:
                            probe_accepted += 1
                            probe_ckpt = _save_ckpt_pair()
                            probe_prime_tok = draft_tokens[-1]
                            probe_prime_pos = probe_prime_pos + len(draft_tokens)
                        else:
                            # Fallback: advance under fp16, like real adaptive-gen does.
                            _restore_ckpt_pair(probe_ckpt)
                            ret_fb = strategies._single_decode(lib, ctx, probe_prime_tok, probe_prime_pos)
                            if ret_fb != 0:
                                break
                            ptr_fb = lib.llama_get_logits_ith(ctx, 0)
                            fb_logits = np.ctypeslib.as_array(ptr_fb, shape=(n_vocab,)).copy()
                            fb_toks = strategies.generate_window(
                                lib, ctx, n_vocab, int(np.argmax(fb_logits)),
                                pos_start=probe_prime_pos + 1, W=probe_w,
                                kv_hook=None,
                                stop_fn=_is_eog)
                            probe_ckpt = _save_ckpt_pair()
                            probe_prime_tok = fb_toks[-1]
                            probe_prime_pos = probe_prime_pos + len(fb_toks)

                    probe_rate = probe_accepted / max(probe_attempted, 1)
                    probe_tag = "PASS" if probe_rate >= probe_min_rate else f"FAIL (<{probe_min_rate:.2f})"
                    print(f"      {cand}: bootstrap OK "
                          f"→ probe {probe_accepted}/{probe_attempted}={probe_rate:.2f} {probe_tag}",
                          flush=True)
                    scored.append({
                        "name":   cand,
                        "rate":   probe_rate,
                        "acc":    probe_accepted,
                        "tot":    probe_attempted,
                        "bits":   _quant_bits_estimate(cand),
                        "hook":   cand_hook,
                        "k_gs":   cand_k_gs,
                        "v_gs":   cand_v_gs,
                    })

                # Among candidates that pass the probe threshold, pick the most
                # aggressive (lowest bits) — the goal is maximum compression that
                # still meets the acceptance bar.  On a bits tie, prefer higher rate.
                # If nothing clears the threshold, fall back to the highest-rate
                # candidate (safest choice among those that were scored).
                if scored:
                    eligible = [s for s in scored if s["rate"] >= probe_min_rate]
                    if eligible:
                        win = min(eligible, key=lambda s: (s["bits"], -s["rate"]))
                    else:
                        win = max(scored, key=lambda s: (s["rate"], s["bits"]))
                    picked = (win["name"], win["hook"], win["k_gs"], win["v_gs"])
                    summ = ", ".join(
                        f"{s['name']}={s['rate']:.2f}({s['acc']}/{s['tot']})"
                        for s in sorted(scored, key=lambda s: (_quant_bits_estimate(s["name"]), s["name"])))
                    print(f"    bootstrap_pick [{label}]: {win['name']}  "
                          f"probe={win['rate']:.2f} ({win['acc']}/{win['tot']})  "
                          f"[{summ}]",
                          flush=True)
                elif failed:
                    # No candidate passed bootstrap+probe.
                    # Rank by n_match (most tokens matched = closest to fp16) then min
                    # bits on ties — prefer the most accurate quant we tried, and among
                    # equals prefer the most aggressive compression.
                    # Candidates with unknown n_match (need_av / single-token) sort last.
                    fb_entry = max(failed, key=lambda s: (s["match"], -s["bits"]))
                    fb = fb_entry["name"]
                    fb_hook = fb_entry["hook"]
                    fb_k_gs = fb_entry["k_gs"]
                    fb_v_gs = fb_entry["v_gs"]
                    match_str = (f"{fb_entry['match']}/{len(bootstrap_tokens)}"
                                 if fb_entry["match"] >= 0 else "?")
                    def _fmt_fail(s):
                        ms = f"{s['match']}/{len(bootstrap_tokens)}" if s["match"] >= 0 else "?"
                        return f"{s['name']}={ms}"
                    others = ", ".join(
                        _fmt_fail(s)
                        for s in sorted(failed, key=lambda s: (_quant_bits_estimate(s["name"]), s["name"]))
                        if s["name"] != fb)
                    picked = (fb, fb_hook, fb_k_gs, fb_v_gs)
                    print(f"    bootstrap_pick [{label}]: {fb}  FALLBACK (best n_match={match_str})"
                          + (f"  [{others}]" if others else ""),
                          flush=True)
                else:
                    print(f"    bootstrap_pick [{label}]: NONE  (no candidates in tried)", flush=True)

                n_attempted += 1
                if picked is not None:
                    ex_draft_name, ex_draft_hook, ex_draft_k_gs, ex_draft_v_gs = picked
                if scored:
                    n_accepted += 1
                # else: keep the passed-in draft_* as fallback
                # Restore fp16 kv_boundary so phase 2 starts clean
                _restore_ckpt_pair(kv_boundary)

            def _hit_stop(ids):
                if not args.stop_strings:
                    return False
                tail = llama.detokenize(lib, vocab, ids[-128:], remove_special=False)
                return any(s in tail for s in args.stop_strings)

            # ── Quant rollout windows: draft quant + verifier, until max_gen ─
            while len(gen_ids) < max_gen:
                if _is_eog(prime_tok) or _hit_stop(gen_ids):
                    break

                pos_start = prime_pos + 1
                w_size    = min(W, max_gen - len(gen_ids))
                n_attempted += 1

                # Step A: draft generates w_size tokens with quantized KV.
                # Restore fp16 boundary, then bulk-quantize KV (respecting sink/recent zones
                # from --sink-tokens/--recent-tokens) so the draft attends to the correct
                # quantized background — matching the real deployment configuration.
                _restore_ckpt_pair(kv_boundary)
                ex_draft_hook(ctx, n_new_k=None, n_new_v=None,
                              n_prompt=0, n_seq_len=prime_pos + 1)
                ret = strategies._single_decode(lib, ctx, prime_tok, prime_pos)
                if ret != 0:
                    break
                ex_draft_hook(ctx, n_new_k=1, n_new_v=1)        # quantize prime_pos cell
                ptr          = lib.llama_get_logits_ith(ctx, 0)
                draft_logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
                quant_rollout_tokens = strategies.generate_window(
                    lib, ctx, n_vocab, int(np.argmax(draft_logits)),
                    pos_start=pos_start, W=w_size,
                    kv_hook=ex_draft_hook,
                    k_group_size=ex_draft_k_gs, v_group_size=ex_draft_v_gs,
                    stop_fn=_is_eog)

                # Step B: verifier replays quant_rollout_tokens (--verifier-quant, or fp16).
                _restore_ckpt_pair(kv_boundary)
                if ver_hook is not None:
                    ver_hook(ctx, n_new_k=None, n_new_v=None,
                             n_prompt=0, n_seq_len=prime_pos + 1)
                ret = strategies._single_decode(lib, ctx, prime_tok, prime_pos)
                if ret != 0:
                    break
                if ver_hook is not None:
                    ver_hook(ctx, n_new_k=1, n_new_v=1)
                ptr         = lib.llama_get_logits_ith(ctx, 0)
                v_logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()

                if len(quant_rollout_tokens) > 1:
                    if need_av:
                        v_greedy, v_log_rows = strategies.verify_window(
                            lib, ctx, n_vocab,
                            quant_rollout_tokens[:-1],
                            pos_start=pos_start,
                            kv_hook=ver_hook,
                            k_group_size=ver_k_gs,
                            v_group_size=ver_v_gs,
                            return_logits=True)
                        window_ok = (
                            strategies.adaptive_verify_accept(
                                v_logits, quant_rollout_tokens[0], av_k, av_p)
                            and all(
                                strategies.adaptive_verify_accept(
                                    v_log_rows[i], quant_rollout_tokens[i + 1], av_k, av_p)
                                for i in range(len(v_greedy))))
                    else:
                        v_pred0 = int(np.argmax(v_logits))
                        v_greedy = strategies.verify_window(
                            lib, ctx, n_vocab,
                            quant_rollout_tokens[:-1],
                            pos_start=pos_start,
                            kv_hook=ver_hook,
                            k_group_size=ver_k_gs,
                            v_group_size=ver_v_gs)
                        window_ok = (v_pred0 == quant_rollout_tokens[0] and
                                     all(v_greedy[i] == quant_rollout_tokens[i + 1]
                                         for i in range(len(v_greedy))))
                else:
                    v_greedy = []
                    window_ok = strategies.adaptive_verify_accept(
                        v_logits, quant_rollout_tokens[0], av_k, av_p)

                if window_ok:
                    n_accepted += 1
                    gen_ids    += quant_rollout_tokens
                    path_per_tok += [ex_draft_name] * len(quant_rollout_tokens)
                    _emit_segment(segments, ex_draft_name, len(quant_rollout_tokens))
                    # KV is fp16 from verify_window; save new boundary blob.
                    kv_boundary = _save_ckpt_pair()
                    prime_tok = quant_rollout_tokens[-1]
                    prime_pos = pos_start + len(quant_rollout_tokens) - 1
                else:
                    # Fallback: fp16 generates this window (restore fp16 first).
                    _restore_ckpt_pair(kv_boundary)
                    ret = strategies._single_decode(lib, ctx, prime_tok, prime_pos)
                    if ret != 0:
                        break
                    ptr       = lib.llama_get_logits_ith(ctx, 0)
                    fb_logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
                    fb_toks   = strategies.generate_window(
                        lib, ctx, n_vocab, int(np.argmax(fb_logits)),
                        pos_start=pos_start, W=w_size,
                        kv_hook=None,
                        stop_fn=_is_eog)
                    gen_ids   += fb_toks
                    path_per_tok += ["fp16"] * len(fb_toks)
                    _emit_segment(segments, "fp16", len(fb_toks))
                    kv_boundary = _save_ckpt_pair()
                    prime_tok = fb_toks[-1]
                    prime_pos = pos_start + len(fb_toks) - 1

            # Trim at first EOG (keep path aligned with trimmed output)
            trimmed = []
            trimmed_path = []
            for tok, pq in zip(gen_ids, path_per_tok):
                if _is_eog(tok):
                    break
                trimmed.append(tok)
                trimmed_path.append(pq)

            acc   = n_accepted / max(n_attempted, 1)
            dfrac = n_accepted * W / max(len(trimmed), 1)
            seg_adj = _adjust_segments_for_trim(segments, len(gen_ids), len(trimmed))
            print(f"  adaptive-gen ex {ei+1}/{len(examples)}: {label} | "
                  f"acc={acc:.2f}  draft_frac={dfrac:.2f}  "
                  f"verify={_adaptive_verify_label(args)}  "
                  f"n_tok={len(trimmed)}", flush=True)
            if seg_adj:
                # Condense segments: total tokens per quant, and number of runs per quant
                seg_toks: Counter = Counter()
                seg_runs: Counter = Counter()
                for q, n in seg_adj:
                    seg_toks[q] += n
                    seg_runs[q] += 1
                quants_seen = sorted(seg_toks, key=lambda q: -seg_toks[q])
                seg_summary = "  ".join(
                    f"{q}: {seg_toks[q]}tok/{seg_runs[q]}runs"
                    for q in quants_seen)
                print(f"    segments({len(seg_adj)}): {seg_summary}", flush=True)
            qviz_lines = []
            qviz_html_fragment = None
            if seg_adj:
                qviz_lines = _segment_quant_viz_rows(seg_adj, len(trimmed), width=_QVIZ_WIDTH)
                if getattr(args, "qviz_html", None):
                    qviz_html_fragment = _segment_quant_viz_html(
                        seg_adj, len(trimmed), width=_QVIZ_WIDTH, title=label)
                if qviz_lines:
                    if not qviz_legend_printed:
                        print(f"    {_QVIZ_LEGEND}", flush=True)
                        if getattr(args, "qviz_ansi", False):
                            print("    qviz-ansi: 24-bit background; taller=darker; "
                                  "int2 light gray → fp16 black; empty=white", flush=True)
                        qviz_legend_printed = True
                    if getattr(args, "qviz_ansi", False):
                        for i, line in enumerate(_segment_quant_viz_ansi_lines(
                                seg_adj, len(trimmed), width=_QVIZ_WIDTH)):
                            print(f"    qviz{i + 1}: {line}", flush=True)
                    else:
                        for i, row in enumerate(qviz_lines):
                            print(f"    qviz{i + 1}: {row}", flush=True)

            results.append({
                "gen_ids":           trimmed,
                "n_tokens":          len(trimmed),
                "n_attempted":       n_attempted,
                "n_accepted":        n_accepted,
                "acceptance_rate":   acc,
                "draft_fraction":    dfrac,
                "verifier_quant":    verifier_quant_name,
                "adaptive_verify":   _adaptive_verify_label(args),
                "segments":          [{"quant": q, "n_tokens": n} for q, n in seg_adj],
                "segment_quant_viz": qviz_lines,
                "segment_quant_viz_html": qviz_html_fragment,
            })

        return results

    def eval_accuracy_pass(examples, gold_answers, kv_hook, k_group_size, v_group_size,
                           pregenerated=None):
        """Run greedy generation on each example, score against gold, return (score_sum, n_total, per_ex, gen_len_stats).

        For --eval-metric exact: n_correct / n_total (accuracy).
        For --eval-metric f1:    mean F1 (0–1) over all examples.
        per_ex: list of (gold_display, pred_display, score) tuples.
        gen_len_stats: dict with mean_gen_tokens / mean_gen_tokens_correct / mean_gen_tokens_wrong.
        """
        use_f1   = (args.eval_metric == "f1")
        use_code = (args.eval_metric == "code")
        ans_re   = re.compile(args.answer_regex) if not use_f1 and not use_code else None
        score_sum  = 0.0
        n_total    = 0
        per_ex     = []
        gen_lens_correct = []
        gen_lens_wrong   = []
        n_truncated      = 0
        n_correct        = 0   # score == 1.0
        n_wrong          = 0   # pred found but incorrect (not a length issue)
        n_trunc_fail     = 0   # truncated AND no answer extracted (length caused failure)
        n_no_match       = 0   # finished cleanly but regex found nothing
        for ei, (pt, ct, label) in enumerate(examples):
            gold = gold_answers[ei] if gold_answers else None
            max_ctx_avail = args.n_ctx - len(pt) - 1
            max_gen = min(args.max_gen_tokens, max_ctx_avail)
            if args.show_text and _show_this(ei, len(examples)):
                show_chunk_text(pt, ct, label=label)
            if args.show_prompt and _show_this(ei, len(examples)):
                prompt_text = llama.detokenize(lib, vocab, pt)
                print(f"\n--- prompt [{label}] ({len(pt)} tokens) ---", flush=True)
                print(prompt_text, flush=True)
                print("---", flush=True)
            if max_gen <= 0:
                per_ex.append((gold, None, 0.0))
                continue
            if pregenerated is not None and ei < len(pregenerated) and pregenerated[ei] is not None:
                gen_ids   = pregenerated[ei]["gen_ids"]
                gen_diags = {}
            else:
                gen_ids, gen_diags = strategies.run_generate(
                    lib, ctx, vocab, pt, n_vocab,
                    kv_hook=kv_hook,
                    max_new_tokens=max_gen,
                    is_eog=_is_eog,
                    stop_strings=args.stop_strings or None,
                    k_group_size=k_group_size,
                    v_group_size=v_group_size,
                    return_diagnostics=bool(args.save_per_example))
            gen_text_raw = llama.detokenize(lib, vocab, gen_ids, remove_special=False)
            if use_code:
                # Preserve leading indentation — code bodies need their 4-space indent.
                # Only strip trailing whitespace and any trailing stop-string remnant.
                gen_text = gen_text_raw.rstrip()
            else:
                gen_text  = gen_text_raw.strip()
                # Strip reasoning blocks: <think> (Qwen3/DeepSeek-R1),
                # analysis channel (GPT-OSS)
                gen_text  = re.sub(r'<think>.*?</think>\s*', '', gen_text, flags=re.DOTALL)
                gen_text  = re.sub(r'<think>.*',             '', gen_text, flags=re.DOTALL).strip()
                gen_text  = re.sub(r'<\|channel\|>analysis<\|message\|>.*?<\|end\|>\s*',
                                   '', gen_text, flags=re.DOTALL)
                # Extract final channel content if present
                m_final   = re.search(r'<\|channel\|>final<\|message\|>(.*)',
                                      gen_text, flags=re.DOTALL)
                if m_final:
                    gen_text = m_final.group(1).strip()
            gen_len   = len(gen_ids)
            truncated = (gen_len >= max_gen)

            if use_code:
                # Save full generated text; scoring deferred to eval_code.py.
                score     = 0.0          # placeholder — real score from pass@1 execution
                pred_disp = gen_text     # full code, no truncation
                gold_disp = ""
                mark      = "(exec later)"
                n_total  += 1
                inconclusive = False
            elif use_f1:
                gold_list   = gold if isinstance(gold, list) else ([gold] if gold else [])
                score       = _f1_score(gen_text, gold_list) if gold_list else 0.0
                pred_disp   = gen_text[:80].replace("\n", "↵")
                gold_disp   = (gold_list[0] if gold_list else "")[:80]
                mark        = f"{score:.2f}"
                n_total    += 1
                score_sum  += score
                inconclusive = False
            else:
                # Use last match: reasoning models often self-correct mid-generation.
                # The final answer after "Wait, let me recalculate" is more faithful.
                all_matches  = list(ans_re.finditer(gen_text))
                all_hedge_ms = list(_HEDGE_RE.finditer(gen_text))
                m            = all_matches[-1] if all_matches else None
                raw          = (m.group(1) or m.group(2)) if m else None
                if raw is None:
                    fb = list(_LAST_NUM_RE.finditer(gen_text))
                    raw = fb[-1].group(1) if fb else None
                pred         = raw.replace(",", "") if raw else None

                # Inconclusive: truncated while still doubting (last hedge after last answer).
                # The model expressed uncertainty and never reached a new conclusion.
                inconclusive = (
                    truncated and m is not None and all_hedge_ms and
                    all_hedge_ms[-1].start() > m.start()
                )

                gold_disp   = str(gold) if gold is not None else ""
                pred_disp   = str(pred) if pred is not None else "None"
                if inconclusive:
                    score     = 0.0
                    pred_disp = f"{pred_disp}?"   # flag that answer was doubted
                    mark      = "✗(inconclusive)"
                else:
                    score = 1.0 if (pred is not None and pred == gold) else 0.0
                    mark  = "✓" if score == 1.0 else "✗"
                    if truncated:
                        mark += "(trunc)"
                if gold is not None:
                    n_total    += 1
                    score_sum  += score

            n_hedges     = len(_HEDGE_RE.findall(gen_text))
            inconclusive = inconclusive if not use_f1 and not use_code else False
            if truncated:
                n_truncated += 1
            # Failure-mode breakdown (exact-match only; f1/code use their own metrics)
            if not use_f1 and not use_code and gold is not None:
                if score == 1.0:
                    n_correct += 1
                elif pred is not None and not inconclusive:
                    n_wrong += 1          # model gave a definite wrong answer
                elif truncated and pred is None:
                    n_trunc_fail += 1     # hit token/ctx limit before answer appeared
                elif pred is None and not truncated:
                    n_no_match += 1       # finished cleanly, regex found nothing
            (gen_lens_correct if score == 1.0 else gen_lens_wrong).append(gen_len)
            ex_entry = {"label": label, "score": score,
                        "gold": gold_disp, "pred": pred_disp,
                        "gen_len": gen_len, "truncated": truncated,
                        "inconclusive": inconclusive, "n_hedges": n_hedges}
            if gen_diags is not None:
                ex_entry["gen_diags"] = gen_diags
            per_ex.append(ex_entry)
            hedge_str = f"  hedges={n_hedges}" if n_hedges > 0 else ""
            print(f"  acc ex {ei+1}/{len(examples)}: {label} | "
                  f"gold={gold_disp}  pred={pred_disp}  {mark}  gen_len={gen_len}{hedge_str}", flush=True)
            if args.show_text and _show_this(ei, len(examples)):
                print(f"    gen: {gen_text!r}", flush=True)
            if args.show_gen and _show_this(ei, len(examples)):
                print(f"\n--- gen [{label}] gold={gold_disp} pred={pred_disp} {mark} ---",
                      flush=True)
                print(gen_text_raw.strip(), flush=True)
                print("---", flush=True)
        mean_score = score_sum / n_total if n_total > 0 else 0.0
        all_lens = gen_lens_correct + gen_lens_wrong
        def _mean(lst): return sum(lst) / len(lst) if lst else None
        all_hedges = [ex["n_hedges"] for ex in per_ex]
        gen_len_stats = {
            "mean_gen_tokens":         _mean(all_lens),
            "mean_gen_tokens_correct": _mean(gen_lens_correct),
            "mean_gen_tokens_wrong":   _mean(gen_lens_wrong),
            "n_truncated":             n_truncated,
            "n_inconclusive":          sum(1 for ex in per_ex if ex["inconclusive"]),
            "mean_hedges":             _mean(all_hedges),
            # Failure-mode breakdown (exact-match tasks only; 0 for f1/code)
            "n_correct":               n_correct,
            "n_wrong":                 n_wrong,
            "n_trunc_fail":            n_trunc_fail,
            "n_no_match":              n_no_match,
        }
        return mean_score, n_total, per_ex, gen_len_stats

    use_gpu = args.n_gpu_layers > 0 and HAS_GPU_QUANT and args.flash_attn
    if args.n_gpu_layers > 0 and not args.flash_attn:
        print("NOTE: --n-gpu-layers set but --flash-attn not set. "
              "Using CPU parse_state for KV quantization (slow for per-token quants). "
              "Add --flash-attn to enable GPU-side quantization.", flush=True)
    if use_gpu:
        print(f"GPU-side KV quantization enabled (CuPy, n_layer={n_layer})", flush=True)

    print(f"Model loaded, n_vocab={n_vocab}, n_layer={n_layer}", flush=True)

    # ── Structured mode ───────────────────────────────────────────────────────
    if args.corpus_mode == "structured":
        if not args.corpus_file.endswith(".jsonl"):
            print("WARNING: --corpus-mode structured expects a .jsonl file", flush=True)

        records = []
        with open(args.corpus_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        max_examples = args.n_chunks if args.n_chunks > 0 else len(records)
        records = records[:max_examples]
        print(f"Structured mode: {len(records)} examples from {args.corpus_file}", flush=True)

        # Tokenize all examples; truncate to n_ctx
        examples         = []
        raw_answers_list = []   # parallel list: rec["answers"] or [] per example
        skipped = 0
        for rec in records:
            # Native format: {"prompt": "...", "completion": "..."}
            # LongBench format: {"context": "...", "input": "...", "answers": [...]}
            if "prompt" in rec:
                prompt_text = rec["prompt"]
            elif "context" in rec and "input" in rec:
                prompt_text = rec["context"] + "\n\n" + rec["input"]
            else:
                prompt_text = rec.get("input", "")

            if "completion" in rec:
                completion_text = rec["completion"]
            elif "answers" in rec and rec["answers"]:
                completion_text = rec["answers"][0] if isinstance(rec["answers"], list) else rec["answers"]
            else:
                completion_text = ""

            pt = llama.tokenize(lib, vocab, args.prompt_prefix + prompt_text + args.prompt_suffix)
            ct = llama.tokenize(lib, vocab, completion_text)
            total = len(pt) + len(ct)
            if total > args.n_ctx:
                # Truncate prompt to make room for completion
                max_pt = args.n_ctx - len(ct) - 1
                if max_pt < 64:
                    skipped += 1
                    continue   # completion alone already fills context
                pt = pt[-max_pt:]   # keep the END of the prompt (most recent context)
            # Build a short label from whatever identifier fields exist
            if "repo" in rec and "file" in rec:
                label = f"{rec['repo']}/{rec['file']}"
            elif "dataset" in rec:
                label = f"{rec['dataset']}#{rec.get('id', len(examples))}"
            else:
                label = str(rec.get("id", i))
            examples.append((pt, ct, label))
            raw_answers_list.append(rec.get("answers", []))
        print(f"  {len(examples)} examples fit in n_ctx={args.n_ctx} "
              f"({skipped} skipped — completion too long)", flush=True)

        if not examples:
            raise RuntimeError("No examples fit in n_ctx. Increase --n-ctx.")

        # Extract gold answers for accuracy evaluation
        gold_answers = []
        if args.eval_accuracy:
            if args.eval_metric == "code":
                # No gold needed — scoring done externally by eval_code.py
                gold_answers = [None] * len(examples)
                print(f"  Code mode: {len(examples)} examples; pass@1 scored by eval_code.py", flush=True)
            elif args.eval_metric == "f1":
                # Gold = all answers from jsonl 'answers' field; fall back to completion text
                for (pt, ct, _), raw_ans in zip(examples, raw_answers_list):
                    if raw_ans:
                        gold_answers.append(raw_ans)  # list of strings
                    else:
                        gold_answers.append([llama.detokenize(lib, vocab, ct, remove_special=False)])
                print(f"  F1 mode: {len(gold_answers)} examples with gold answers", flush=True)
            else:
                ans_re = re.compile(args.answer_regex)
                for pt, ct, _ in examples:
                    text = llama.detokenize(lib, vocab, ct, remove_special=False)
                    m    = ans_re.search(text)
                    raw = (m.group(1) or m.group(2)) if m else None
                    gold_answers.append(raw.replace(",", "") if raw else None)
                n_gold = sum(a is not None for a in gold_answers)
                print(f"  Gold answers found: {n_gold}/{len(examples)}", flush=True)

        if args.skip_ppl and not args.eval_accuracy:
            raise ValueError("--skip-ppl requires --eval-accuracy (nothing left to compute otherwise)")

        # n_ctx: cover teacher-forced PPL pass and, if needed, generation pass
        if args.skip_ppl:
            # PPL pass skipped — only need room for prompt + generation
            actual_n_ctx = max(len(pt) for pt, _, _ in examples) + args.max_gen_tokens + 1
        else:
            actual_n_ctx = max(len(pt) + len(ct) for pt, ct, _ in examples) + 1
            if args.eval_accuracy:
                gen_n_ctx    = max(len(pt) for pt, _, _ in examples) + args.max_gen_tokens + 1
                actual_n_ctx = max(actual_n_ctx, gen_n_ctx)
        actual_n_ctx = min(actual_n_ctx, args.n_ctx)
        max_prompt   = max(len(pt) for pt, _, _ in examples)

        cparams = lib.llama_context_default_params()
        cparams.n_ctx           = actual_n_ctx
        cparams.n_batch         = max(max_prompt, 512)
        cparams.n_ubatch        = min(max(max_prompt, 512), 512)
        cparams.n_threads       = args.n_threads
        cparams.n_threads_batch = args.n_threads
        cparams.no_perf         = True
        if args.flash_attn:
            cparams.flash_attn_type = 1
        ctx = lib.llama_init_from_model(model, cparams)

        results = {}
        if args.skip_ppl:
            # ── Skip PPL baseline: no teacher-forced pass, no KL reference ────
            # KL will be reported as None for all quants.
            print("\n[fp16 baseline] SKIPPED (--skip-ppl)", flush=True)
            base_lps_per_ex   = None
            base_dists_per_ex = [None] * len(examples)
            fp16_ppl          = float("nan")
            if "fp16" in args.quants:
                t0 = time.time()
                print("\n[fp16]", flush=True)
                mean_score, nt, per_ex_fp16, gl = eval_accuracy_pass(
                    examples, gold_answers,
                    kv_hook=None,
                    k_group_size=args.quant_group_size,
                    v_group_size=args.quant_group_size)
                metric_label = "f1" if args.eval_metric == "f1" else "accuracy"
                elapsed = time.time() - t0
                entry = {"ppl": None, "mean_kl": None, "n_tokens": None,
                         metric_label: mean_score, "n_total": nt}
                entry.update(gl)
                if args.save_per_example:
                    _per_example_results["fp16"] = per_ex_fp16
                    with open(args.save_per_example, "w") as _f:
                        json.dump(_per_example_results, _f, indent=2)
                print(f"  => {metric_label}={mean_score:.4f}  "
                      f"gen_len={gl['mean_gen_tokens']:.1f}  (n={nt})"
                      + _failure_summary(gl, nt)
                      + f"  ({elapsed:.1f}s)", flush=True)
                results["fp16"] = entry
                save_results(results)
        else:
            # ── fp16 baseline: collect full log distributions for KL reference ──
            print("\n[fp16 baseline]", flush=True)
            t0 = time.time()
            base_lps_per_ex   = []   # list of log_probs per example
            base_dists_per_ex = []   # list of [n_completion, n_vocab] float16
            for ei, (pt, ct, label) in enumerate(examples):
                if args.show_text and _show_this(ei, len(examples)):
                    show_chunk_text(pt, ct, label=label)
                r = strategies.run_structured(lib, ctx, pt, ct, n_vocab,
                                              kv_hook=None,
                                              quantize_prompt_only=False,
                                              return_log_dists=True,
                                              return_top1=_want_diags(ei, len(examples)),
                                              return_diagnostics=_want_diags(ei, len(examples)))
                base_lps_per_ex.append(r.log_probs)
                base_dists_per_ex.append(r.log_dists)
                if _want_diags(ei, len(examples)) and r.top1s:
                    c = _text_per_chunk.setdefault(ei, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                    c["actual_ids"]    = ct[1:len(r.top1s) + 1]
                    c["top1s"]["fp16"]     = r.top1s
                    c["log_probs"]["fp16"] = r.log_probs
                    c["diags"]["fp16"]     = r.diags
                ppl_so_far = math.exp(-np.mean([lp for ex in base_lps_per_ex for lp in ex]))
                print(f"  ex {ei+1}/{len(examples)}: {label} | ppl={ppl_so_far:.4f}", flush=True)
            fp16_ppl = math.exp(-np.mean([lp for ex in base_lps_per_ex for lp in ex]))
            elapsed = time.time() - t0
            print(f"  => PPL={fp16_ppl:.4f}  kl=0.000  ({elapsed:.1f}s)", flush=True)

            if "fp16" in args.quants:
                n_tok = sum(len(lps) for lps in base_lps_per_ex)
                entry = {"ppl": fp16_ppl, "mean_kl": 0.0, "n_tokens": n_tok}
                if args.eval_accuracy:
                    mean_score, nt, per_ex_fp16, gl = eval_accuracy_pass(
                        examples, gold_answers,
                        kv_hook=None,
                        k_group_size=args.quant_group_size,
                        v_group_size=args.quant_group_size)
                    metric_label = "f1" if args.eval_metric == "f1" else "accuracy"
                    entry[metric_label] = mean_score
                    entry["n_total"]    = nt
                    entry.update(gl)
                    if args.save_per_example:
                        _per_example_results["fp16"] = per_ex_fp16
                    print(f"  => PPL={fp16_ppl:.4f}  kl=0.000  {metric_label}={mean_score:.4f}  "
                          f"gen_len={gl['mean_gen_tokens']:.1f}  (n={nt})"
                      + _failure_summary(gl, nt)
                      + f"  ({elapsed:.1f}s)", flush=True)
                results["fp16"] = entry
                save_results(results)

        _bins_data = {}  # quant_name -> {inner_name: tracker dict}; used by --save-bins
        adaptive_gen_done = False

        for quant_name in args.quants:
            if quant_name == "fp16":
                continue
            if kv_prof is not None:
                kv_prof.reset()
            t0 = time.time()
            print(f"\n[{quant_name}]", flush=True)

            k_names, v_names = quant_mod.resolve_quant_layers(quant_name, n_layer)
            k_group_size, v_group_size = get_kv_group_sizes(k_names, v_names, args.quant_group_size)
            _trackers = {}
            def _factory(n, _t=_trackers, _a=args.asym):
                if n not in _t:
                    _t[n] = quant_mod.BinTracker(n, asym=_a)
                return _t[n]
            kv_hook = make_kv_hook(lib, k_names, v_names, args.n_pos_per_embd,
                                   use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                                   default_group_size=args.quant_group_size,
                                   n_sink=args.sink_tokens,
                                   n_recent=args.recent_tokens,
                                   k_sink_names=k_sink_names,
                                   v_sink_names=v_sink_names,
                                   k_recent_names=k_recent_names,
                                   v_recent_names=v_recent_names,
                                   asym=args.asym,
                                   quant_fn_factory=_factory if args.save_bins else None,
                                   profile=kv_prof)

            if args.skip_ppl:
                # ── Skip PPL pass: accuracy only ──────────────────────────────
                valid = []
                if args.adaptive_sim and quant_name != args.verifier_quant:
                    # Build verifier hook once per quant (verifier_quant stays fixed)
                    ver_k_names, ver_v_names = quant_mod.resolve_quant_layers(
                        args.verifier_quant, n_layer)
                    ver_k_gs, ver_v_gs = get_kv_group_sizes(
                        ver_k_names, ver_v_names, args.quant_group_size)
                    ver_hook = make_kv_hook(
                        lib, ver_k_names, ver_v_names, args.n_pos_per_embd,
                        use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                        default_group_size=args.quant_group_size,
                        n_sink=args.sink_tokens,
                        n_recent=args.recent_tokens,
                        k_sink_names=k_sink_names,
                        v_sink_names=v_sink_names,
                        k_recent_names=k_recent_names,
                        v_recent_names=v_recent_names,
                        asym=args.asym, quant_fn_factory=None,
                        profile=kv_prof)
                    adaptive_draft_hook = make_kv_hook(
                        lib, k_names, v_names, args.n_pos_per_embd,
                        use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                        default_group_size=args.quant_group_size,
                        n_sink=args.sink_tokens,
                        n_recent=args.recent_tokens,
                        k_sink_names=k_sink_names,
                        v_sink_names=v_sink_names,
                        k_recent_names=k_recent_names,
                        v_recent_names=v_recent_names,
                        asym=args.asym,
                        quant_fn_factory=_factory if args.save_bins else None,
                        profile=kv_prof)
                    print(f"  [adaptive-sim] draft={quant_name} verifier={args.verifier_quant} "
                          f"window={args.adaptive_window} verify={_adaptive_verify_label(args)} "
                          f"sink={args.sink_tokens} recent={args.recent_tokens}",
                          flush=True)
                    sim_per_ex = run_adaptive_sim(
                        examples, adaptive_draft_hook, ver_hook,
                        k_group_size, v_group_size, ver_k_gs, ver_v_gs)
                    valid = [s for s in sim_per_ex if s is not None]
                    mean_acc_rate   = sum(s["acceptance_rate"] for s in valid) / len(valid) if valid else 0.0
                    mean_draft_frac = sum(s["draft_fraction"]  for s in valid) / len(valid) if valid else 0.0
                    n_all_ok = sum(1 for s in valid if s["first_fail_window"] is None)
                    print(f"  [adaptive-sim] mean_acceptance={mean_acc_rate:.3f}  "
                          f"mean_draft_frac={mean_draft_frac:.3f}  "
                          f"all_ok={n_all_ok}/{len(valid)}", flush=True)
                    if args.save_per_example:
                        _per_example_results[f"{quant_name}__adaptive_sim"] = sim_per_ex
                # ── Adaptive gen: quant rollout windows + verifier replay ─────────
                gen_pregenerated = None
                gen_stats_valid  = []
                if args.adaptive_gen and not adaptive_gen_done and quant_name != "fp16":
                    adaptive_draft_hook_gen = make_kv_hook(
                        lib, k_names, v_names, args.n_pos_per_embd,
                        use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                        default_group_size=args.quant_group_size,
                        n_sink=args.sink_tokens,
                        n_recent=args.recent_tokens,
                        k_sink_names=k_sink_names,
                        v_sink_names=v_sink_names,
                        k_recent_names=k_recent_names,
                        v_recent_names=v_recent_names,
                        asym=args.asym,
                        quant_fn_factory=_factory if args.save_bins else None,
                        profile=kv_prof)
                    ver_vq = args.verifier_quant
                    if ver_vq != "fp16":
                        ver_k_names, ver_v_names = quant_mod.resolve_quant_layers(
                            ver_vq, n_layer)
                        ver_k_gs, ver_v_gs = get_kv_group_sizes(
                            ver_k_names, ver_v_names, args.quant_group_size)
                        ver_hook_gen = make_kv_hook(
                            lib, ver_k_names, ver_v_names, args.n_pos_per_embd,
                            use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                            default_group_size=args.quant_group_size,
                            n_sink=args.sink_tokens,
                            n_recent=args.recent_tokens,
                            k_sink_names=k_sink_names,
                            v_sink_names=v_sink_names,
                            k_recent_names=k_recent_names,
                            v_recent_names=v_recent_names,
                            asym=args.asym, quant_fn_factory=None,
                            profile=kv_prof)
                    else:
                        ver_hook_gen = None
                        ver_k_gs, ver_v_gs = k_group_size, v_group_size
                    ag_use_shadow = (
                        use_gpu and HAS_GPU_KV_SHADOW and not args.no_adaptive_gen_gpu_shadow)
                    cand_specs = [q for q in args.quants if q != "fp16"]
                    print(f"  [adaptive-gen] draft=auto({len(cand_specs)}) verify={ver_vq} "
                          f"bootstrap_window={args.bootstrap_window} rollout_window={args.adaptive_window} "
                          f"sink={args.sink_tokens} recent={args.recent_tokens} "
                          f"kv_ckpt={'gpu_d2d' if ag_use_shadow else 'cpu_blob'}",
                          flush=True)
                    gen_results = run_adaptive_gen(
                        examples, adaptive_draft_hook_gen, k_group_size, v_group_size,
                        use_gpu_shadow=ag_use_shadow, n_layer=n_layer,
                        draft_quant_name=quant_name,
                        ver_hook=ver_hook_gen, ver_k_gs=ver_k_gs, ver_v_gs=ver_v_gs,
                        verifier_quant_name=ver_vq,
                        draft_quant_candidates=cand_specs)
                    adaptive_gen_done = True
                    gen_pregenerated = gen_results
                    gen_stats_valid  = [r for r in gen_results if r is not None]
                    if gen_stats_valid:
                        mg_acc   = sum(r["acceptance_rate"] for r in gen_stats_valid) / len(gen_stats_valid)
                        mg_dfrac = sum(r["draft_fraction"]  for r in gen_stats_valid) / len(gen_stats_valid)
                        print(f"  [adaptive-gen] mean_acceptance={mg_acc:.3f}  "
                              f"mean_draft_frac={mg_dfrac:.3f}", flush=True)
                    if args.save_per_example:
                        _per_example_results[f"{quant_name}__adaptive_gen"] = gen_results
                    if args.qviz_html and gen_results:
                        out_h = args.qviz_html
                        with open(out_h, "w", encoding="utf-8") as hf:
                            hf.write("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
                                     "<title>segment qviz</title>\n")
                            hf.write("<style>body{font-family:system-ui,sans-serif;margin:12px}"
                                     "</style></head><body>\n")
                            for r in gen_results:
                                if r and r.get("segment_quant_viz_html"):
                                    hf.write(r["segment_quant_viz_html"])
                                    hf.write("<hr/>\n")
                            hf.write("</body></html>\n")
                        print(f"  [adaptive-gen] qviz HTML written: {out_h}", flush=True)

                mean_score, nt, per_ex_q, gl = eval_accuracy_pass(
                    examples, gold_answers,
                    kv_hook=kv_hook,
                    k_group_size=k_group_size,
                    v_group_size=v_group_size,
                    pregenerated=gen_pregenerated)
                metric_label = "f1" if args.eval_metric == "f1" else "accuracy"
                elapsed = time.time() - t0
                entry = {"ppl": None, "mean_kl": None, "n_tokens": None,
                         metric_label: mean_score, "n_total": nt}
                entry.update(gl)
                if args.adaptive_sim and quant_name != args.verifier_quant and valid:
                    entry["adaptive_sim"] = {
                        "verifier_quant":    args.verifier_quant,
                        "window_size":       args.adaptive_window,
                        "adaptive_verify":   _adaptive_verify_label(args),
                        "sink_tokens":       args.sink_tokens,
                        "recent_tokens":     args.recent_tokens,
                        "mean_acceptance":   mean_acc_rate,
                        "mean_draft_frac":   mean_draft_frac,
                        "n_all_ok":          n_all_ok,
                        "n_examples":        len(valid),
                    }
                if args.adaptive_gen and quant_name != "fp16" and gen_stats_valid:
                    ag_entry = {
                        "rollout_window":    args.adaptive_window,
                        "bootstrap_window":  args.bootstrap_window,
                        "verifier_quant":    args.verifier_quant,
                        "adaptive_verify":   _adaptive_verify_label(args),
                        "sink_tokens":       args.sink_tokens,
                        "recent_tokens":     args.recent_tokens,
                        "mean_acceptance":   mg_acc,
                        "mean_draft_frac":   mg_dfrac,
                        "n_examples":        len(gen_stats_valid),
                    }
                    entry["adaptive_gen"] = ag_entry
                if args.save_per_example:
                    _per_example_results[quant_name] = per_ex_q
                    with open(args.save_per_example, "w") as _f:
                        json.dump(_per_example_results, _f, indent=2)
                print(f"  => {metric_label}={mean_score:.4f}  "
                      f"gen_len={gl['mean_gen_tokens']:.1f}  (n={nt})"
                      + _failure_summary(gl, nt)
                      + f"  ({elapsed:.1f}s)", flush=True)
            else:
                all_lp = []
                all_kl = []
                for ei, (pt, ct, label) in enumerate(examples):
                    r = strategies.run_structured(lib, ctx, pt, ct, n_vocab,
                                                  kv_hook=kv_hook,
                                                  quantize_prompt_only=args.quantize_prompt_only,
                                                  k_group_size=k_group_size,
                                                  v_group_size=v_group_size,
                                                  base_log_dists=base_dists_per_ex[ei],
                                                  return_top1=_want_diags(ei, len(examples)),
                                                  return_diagnostics=_want_diags(ei, len(examples)))
                    all_lp.extend(r.log_probs)
                    all_kl.extend(r.kl_divs)
                    if _want_diags(ei, len(examples)) and r.top1s:
                        c = _text_per_chunk.setdefault(ei, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                        c["top1s"][quant_name]     = r.top1s
                        c["log_probs"][quant_name] = r.log_probs
                        c["diags"][quant_name]     = r.diags
                    ppl_so_far = math.exp(-sum(all_lp) / len(all_lp))
                    kl_so_far  = sum(all_kl) / len(all_kl)
                    print(f"  ex {ei+1}/{len(examples)}: {label} | "
                          f"ppl={ppl_so_far:.4f}  kl={kl_so_far:.4f}", flush=True)

                ppl     = math.exp(-sum(all_lp) / len(all_lp))
                mean_kl = sum(all_kl) / len(all_kl)
                elapsed = time.time() - t0
                entry = {"ppl": ppl, "mean_kl": mean_kl, "n_tokens": len(all_lp)}
                if args.eval_accuracy:
                    mean_score, nt, per_ex_q, gl = eval_accuracy_pass(
                        examples, gold_answers,
                        kv_hook=kv_hook,
                        k_group_size=k_group_size,
                        v_group_size=v_group_size)
                    metric_label = "f1" if args.eval_metric == "f1" else "accuracy"
                    entry[metric_label] = mean_score
                    entry["n_total"]    = nt
                    entry.update(gl)
                    if args.save_per_example:
                        _per_example_results[quant_name] = per_ex_q
                        with open(args.save_per_example, "w") as _f:
                            json.dump(_per_example_results, _f, indent=2)
                    print(f"  => PPL={ppl:.4f}  kl={mean_kl:.4f}  {metric_label}={mean_score:.4f}  "
                          f"gen_len={gl['mean_gen_tokens']:.1f}  (n={nt})"
                      + _failure_summary(gl, nt)
                      + f"  ({elapsed:.1f}s)", flush=True)
                else:
                    print(f"  => PPL={ppl:.4f}  kl={mean_kl:.4f}  ({elapsed:.1f}s)", flush=True)
            results[quant_name] = entry
            save_results(results)
            if kv_prof is not None:
                profile_agg[quant_name] = kv_prof.to_dict()
                for line in kv_prof.summary_lines():
                    print(line, flush=True)
            if args.save_bins and _trackers:
                _bins_data[quant_name] = {n: t.to_dict() for n, t in _trackers.items()}

        if args.save_bins and _bins_data:
            with open(args.save_bins, "w") as _f:
                json.dump(_bins_data, _f, indent=2)
            print(f"\nBin counts saved to {args.save_bins}", flush=True)
        if args.save_per_example and _per_example_results:
            with open(args.save_per_example, "w") as _f:
                json.dump(_per_example_results, _f, indent=2)
            print(f"Per-example results saved to {args.save_per_example}", flush=True)
        if args.save_diags and _text_per_chunk:
            out = {}
            for idx, cdata in _text_per_chunk.items():
                out[str(idx)] = {"log_probs": cdata["log_probs"], "diags": cdata["diags"]}
            with open(args.save_diags, "w") as _f:
                json.dump(out, _f)
            print(f"Diagnostics saved to {args.save_diags}", flush=True)
        if args.show_text:
            show_predictions()
        if args.profile_kv_out and profile_agg:
            with open(args.profile_kv_out, "w", encoding="utf-8") as f:
                json.dump(profile_agg, f, indent=2)
            print(f"\nKV profile JSON → {args.profile_kv_out}", flush=True)
        lib.llama_free(ctx)
        lib.llama_model_free(model)
        lib.llama_backend_free()
        return

    # ── Flat mode (default) ───────────────────────────────────────────────────
    text = open(args.corpus_file, encoding="utf-8").read()
    all_tokens = llama.tokenize(lib, vocab, text)
    print(f"Total tokens: {len(all_tokens)}", flush=True)

    # ── Multi-window mode: fixed prefill, decode once, PPL at multiple cutoffs ─
    if args.score_windows:
        if not args.prefill_tokens:
            raise ValueError("--score-windows requires --prefill-tokens")
        score_windows = sorted(args.score_windows)
        max_window    = score_windows[-1]
        n_prefill     = args.prefill_tokens
        chunk_size    = n_prefill + max_window

        n_available = (len(all_tokens) - chunk_size) // chunk_size + 1
        n_want = min(args.n_chunks, n_available)
        stride = max(chunk_size, len(all_tokens) // n_want) if n_want > 1 else chunk_size
        chunks = []
        for c in range(n_want):
            start = c * stride
            end   = start + chunk_size
            if end > len(all_tokens):
                break
            chunks.append(all_tokens[start:end])

        print(f"Multi-window mode: {len(chunks)} chunks (strided across corpus), "
              f"prefill={n_prefill}, decode={max_window}, windows={score_windows}", flush=True)

        n_batch = max(n_prefill, 512)
        cparams = lib.llama_context_default_params()
        cparams.n_ctx           = chunk_size
        cparams.n_batch         = n_batch
        cparams.n_ubatch        = min(n_batch, 512)
        cparams.n_threads       = args.n_threads
        cparams.n_threads_batch = args.n_threads
        cparams.no_perf         = True
        if args.flash_attn:
            cparams.flash_attn_type = 1
        ctx = lib.llama_init_from_model(model, cparams)

        # ── fp16 baseline: collect full log distributions for KL reference ──
        print("\n[fp16 baseline]", flush=True)
        t0 = time.time()
        base_lps_per_chunk   = []
        base_dists_per_chunk = []   # [n_chunks][max_window, n_vocab] float16
        for ci, chunk in enumerate(chunks):
            if args.show_text and _show_this(ci, len(chunks)):
                show_chunk_text(chunk[:n_prefill], chunk[n_prefill:],
                                label=f"chunk {ci+1}/{len(chunks)}")
            r = strategies.run_chunk_batch_prefill(
                lib, ctx, chunk, n_vocab, None,
                n_prompt=n_prefill,
                return_log_dists=True,
                return_top1=_want_diags(ci, len(chunks)),
                return_diagnostics=_want_diags(ci, len(chunks)))
            base_lps_per_chunk.append(r.log_probs)
            base_dists_per_chunk.append(r.log_dists)
            if _want_diags(ci, len(chunks)) and r.top1s:
                c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                c["actual_ids"]        = [chunk[n_prefill + t + 1] for t in range(len(r.top1s))]
                c["top1s"]["fp16"]     = r.top1s
                c["log_probs"]["fp16"] = r.log_probs
                c["diags"]["fp16"]     = r.diags
            ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
            print(f"  chunk {ci+1}/{len(chunks)}: ppl@{max_window}={ppl:.4f}", flush=True)
        elapsed = time.time() - t0
        fp16_entry = {}
        for w in score_windows:
            flat = [lp for lps in base_lps_per_chunk for lp in lps[:w]]
            fp16_entry[f"ppl_{w}"] = math.exp(-sum(flat) / len(flat))
        fp16_entry["mean_kl"] = 0.0
        ppl_str = "  ".join(f"ppl@{w}={fp16_entry[f'ppl_{w}']:.4f}" for w in score_windows)
        print(f"  => {ppl_str}  kl=0.000  ({elapsed:.1f}s)", flush=True)

        results = {}
        if "fp16" in args.quants:
            results["fp16"] = fp16_entry
            save_results(results)

        _bins_data = {}

        for quant_name in args.quants:
            if quant_name == "fp16":
                continue
            if kv_prof is not None:
                kv_prof.reset()
            t0 = time.time()
            print(f"\n[{quant_name}]", flush=True)

            k_names, v_names = quant_mod.resolve_quant_layers(quant_name, n_layer)
            k_group_size, v_group_size = get_kv_group_sizes(k_names, v_names, args.quant_group_size)
            _trackers = {}
            def _factory(n, _t=_trackers, _a=args.asym):
                if n not in _t:
                    _t[n] = quant_mod.BinTracker(n, asym=_a)
                return _t[n]
            kv_hook = make_kv_hook(lib, k_names, v_names, args.n_pos_per_embd,
                                   use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                                   default_group_size=args.quant_group_size,
                                   n_sink=args.sink_tokens,
                                   n_recent=args.recent_tokens,
                                   k_sink_names=k_sink_names,
                                   v_sink_names=v_sink_names,
                                   k_recent_names=k_recent_names,
                                   v_recent_names=v_recent_names,
                                   asym=args.asym,
                                   quant_fn_factory=_factory if args.save_bins else None,
                                   profile=kv_prof)

            all_lps = []
            all_kls = []
            for ci, chunk in enumerate(chunks):
                r = strategies.run_chunk_batch_prefill(
                    lib, ctx, chunk, n_vocab, kv_hook,
                    n_prompt=n_prefill,
                    _quantize_prompt_only=args.quantize_prompt_only,
                    k_group_size=k_group_size,
                    v_group_size=v_group_size,
                    base_log_dists=base_dists_per_chunk[ci],
                    return_top1=_want_diags(ci, len(chunks)),
                    return_diagnostics=_want_diags(ci, len(chunks)))
                all_lps.append(r.log_probs)
                all_kls.extend(r.kl_divs)
                if _want_diags(ci, len(chunks)) and r.top1s:
                    c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                    c["top1s"][quant_name]     = r.top1s
                    c["log_probs"][quant_name] = r.log_probs
                    c["diags"][quant_name]     = r.diags
                ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
                kl  = sum(r.kl_divs) / len(r.kl_divs)
                print(f"  chunk {ci+1}/{len(chunks)}: ppl@{max_window}={ppl:.4f}  kl={kl:.4f}", flush=True)

            entry = {}
            for w in score_windows:
                flat = [lp for lps in all_lps for lp in lps[:w]]
                entry[f"ppl_{w}"] = math.exp(-sum(flat) / len(flat))
            entry["mean_kl"] = sum(all_kls) / len(all_kls)
            elapsed = time.time() - t0
            ppl_str = "  ".join(f"ppl@{w}={entry[f'ppl_{w}']:.4f}" for w in score_windows)
            print(f"  => {ppl_str}  kl={entry['mean_kl']:.4f}  ({elapsed:.1f}s)", flush=True)
            results[quant_name] = entry
            save_results(results)
            if kv_prof is not None:
                profile_agg[quant_name] = kv_prof.to_dict()
                for line in kv_prof.summary_lines():
                    print(line, flush=True)
            if args.save_bins and _trackers:
                _bins_data[quant_name] = {n: t.to_dict() for n, t in _trackers.items()}

        if args.save_bins and _bins_data:
            with open(args.save_bins, "w") as _f:
                json.dump(_bins_data, _f, indent=2)
            print(f"\nBin counts saved to {args.save_bins}", flush=True)
        if args.save_per_example and _per_example_results:
            with open(args.save_per_example, "w") as _f:
                json.dump(_per_example_results, _f, indent=2)
            print(f"Per-example results saved to {args.save_per_example}", flush=True)
        if args.save_diags and _text_per_chunk:
            out = {}
            for idx, cdata in _text_per_chunk.items():
                out[str(idx)] = {"log_probs": cdata["log_probs"], "diags": cdata["diags"]}
            with open(args.save_diags, "w") as _f:
                json.dump(out, _f)
            print(f"Diagnostics saved to {args.save_diags}", flush=True)
        if args.show_text:
            show_predictions()
        if args.profile_kv_out and profile_agg:
            with open(args.profile_kv_out, "w", encoding="utf-8") as f:
                json.dump(profile_agg, f, indent=2)
            print(f"\nKV profile JSON → {args.profile_kv_out}", flush=True)
        print(f"\nDone. Results in {args.out}")
        lib.llama_free(ctx)
        lib.llama_model_free(model)
        lib.llama_backend_free()
        return

    # ── Standard flat mode ────────────────────────────────────────────────────
    n_available = (len(all_tokens) - args.n_ctx) // args.n_ctx + 1
    n_want = min(args.n_chunks, n_available)
    stride = max(args.n_ctx, len(all_tokens) // n_want) if n_want > 1 else args.n_ctx
    chunks = []
    for c in range(n_want):
        start = c * stride
        end   = start + args.n_ctx
        if end > len(all_tokens):
            break
        chunks.append(all_tokens[start:end])

    n_measured = args.n_ctx - args.n_prompt - 1
    assert n_measured > 0, f"--n-prompt ({args.n_prompt}) must be < n-ctx-1 ({args.n_ctx - 1})"
    if args.prefill_batch and args.n_prompt < 1:
        raise ValueError("--prefill-batch requires --n-prompt >= 1")
    if args.quantize_prompt_only and not args.prefill_batch:
        raise ValueError("--quantize-prompt-only requires --prefill-batch")
    if args.quantize_prompt_only:
        mode = "batch-prefill+prompt-only-quant"
    elif args.prefill_batch:
        mode = "batch-prefill"
    else:
        mode = "token-by-token"
    print(f"Running {len(chunks)} chunks × {args.n_ctx} tokens each "
          f"(prompt={args.n_prompt} [{mode}], measured={n_measured})", flush=True)

    n_batch = max(args.n_prompt, 512) if args.prefill_batch else 512
    cparams = lib.llama_context_default_params()
    cparams.n_ctx           = args.n_ctx
    cparams.n_batch         = n_batch
    cparams.n_ubatch        = min(n_batch, 512)
    cparams.n_threads       = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.no_perf         = True
    if args.flash_attn:
        cparams.flash_attn_type = 1
    ctx = lib.llama_init_from_model(model, cparams)

    # ── fp16 baseline: collect full log distributions for KL reference ──────
    def _run_one_flat(chunk, kv_hook, k_gs, v_gs, base_dists, ret_dists,
                      ret_top1=False, ret_diags=False):
        if args.prefill_batch:
            return strategies.run_chunk_batch_prefill(
                lib, ctx, chunk, n_vocab, kv_hook,
                n_prompt=args.n_prompt,
                _quantize_prompt_only=args.quantize_prompt_only,
                k_group_size=k_gs, v_group_size=v_gs,
                base_log_dists=base_dists, return_log_dists=ret_dists,
                return_top1=ret_top1, return_diagnostics=ret_diags)
        else:
            return strategies.run_chunk_token_by_token(
                lib, ctx, chunk, n_vocab, kv_hook,
                n_prompt=args.n_prompt,
                k_group_size=k_gs, v_group_size=v_gs,
                base_log_dists=base_dists, return_log_dists=ret_dists,
                return_top1=ret_top1, return_diagnostics=ret_diags)

    print("\n[fp16 baseline]", flush=True)
    t0 = time.time()
    base_lps_per_chunk   = []
    base_dists_per_chunk = []
    for ci, chunk in enumerate(chunks):
        if args.show_text and _show_this(ci, len(chunks)):
            show_chunk_text(chunk[:args.n_prompt], chunk[args.n_prompt:])
        r = _run_one_flat(chunk, None, args.quant_group_size, args.quant_group_size,
                          base_dists=None, ret_dists=True,
                          ret_top1=_want_diags(ci, len(chunks)),
                          ret_diags=_want_diags(ci, len(chunks)))
        base_lps_per_chunk.append(r.log_probs)
        base_dists_per_chunk.append(r.log_dists)
        if args.show_text and _show_this(ci, len(chunks)) and r.top1s:
            c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
            c["actual_ids"]        = [chunk[args.n_prompt + t + 1] for t in range(len(r.top1s))]
            c["top1s"]["fp16"]     = r.top1s
            c["log_probs"]["fp16"] = r.log_probs
            c["diags"]["fp16"]     = r.diags
        ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
        print(f"  chunk {ci+1}/{len(chunks)}: ppl={ppl:.4f}", flush=True)
    all_base_lps = [lp for lps in base_lps_per_chunk for lp in lps]
    fp16_ppl = math.exp(-sum(all_base_lps) / len(all_base_lps))
    elapsed  = time.time() - t0
    print(f"  => PPL={fp16_ppl:.4f}  kl=0.000  ({elapsed:.1f}s)", flush=True)

    results = {}
    if "fp16" in args.quants:
        results["fp16"] = {"ppl": fp16_ppl, "mean_kl": 0.0, "n_tokens": len(all_base_lps)}
        save_results(results)

    _bins_data = {}

    for quant_name in args.quants:
        if quant_name == "fp16":
            continue
        if kv_prof is not None:
            kv_prof.reset()
        t0 = time.time()
        print(f"\n[{quant_name}]", flush=True)

        k_names, v_names = quant_mod.resolve_quant_layers(quant_name, n_layer)
        k_group_size, v_group_size = get_kv_group_sizes(k_names, v_names, args.quant_group_size)
        _trackers = {}
        def _factory(n, _t=_trackers, _a=args.asym):
            if n not in _t:
                _t[n] = quant_mod.BinTracker(n, asym=_a)
            return _t[n]
        kv_hook = make_kv_hook(lib, k_names, v_names, args.n_pos_per_embd,
                               use_gpu=use_gpu, n_layer=n_layer, ctx_ptr=ctx,
                               default_group_size=args.quant_group_size,
                               n_sink=args.sink_tokens,
                               n_recent=args.recent_tokens,
                               k_sink_names=k_sink_names,
                               v_sink_names=v_sink_names,
                               k_recent_names=k_recent_names,
                               v_recent_names=v_recent_names,
                               asym=args.asym,
                               quant_fn_factory=_factory if args.save_bins else None,
                               profile=kv_prof)

        all_lp = []
        all_kl = []
        for ci, chunk in enumerate(chunks):
            r = _run_one_flat(chunk, kv_hook, k_group_size, v_group_size,
                              base_dists=base_dists_per_chunk[ci], ret_dists=False,
                              ret_top1=_want_diags(ci, len(chunks)),
                              ret_diags=_want_diags(ci, len(chunks)))
            all_lp.extend(r.log_probs)
            all_kl.extend(r.kl_divs)
            if _want_diags(ci, len(chunks)) and r.top1s:
                c = _text_per_chunk.setdefault(ci, {"actual_ids": [], "top1s": {}, "log_probs": {}, "diags": {}})
                c["top1s"][quant_name]     = r.top1s
                c["log_probs"][quant_name] = r.log_probs
                c["diags"][quant_name]     = r.diags
            ppl = math.exp(-sum(r.log_probs) / len(r.log_probs))
            kl  = sum(r.kl_divs) / len(r.kl_divs)
            print(f"  chunk {ci+1}/{len(chunks)}: ppl={ppl:.4f}  kl={kl:.4f}", flush=True)

        ppl     = math.exp(-sum(all_lp) / len(all_lp))
        mean_kl = sum(all_kl) / len(all_kl)
        elapsed = time.time() - t0
        print(f"  => PPL={ppl:.4f}  kl={mean_kl:.4f}  ({elapsed:.1f}s)", flush=True)
        results[quant_name] = {"ppl": ppl, "mean_kl": mean_kl, "n_tokens": len(all_lp)}
        save_results(results)
        if kv_prof is not None:
            profile_agg[quant_name] = kv_prof.to_dict()
            for line in kv_prof.summary_lines():
                print(line, flush=True)
        if args.save_bins and _trackers:
            _bins_data[quant_name] = {n: t.to_dict() for n, t in _trackers.items()}

    if args.save_bins and _bins_data:
        with open(args.save_bins, "w") as _f:
            json.dump(_bins_data, _f, indent=2)
        print(f"\nBin counts saved to {args.save_bins}", flush=True)
    if args.show_text:
        show_predictions()
    if args.save_per_example and _per_example_results:
        with open(args.save_per_example, "w") as f:
            json.dump(_per_example_results, f, indent=2)
        print(f"Per-example results saved to {args.save_per_example}")
    if args.save_diags and _text_per_chunk:
        out = {}
        for idx, cdata in _text_per_chunk.items():
            out[str(idx)] = {
                "log_probs": cdata["log_probs"],
                "diags":     cdata["diags"],
            }
        with open(args.save_diags, "w") as f:
            json.dump(out, f)
        print(f"Diagnostics saved to {args.save_diags}")
    if args.profile_kv_out and profile_agg:
        with open(args.profile_kv_out, "w", encoding="utf-8") as f:
            json.dump(profile_agg, f, indent=2)
        print(f"\nKV profile JSON → {args.profile_kv_out}", flush=True)
    print(f"\nDone. Results in {args.out}")

    lib.llama_free(ctx)
    lib.llama_model_free(model)
    lib.llama_backend_free()


if __name__ == "__main__":
    main()
