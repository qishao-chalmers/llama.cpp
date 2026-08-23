#!/usr/bin/env python3
"""profile_layer_timing.py — Per-layer GPU time during native llama.cpp decode.

Uses llama_context_params.cb_eval (ggml_backend_sched_eval_callback): with ask=True
for every graph node, the backend scheduler runs one node per batch and synchronizes
after each node, so elapsed ggml_time_us() between consecutive callbacks approximates
per-op GPU time. Tensor names follow llama.cpp's graph_get_cb(): layer tensors are
named like ``norm-0``, ``kq_soft_max-7``, ``ffn_down-31`` (``{name}-{il}``).

**Important**
  - This is intentionally slow (one sync per graph node). Use small ``--decode-len``.
  - Sum of per-layer times is an attribution of node-level timings; wall decode time
    also includes scheduler overhead not tied to a ``-*-N`` tensor (see ``other_ms``).
  - For MoE / unusual archs, verify tensor name patterns in a short run.

Example::

    python3 research/scripts/profile_layer_timing.py models/Qwen3-8B-Q8_0.gguf \\
        --n-gpu-layers 99 --flash-attn --prompt-len 256 --decode-len 8 \\
        --out research/results/layer_profile.json
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import re
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import llama_bindings as llama

PREFILL_CHUNK = 512
_LAYER_SUFFIX_RE = re.compile(r"^(.+)-(\d+)$")


@contextlib.contextmanager
def _suppress_c_stderr():
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def _prefill(lib, ctx, n_tokens: int, seq_id: int = 0):
    pos = 0
    while pos < n_tokens:
        chunk = min(PREFILL_CHUNK, n_tokens - pos)
        batch = lib.llama_batch_init(chunk, 0, 1)
        for i in range(chunk):
            batch.token[i] = 1
            batch.pos[i] = pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i] = 0
        batch.n_tokens = chunk
        lib.llama_decode(ctx, batch)
        lib.llama_batch_free(batch)
        pos += chunk


def _decode_one(lib, ctx, pos: int, seq_id: int = 0):
    batch = lib.llama_batch_init(1, 0, 1)
    batch.n_tokens = 1
    batch.token[0] = 1
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = seq_id
    batch.logits[0] = 1
    lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)


class _LayerProfileState:
    """Mutable state + ctypes eval callback (keep .cb_ref alive)."""

    __slots__ = (
        "lib", "n_layer", "layer_us", "layer_nodes", "other_us", "other_nodes",
        "last_us", "seen_first", "_cb",
    )

    def __init__(self, lib, n_layer: int):
        self.lib = lib
        self.n_layer = n_layer
        self.layer_us = [0] * n_layer
        self.layer_nodes = [0] * n_layer
        self.other_us = 0
        self.other_nodes = 0
        self.last_us: int | None = None
        self.seen_first = False
        self._cb = llama.EVAL_CB_TYPE(self._eval_cb)

    def reset(self):
        self.layer_us = [0] * self.n_layer
        self.layer_nodes = [0] * self.n_layer
        self.other_us = 0
        self.other_nodes = 0
        self.last_us = None
        self.seen_first = False

    def _eval_cb(self, t, ask, _userdata):
        if ask:
            # True => stop batching at this node (one node per compute + sync).
            return True
        now = int(self.lib.ggml_time_us())
        raw = self.lib.ggml_get_name(t)
        if raw is None:
            name = ""
        elif isinstance(raw, bytes):
            name = raw.decode("utf-8", errors="replace")
        else:
            name = str(raw)

        if not self.seen_first:
            self.seen_first = True
            self.last_us = now
            return True

        delta = now - int(self.last_us)
        self.last_us = now

        m = _LAYER_SUFFIX_RE.match(name)
        if m:
            il = int(m.group(2))
            if 0 <= il < self.n_layer:
                self.layer_us[il] += delta
                self.layer_nodes[il] += 1
            else:
                self.other_us += delta
                self.other_nodes += 1
        else:
            self.other_us += delta
            self.other_nodes += 1
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="GGUF path")
    ap.add_argument("--lib", default=None, help="Path to libllama.so")
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--n-warmup", type=int, default=2,
                    help="Decode steps with profiler enabled but discarded (default: 2)")
    ap.add_argument("--decode-len", type=int, default=8,
                    help="Timed decode steps (keep small; profiling is slow)")
    ap.add_argument("--kv-type", default="f16", choices=["f16", "q8_0", "q5_0", "q4_0", "q4_1"])
    ap.add_argument("--out", default=None, help="Write JSON summary path")
    args = ap.parse_args()

    kv_map = {"f16": 1, "q8_0": 8, "q5_0": 6, "q4_0": 2, "q4_1": 3}
    kv_id = kv_map[args.kv_type]

    lib_path = args.lib or os.path.join(_SCRIPT_DIR, "../../build_release/bin/libllama.so")
    lib = llama.load_lib(lib_path)
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    with _suppress_c_stderr():
        model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    if not model:
        print("Failed to load model", file=sys.stderr)
        sys.exit(1)

    n_layer = int(lib.llama_model_n_layer(model))
    prof = _LayerProfileState(lib, n_layer)

    seq_slots = args.prompt_len + args.n_warmup + args.decode_len + 32
    cparams = lib.llama_context_default_params()
    cparams.n_ctx = seq_slots
    cparams.n_seq_max = 1
    cparams.n_batch = max(512, 1)
    cparams.n_ubatch = max(512, 1)
    cparams.n_threads = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.no_perf = True
    cparams.type_k = kv_id
    cparams.type_v = kv_id
    cparams.cb_eval = ctypes.cast(prof._cb, ctypes.c_void_p)
    cparams.cb_eval_user_data = None
    if args.flash_attn:
        cparams.flash_attn_type = 1

    with _suppress_c_stderr():
        ctx = lib.llama_init_from_model(model, cparams)
    if not ctx:
        print("Failed to create context", file=sys.stderr)
        lib.llama_model_free(model)
        sys.exit(1)

    _prefill(lib, ctx, args.prompt_len)
    pos = args.prompt_len

    for _ in range(args.n_warmup):
        prof.reset()
        _decode_one(lib, ctx, pos)
        pos += 1

    total_layer_us = [0] * n_layer
    total_other_us = 0
    wall_s_decode = 0.0

    for _ in range(args.decode_len):
        prof.reset()
        t0 = time.perf_counter()
        _decode_one(lib, ctx, pos)
        wall_s_decode += time.perf_counter() - t0
        pos += 1
        for i in range(n_layer):
            total_layer_us[i] += prof.layer_us[i]
        total_other_us += prof.other_us

    lib.llama_free(ctx)
    lib.llama_model_free(model)

    denom = float(args.decode_len)
    layer_ms = [total_layer_us[i] / denom / 1000.0 for i in range(n_layer)]
    other_ms = total_other_us / denom / 1000.0
    sum_layers_ms = sum(layer_ms)
    wall_ms_tok = (wall_s_decode / denom) * 1000.0

    out = {
        "model": os.path.abspath(args.model),
        "n_layer": n_layer,
        "kv_type": args.kv_type,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "n_warmup": args.n_warmup,
        "flash_attn": bool(args.flash_attn),
        "mean_wall_ms_per_decode_step": wall_ms_tok,
        "mean_other_ms_per_decode_step": other_ms,
        "mean_sum_layer_ms_per_decode_step": sum_layers_ms,
        "layer_ms_per_decode_step": layer_ms,
        "layer_fraction_of_sum_layers": [
            (layer_ms[i] / sum_layers_ms) if sum_layers_ms > 0 else 0.0
            for i in range(n_layer)
        ],
    }

    print(f"n_layer={n_layer}  decode_len={args.decode_len}  kv={args.kv_type}")
    print(f"mean wall (perf_counter) ms/step: {wall_ms_tok:.4f}")
    print(f"mean Σ layer ms/step (from cb_eval): {sum_layers_ms:.4f}  other_ms/step: {other_ms:.4f}")
    print("layer :  ms/step  fraction(of Σ layers)")
    for i in range(n_layer):
        frac = layer_ms[i] / sum_layers_ms if sum_layers_ms > 0 else 0.0
        print(f"  {i:3d} : {layer_ms[i]:8.4f}  {frac:7.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
