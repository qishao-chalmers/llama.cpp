#!/usr/bin/env python3
"""
visualize_kv_outliers.py — Show outlier structure in FP16 KV cache.

Fills KV cache with 64 tokens, extracts K and V arrays for every layer,
and produces a figure showing per-dimension value ranges and value histograms
to make the outlier phenomenon visible.

Usage:
    python3 research/poc/visualize_kv_outliers.py models/Qwen3-8B-Q8_0.gguf \
        --text research/poc/wikitext2_1chunk.txt \
        --out research/poc/kv_outliers.png
"""

import argparse, ctypes, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH   = os.path.join(SCRIPT_DIR, "../../build_release/bin/libllama.so")
sys.path.insert(0, SCRIPT_DIR)

from parse_state import parse_kv_state, _setup_state_funcs

# ── ctypes bindings (minimal, same layout as run_sweep.py) ───────────────────

lib = ctypes.CDLL(LIB_PATH)

class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices",ctypes.c_void_p),("tensor_buft_overrides",ctypes.c_void_p),
        ("n_gpu_layers",ctypes.c_int32),("split_mode",ctypes.c_int32),("main_gpu",ctypes.c_int32),
        ("tensor_split",ctypes.c_void_p),("progress_callback",ctypes.c_void_p),
        ("progress_callback_user_data",ctypes.c_void_p),("kv_overrides",ctypes.c_void_p),
        ("vocab_only",ctypes.c_bool),("use_mmap",ctypes.c_bool),("use_direct_io",ctypes.c_bool),
        ("use_mlock",ctypes.c_bool),("check_tensors",ctypes.c_bool),("use_extra_bufts",ctypes.c_bool),
        ("no_host",ctypes.c_bool),("no_alloc",ctypes.c_bool),
    ]

class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx",ctypes.c_uint32),("n_batch",ctypes.c_uint32),("n_ubatch",ctypes.c_uint32),("n_seq_max",ctypes.c_uint32),
        ("n_threads",ctypes.c_int32),("n_threads_batch",ctypes.c_int32),
        ("rope_scaling_type",ctypes.c_int32),("pooling_type",ctypes.c_int32),
        ("attention_type",ctypes.c_int32),("flash_attn_type",ctypes.c_int32),
        ("rope_freq_base",ctypes.c_float),("rope_freq_scale",ctypes.c_float),
        ("yarn_ext_factor",ctypes.c_float),("yarn_attn_factor",ctypes.c_float),
        ("yarn_beta_fast",ctypes.c_float),("yarn_beta_slow",ctypes.c_float),
        ("yarn_orig_ctx",ctypes.c_uint32),("defrag_thold",ctypes.c_float),
        ("cb_eval",ctypes.c_void_p),("cb_eval_user_data",ctypes.c_void_p),
        ("type_k",ctypes.c_int32),("type_v",ctypes.c_int32),
        ("abort_callback",ctypes.c_void_p),("abort_callback_data",ctypes.c_void_p),
        ("embeddings",ctypes.c_bool),("offload_kqv",ctypes.c_bool),("no_perf",ctypes.c_bool),("op_offload",ctypes.c_bool),
        ("swa_full",ctypes.c_bool),("kv_unified",ctypes.c_bool),
        ("samplers",ctypes.c_void_p),("n_samplers",ctypes.c_size_t),
    ]

class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",ctypes.c_int32),("token",ctypes.POINTER(ctypes.c_int32)),
        ("embd",ctypes.c_void_p),("pos",ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id",ctypes.POINTER(ctypes.c_int32)),
        ("seq_id",ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits",ctypes.POINTER(ctypes.c_int8)),
    ]

class _M(ctypes.Structure): pass
class _C(ctypes.Structure): pass
class _V(ctypes.Structure): pass
class _Mem(ctypes.Structure): pass
MP=ctypes.POINTER(_M); CP=ctypes.POINTER(_C); VP=ctypes.POINTER(_V); MemP=ctypes.POINTER(_Mem)

lib.llama_backend_init.restype=None;           lib.llama_backend_init.argtypes=[]
lib.llama_model_default_params.restype=LlamaModelParams;   lib.llama_model_default_params.argtypes=[]
lib.llama_context_default_params.restype=LlamaContextParams; lib.llama_context_default_params.argtypes=[]
lib.llama_model_load_from_file.restype=MP;     lib.llama_model_load_from_file.argtypes=[ctypes.c_char_p,LlamaModelParams]
lib.llama_model_free.restype=None;             lib.llama_model_free.argtypes=[MP]
lib.llama_init_from_model.restype=CP;          lib.llama_init_from_model.argtypes=[MP,LlamaContextParams]
lib.llama_free.restype=None;                   lib.llama_free.argtypes=[CP]
lib.llama_model_get_vocab.restype=VP;          lib.llama_model_get_vocab.argtypes=[MP]
lib.llama_tokenize.restype=ctypes.c_int32
lib.llama_tokenize.argtypes=[VP,ctypes.c_char_p,ctypes.c_int32,ctypes.POINTER(ctypes.c_int32),ctypes.c_int32,ctypes.c_bool,ctypes.c_bool]
lib.llama_batch_init.restype=LlamaBatch;       lib.llama_batch_init.argtypes=[ctypes.c_int32,ctypes.c_int32,ctypes.c_int32]
lib.llama_batch_free.restype=None;             lib.llama_batch_free.argtypes=[LlamaBatch]
lib.llama_decode.restype=ctypes.c_int32;       lib.llama_decode.argtypes=[CP,LlamaBatch]
lib.llama_get_memory.restype=MemP;             lib.llama_get_memory.argtypes=[CP]
lib.llama_memory_clear.restype=None;           lib.llama_memory_clear.argtypes=[MemP,ctypes.c_bool]
lib.llama_backend_free.restype=None;           lib.llama_backend_free.argtypes=[]
_setup_state_funcs(lib, CP)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fill_kv_cache(model_path, text, n_ctx=128, n_threads=8, n_gpu_layers=0):
    """Prefill model with `text` and return the parsed KVState."""
    lib.llama_backend_init()
    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = n_gpu_layers
    model = lib.llama_model_load_from_file(model_path.encode(), mparams)
    vocab  = lib.llama_model_get_vocab(model)

    cparams = lib.llama_context_default_params()
    cparams.n_ctx = n_ctx
    cparams.n_threads = cparams.n_threads_batch = n_threads
    cparams.no_perf = True
    ctx = lib.llama_init_from_model(model, cparams)

    # Tokenize (allocate large buffer, then truncate to n_ctx)
    text_b = text.encode("utf-8")
    big_buf = (ctypes.c_int32 * 65536)()
    n_all = lib.llama_tokenize(vocab, text_b, len(text_b), big_buf, 65536, False, True)
    assert n_all > 0, f"tokenize failed: {n_all}"
    n_tok = min(n_all, n_ctx)
    tokens = list(big_buf[:n_tok])
    print(f"Prefilling with {n_tok} tokens...", flush=True)

    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)

    batch = lib.llama_batch_init(n_tok, 0, 1)
    batch.n_tokens = n_tok
    for i, t in enumerate(tokens):
        batch.token[i] = t
        batch.pos[i]   = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 0
    batch.logits[n_tok - 1] = 1
    ret = lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)
    assert ret == 0, f"decode failed: {ret}"

    # Extract state
    sz  = lib.llama_state_seq_get_size(ctx, 0)
    raw = ctypes.create_string_buffer(sz)
    n_w = lib.llama_state_seq_get_data(ctx, raw, sz, 0)
    state = parse_kv_state(bytes(raw[:n_w]))
    state._model_name = os.path.basename(model_path).replace(".gguf", "")
    print(f"KV state: cell_count={state.cell_count}, n_layer={state.n_layer}, "
          f"K shape={state.k[0].shape}, V shape={state.v[0].shape}", flush=True)

    lib.llama_free(ctx)
    lib.llama_model_free(model)
    lib.llama_backend_free()
    return state


def collect_per_dim_stats(arrays):
    """
    Given list of [cell_count, dim] arrays (one per layer),
    return per-dimension max-abs aggregated across all layers.
    Shape: (dim,)
    """
    # Stack all layers: (n_layer * cell_count, dim)
    stacked = np.concatenate([a.astype(np.float32) for a in arrays], axis=0)
    return np.abs(stacked).max(axis=0)   # (dim,)


def all_values(arrays):
    """Flatten all values from all layers into one 1-D float32 array."""
    return np.concatenate([a.astype(np.float32).ravel() for a in arrays])


# ── Figure ────────────────────────────────────────────────────────────────────

def make_outlier_figure(state: "KVState", out_path: str):
    k_maxabs = collect_per_dim_stats(state.k)   # (k_dim,)
    v_maxabs = collect_per_dim_stats(state.v)   # (v_dim,)
    k_vals   = all_values(state.k)
    v_vals   = all_values(state.v)

    n_layer, cell_count = state.n_layer, state.cell_count
    k_dim = state.k[0].shape[1]
    v_dim = state.v[0].shape[1]

    # Identify outlier threshold: dims above 3× median max-abs
    k_median = np.median(k_maxabs)
    v_median = np.median(v_maxabs)
    k_outlier_thresh = 3.0 * k_median
    v_outlier_thresh = 3.0 * v_median
    k_outlier_mask = k_maxabs > k_outlier_thresh
    v_outlier_mask = v_maxabs > v_outlier_thresh
    n_k_out = k_outlier_mask.sum()
    n_v_out = v_outlier_mask.sum()
    print(f"K outlier dims (>{k_outlier_thresh:.2f}): {n_k_out}/{k_dim}")
    print(f"V outlier dims (>{v_outlier_thresh:.2f}): {n_v_out}/{v_dim}")

    # Collect values specifically from outlier vs normal dims, across all layers
    k_out_vals = np.concatenate([a.astype(np.float32)[:, k_outlier_mask].ravel() for a in state.k])
    k_nrm_vals = np.concatenate([a.astype(np.float32)[:, ~k_outlier_mask].ravel() for a in state.k])
    v_out_vals = np.concatenate([a.astype(np.float32)[:, v_outlier_mask].ravel() for a in state.v])
    v_nrm_vals = np.concatenate([a.astype(np.float32)[:, ~v_outlier_mask].ravel() for a in state.v])

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    model_tag = getattr(state, "_model_name", "unknown-model")
    fig.suptitle(
        f"KV Cache Outlier Structure — {model_tag} · FP16 · {cell_count} tokens · {n_layer} layers\n"
        f"K dim={k_dim}, V dim={v_dim}",
        fontsize=12, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: K per-dim max-abs (sorted) ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_k = np.sort(k_maxabs)[::-1]
    colors_k = ["#F44336" if v > k_outlier_thresh else "#90CAF9" for v in sorted_k]
    ax1.bar(np.arange(k_dim), sorted_k, color=colors_k, width=1.0, linewidth=0)
    ax1.axhline(k_outlier_thresh, color="#F44336", linestyle="--", linewidth=1,
                label=f"3× median = {k_outlier_thresh:.2f}  ({n_k_out} outlier dims)")
    ax1.set_xlabel("Dimension (sorted by max |value|)", fontsize=9)
    ax1.set_ylabel("Max |value| across all layers & tokens", fontsize=9)
    ax1.set_title("K Cache: per-dimension max magnitude", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, k_dim)

    # ── Panel 2: V per-dim max-abs (sorted) ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_v = np.sort(v_maxabs)[::-1]
    colors_v = ["#F44336" if v > v_outlier_thresh else "#A5D6A7" for v in sorted_v]
    ax2.bar(np.arange(v_dim), sorted_v, color=colors_v, width=1.0, linewidth=0)
    ax2.axhline(v_outlier_thresh, color="#F44336", linestyle="--", linewidth=1,
                label=f"3× median = {v_outlier_thresh:.2f}  ({n_v_out} outlier dims)")
    ax2.set_xlabel("Dimension (sorted by max |value|)", fontsize=9)
    ax2.set_ylabel("Max |value| across all layers & tokens", fontsize=9)
    ax2.set_title("V Cache: per-dimension max magnitude", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, v_dim)

    # ── Panel 3: K value histogram — all values ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    clip = np.percentile(np.abs(k_vals), 99.9)
    bins = np.linspace(-clip, clip, 120)
    ax3.hist(k_nrm_vals.clip(-clip, clip), bins=bins, color="#90CAF9", alpha=0.8,
             label=f"Normal dims ({k_dim - n_k_out})", density=True, linewidth=0)
    ax3.hist(k_out_vals.clip(-clip, clip), bins=bins, color="#F44336", alpha=0.7,
             label=f"Outlier dims ({n_k_out})", density=True, linewidth=0)
    ax3.set_xlabel("Value", fontsize=9)
    ax3.set_ylabel("Density", fontsize=9)
    ax3.set_title("K Cache: value distribution (outlier vs normal dims)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.set_yscale("log")

    # ── Panel 4: V value histogram ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    clip_v = np.percentile(np.abs(v_vals), 99.9)
    bins_v = np.linspace(-clip_v, clip_v, 120)
    ax4.hist(v_nrm_vals.clip(-clip_v, clip_v), bins=bins_v, color="#A5D6A7", alpha=0.8,
             label=f"Normal dims ({v_dim - n_v_out})", density=True, linewidth=0)
    ax4.hist(v_out_vals.clip(-clip_v, clip_v), bins=bins_v, color="#F44336", alpha=0.7,
             label=f"Outlier dims ({n_v_out})", density=True, linewidth=0)
    ax4.set_xlabel("Value", fontsize=9)
    ax4.set_ylabel("Density", fontsize=9)
    ax4.set_title("V Cache: value distribution (outlier vs normal dims)", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.set_yscale("log")

    # ── Panel 5: per-layer max-abs for K (heatmap of outlier severity) ─────────
    ax5 = fig.add_subplot(gs[2, 0])
    # For each layer, compute max-abs per dim → shape (n_layer, k_dim)
    layer_k_maxabs = np.stack([np.abs(a.astype(np.float32)).max(axis=0) for a in state.k])
    im5 = ax5.imshow(layer_k_maxabs, aspect="auto", cmap="hot",
                     interpolation="nearest", vmin=0)
    plt.colorbar(im5, ax=ax5, label="max |value|")
    ax5.set_xlabel("Feature dimension", fontsize=9)
    ax5.set_ylabel("Layer", fontsize=9)
    ax5.set_title("K Cache: max magnitude per (layer, dim) — outlier dims glow", fontsize=10)

    # ── Panel 6: per-layer max-abs for V ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    layer_v_maxabs = np.stack([np.abs(a.astype(np.float32)).max(axis=0) for a in state.v])
    im6 = ax6.imshow(layer_v_maxabs, aspect="auto", cmap="hot",
                     interpolation="nearest", vmin=0)
    plt.colorbar(im6, ax=ax6, label="max |value|")
    ax6.set_xlabel("Feature dimension", fontsize=9)
    ax6.set_ylabel("Layer", fontsize=9)
    ax6.set_title("V Cache: max magnitude per (layer, dim)", fontsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--text",     default="research/poc/wikitext2_1chunk.txt")
    parser.add_argument("--n-ctx",        type=int, default=128)
    parser.add_argument("--n-threads",    type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--out",          default="research/poc/kv_outliers.png")
    args = parser.parse_args()

    text = open(args.text, encoding="utf-8").read()
    state = fill_kv_cache(args.model, text, n_ctx=args.n_ctx, n_threads=args.n_threads,
                          n_gpu_layers=args.n_gpu_layers)
    make_outlier_figure(state, args.out)


if __name__ == "__main__":
    main()
