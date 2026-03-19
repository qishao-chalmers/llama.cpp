#!/usr/bin/env python3
"""
KV Cache Quantization Perplexity Experiment
Baseline: FP16, no modification.
Subsequent experiments will add a kv_hook to modify the KV cache after each decode step.

Usage:
    python3 perplexity.py <model.gguf> <text_file> [--n-ctx 512] [--n-threads 8]
"""

import ctypes
import numpy as np
import math
import sys
import os
import argparse

# Local modules (optional — only needed when --quant is used)
_parse_state = None
_quant       = None

def _load_local_modules():
    global _parse_state, _quant
    if _parse_state is None:
        import importlib.util, pathlib
        _dir = pathlib.Path(__file__).parent
        for name, mod_ref in [("parse_state", "_parse_state"), ("quant", "_quant")]:
            spec = importlib.util.spec_from_file_location(name, _dir / f"{name}.py")
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            globals()[mod_ref] = mod

# ── Library ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH   = os.path.join(SCRIPT_DIR, "../../build_release/bin/libllama.so")

lib = ctypes.CDLL(LIB_PATH)

# ── Opaque pointer types ───────────────────────────────────────────────────────

class _LlamaModel(ctypes.Structure):   pass
class _LlamaContext(ctypes.Structure): pass
class _LlamaVocab(ctypes.Structure):   pass
class _LlamaMemory(ctypes.Structure):  pass

LlamaModelPtr   = ctypes.POINTER(_LlamaModel)
LlamaContextPtr = ctypes.POINTER(_LlamaContext)
LlamaVocabPtr   = ctypes.POINTER(_LlamaVocab)
LlamaMemoryPtr  = ctypes.POINTER(_LlamaMemory)

# ── Structs ────────────────────────────────────────────────────────────────────

class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices",                     ctypes.c_void_p),
        ("tensor_buft_overrides",       ctypes.c_void_p),
        ("n_gpu_layers",                ctypes.c_int32),
        ("split_mode",                  ctypes.c_int32),
        ("main_gpu",                    ctypes.c_int32),
        # 4 bytes auto-padding here to align next pointer to 8 bytes
        ("tensor_split",                ctypes.c_void_p),
        ("progress_callback",           ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides",                ctypes.c_void_p),
        ("vocab_only",                  ctypes.c_bool),
        ("use_mmap",                    ctypes.c_bool),
        ("use_direct_io",               ctypes.c_bool),
        ("use_mlock",                   ctypes.c_bool),
        ("check_tensors",               ctypes.c_bool),
        ("use_extra_bufts",             ctypes.c_bool),
        ("no_host",                     ctypes.c_bool),
        ("no_alloc",                    ctypes.c_bool),
    ]

class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx",               ctypes.c_uint32),
        ("n_batch",             ctypes.c_uint32),
        ("n_ubatch",            ctypes.c_uint32),
        ("n_seq_max",           ctypes.c_uint32),
        ("n_threads",           ctypes.c_int32),
        ("n_threads_batch",     ctypes.c_int32),
        ("rope_scaling_type",   ctypes.c_int32),
        ("pooling_type",        ctypes.c_int32),
        ("attention_type",      ctypes.c_int32),
        ("flash_attn_type",     ctypes.c_int32),
        ("rope_freq_base",      ctypes.c_float),
        ("rope_freq_scale",     ctypes.c_float),
        ("yarn_ext_factor",     ctypes.c_float),
        ("yarn_attn_factor",    ctypes.c_float),
        ("yarn_beta_fast",      ctypes.c_float),
        ("yarn_beta_slow",      ctypes.c_float),
        ("yarn_orig_ctx",       ctypes.c_uint32),
        ("defrag_thold",        ctypes.c_float),
        ("cb_eval",             ctypes.c_void_p),
        ("cb_eval_user_data",   ctypes.c_void_p),
        ("type_k",              ctypes.c_int32),
        ("type_v",              ctypes.c_int32),
        ("abort_callback",      ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings",          ctypes.c_bool),
        ("offload_kqv",         ctypes.c_bool),
        ("no_perf",             ctypes.c_bool),
        ("op_offload",          ctypes.c_bool),
        ("swa_full",            ctypes.c_bool),
        ("kv_unified",          ctypes.c_bool),
        # 2 bytes auto-padding here to align samplers pointer to 8 bytes
        ("samplers",            ctypes.c_void_p),
        ("n_samplers",          ctypes.c_size_t),
    ]

class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",  ctypes.c_int32),
        ("token",     ctypes.POINTER(ctypes.c_int32)),
        ("embd",      ctypes.c_void_p),
        ("pos",       ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id",  ctypes.POINTER(ctypes.c_int32)),
        ("seq_id",    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits",    ctypes.POINTER(ctypes.c_int8)),
    ]

# ── Function signatures ────────────────────────────────────────────────────────

lib.llama_backend_init.restype  = None
lib.llama_backend_init.argtypes = []

lib.llama_backend_free.restype  = None
lib.llama_backend_free.argtypes = []

lib.llama_model_default_params.restype  = LlamaModelParams
lib.llama_model_default_params.argtypes = []

lib.llama_context_default_params.restype  = LlamaContextParams
lib.llama_context_default_params.argtypes = []

lib.llama_model_load_from_file.restype  = LlamaModelPtr
lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, LlamaModelParams]

lib.llama_model_free.restype  = None
lib.llama_model_free.argtypes = [LlamaModelPtr]

lib.llama_init_from_model.restype  = LlamaContextPtr
lib.llama_init_from_model.argtypes = [LlamaModelPtr, LlamaContextParams]

lib.llama_free.restype  = None
lib.llama_free.argtypes = [LlamaContextPtr]

lib.llama_model_get_vocab.restype  = LlamaVocabPtr
lib.llama_model_get_vocab.argtypes = [LlamaModelPtr]

lib.llama_vocab_n_tokens.restype  = ctypes.c_int32
lib.llama_vocab_n_tokens.argtypes = [LlamaVocabPtr]

lib.llama_tokenize.restype  = ctypes.c_int32
lib.llama_tokenize.argtypes = [
    LlamaVocabPtr,
    ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ctypes.c_bool, ctypes.c_bool,
]

lib.llama_batch_init.restype  = LlamaBatch
lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

lib.llama_batch_free.restype  = None
lib.llama_batch_free.argtypes = [LlamaBatch]

lib.llama_decode.restype  = ctypes.c_int32
lib.llama_decode.argtypes = [LlamaContextPtr, LlamaBatch]

lib.llama_get_logits_ith.restype  = ctypes.POINTER(ctypes.c_float)
lib.llama_get_logits_ith.argtypes = [LlamaContextPtr, ctypes.c_int32]

lib.llama_get_memory.restype  = LlamaMemoryPtr
lib.llama_get_memory.argtypes = [LlamaContextPtr]

lib.llama_memory_clear.restype  = None
lib.llama_memory_clear.argtypes = [LlamaMemoryPtr, ctypes.c_bool]

lib.llama_state_get_size.restype  = ctypes.c_size_t
lib.llama_state_get_size.argtypes = [LlamaContextPtr]

lib.llama_state_get_data.restype  = ctypes.c_size_t
lib.llama_state_get_data.argtypes = [LlamaContextPtr, ctypes.c_char_p, ctypes.c_size_t]

lib.llama_state_set_data.restype  = ctypes.c_size_t
lib.llama_state_set_data.argtypes = [LlamaContextPtr, ctypes.c_char_p, ctypes.c_size_t]

# ── Helpers ────────────────────────────────────────────────────────────────────

def log_softmax(logits_np):
    """Numerically stable log-softmax over a 1D float32 array."""
    x = logits_np - logits_np.max()
    return x - np.log(np.exp(x).sum())


def tokenize(vocab, text: str, add_special=False):
    text_bytes = text.encode("utf-8")
    max_tokens = len(text_bytes) + 16
    tokens_buf = (ctypes.c_int32 * max_tokens)()
    n = lib.llama_tokenize(
        vocab,
        text_bytes, len(text_bytes),
        tokens_buf, max_tokens,
        add_special, True,
    )
    if n < 0:
        raise RuntimeError(f"tokenize failed: {n}")
    return list(tokens_buf[:n])


def get_logits(ctx, pos: int, n_vocab: int) -> np.ndarray:
    """Return logits at position pos as a numpy float32 array."""
    ptr = lib.llama_get_logits_ith(ctx, pos)
    return np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()


def decode_batch(ctx, tokens: list, pos_start: int, enable_logits_at: list = None):
    """
    Decode a batch of tokens starting at pos_start.
    enable_logits_at: list of local indices (within the batch) for which to request logits.
                      If None, request logits for the last token only.
    """
    n = len(tokens)
    batch = lib.llama_batch_init(n, 0, 1)
    for i, tok in enumerate(tokens):
        batch.token[i]    = tok
        batch.pos[i]      = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i]   = 0
    if enable_logits_at is None:
        batch.logits[n - 1] = 1
    else:
        for i in enable_logits_at:
            batch.logits[i] = 1
    batch.n_tokens = n
    ret = lib.llama_decode(ctx, batch)
    lib.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"llama_decode failed: {ret}")


# ── Perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexity(ctx, vocab, tokens: list, n_ctx: int, n_vocab: int,
                       kv_hook=None, prefill_token_by_token=False):
    """
    Compute perplexity over tokens.

    kv_hook: optional callable(ctx, n_ctx_tokens_so_far) called after every
             decode step to modify the KV cache in-place.
             Signature: kv_hook(ctx) -> None

    prefill_token_by_token: if True, process prompt one token at a time so
                            that the hook also applies during prefill
                            (Strategy B). If False, batch-prefill then hook
                            (Strategy A).
    """
    N = len(tokens)
    if N < 2:
        raise ValueError("Need at least 2 tokens to compute perplexity.")

    log_probs = []
    mem = lib.llama_get_memory(ctx)
    lib.llama_memory_clear(mem, True)

    if not prefill_token_by_token:
        # ── Strategy A: batch prefill ──────────────────────────────────────────
        # Feed all tokens except the last in one batch, request logits for all.
        prefill = tokens[:-1]
        decode_batch(ctx, prefill, pos_start=0, enable_logits_at=list(range(len(prefill))))
        if kv_hook:
            kv_hook(ctx)

        # Collect log P(token[i+1] | token[0..i]) for i in 0..N-2
        for i, next_tok in enumerate(tokens[1:]):
            logits = get_logits(ctx, i, n_vocab)
            lp     = log_softmax(logits)[next_tok]
            log_probs.append(float(lp))

        print(f"  prefill done, {len(prefill)} tokens", flush=True)

    else:
        # ── Strategy B: token-by-token ─────────────────────────────────────────
        for i in range(N - 1):
            decode_batch(ctx, [tokens[i]], pos_start=i)
            if kv_hook:
                kv_hook(ctx)
            logits = get_logits(ctx, 0, n_vocab)
            lp     = log_softmax(logits)[tokens[i + 1]]
            log_probs.append(float(lp))
            if (i + 1) % 50 == 0:
                ppl_so_far = math.exp(-sum(log_probs) / len(log_probs))
                print(f"  [{i+1}/{N-1}] ppl so far: {ppl_so_far:.4f}", flush=True)

    ppl = math.exp(-sum(log_probs) / len(log_probs))
    return ppl, log_probs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model",     help="Path to .gguf model file")
    parser.add_argument("text_file", help="Path to text file")
    parser.add_argument("--n-ctx",          type=int, default=512)
    parser.add_argument("--n-threads",      type=int, default=8)
    parser.add_argument("--n-gpu-layers",   type=int, default=0)
    parser.add_argument("--n-pos-per-embd", type=int, default=1,
                        help="Set to 4 for M-RoPE models (e.g. Qwen-VL)")
    parser.add_argument("--token-by-token", action="store_true",
                        help="Process prompt token-by-token (Strategy B, slower)")
    parser.add_argument("--quant", default=None,
                        help="Simulated KV quantization: fp16 bf16 fp8_e4m3 fp8_e5m2 "
                             "int8 int8_ch int4 int4_ch nf4 int2")
    parser.add_argument("--quant-target", default="kv", choices=["k", "v", "kv"],
                        help="Which cache to quantize: k, v, or kv (default: kv)")
    args = parser.parse_args()

    # Init backend
    lib.llama_backend_init()

    # Load model
    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = args.n_gpu_layers
    print(f"Loading model: {args.model}", flush=True)
    model = lib.llama_model_load_from_file(args.model.encode(), mparams)
    if not model:
        print("ERROR: failed to load model"); sys.exit(1)

    vocab   = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)
    print(f"Vocab size: {n_vocab}", flush=True)

    # Create context
    cparams = lib.llama_context_default_params()
    cparams.n_ctx           = args.n_ctx
    cparams.n_threads       = args.n_threads
    cparams.n_threads_batch = args.n_threads
    cparams.no_perf         = True
    ctx = lib.llama_init_from_model(model, cparams)
    if not ctx:
        print("ERROR: failed to create context"); sys.exit(1)

    # Tokenize
    text = open(args.text_file, encoding="utf-8").read()
    tokens = tokenize(vocab, text)
    # Clamp to n_ctx tokens
    tokens = tokens[:args.n_ctx]
    print(f"Tokens: {len(tokens)}", flush=True)

    # Build kv_hook if quantization is requested
    kv_hook = None
    if args.quant and args.quant != "fp16":
        _load_local_modules()
        q_fn = _quant.get_quant_fn(args.quant)
        k_fn = q_fn if "k" in args.quant_target else None
        v_fn = q_fn if "v" in args.quant_target else None

        def kv_hook(ctx_arg):
            _parse_state.apply_kv_hook(
                lib, ctx_arg, LlamaContextPtr,
                k_fn=k_fn, v_fn=v_fn, seq_id=0,
                n_pos_per_embd=args.n_pos_per_embd,
            )

    label = args.quant if args.quant else "fp16"
    strategy = "token-by-token" if args.token_by_token else "batch prefill"
    print(f"\nComputing perplexity ({strategy}, quant={label})...", flush=True)

    ppl, _ = compute_perplexity(
        ctx, vocab, tokens,
        n_ctx=args.n_ctx,
        n_vocab=n_vocab,
        kv_hook=kv_hook,
        prefill_token_by_token=args.token_by_token,
    )
    print(f"\nPerplexity ({label}): {ppl:.4f}")

    # Cleanup
    lib.llama_free(ctx)
    lib.llama_model_free(model)
    lib.llama_backend_free()


if __name__ == "__main__":
    main()
