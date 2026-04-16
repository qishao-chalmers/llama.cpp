"""
llama_bindings.py — ctypes structs, pointer types, lib loading, and helpers.

Provides:
  - LlamaModelParams, LlamaContextParams, LlamaBatch structs
  - Opaque pointer types: ModelPtr, ContextPtr, VocabPtr, MemoryPtr
  - setup_lib(lib)   — sets all restype/argtypes on a loaded CDLL
  - load_lib(path)   — loads the .so and calls setup_lib
  - log_softmax(x)   — numerically stable log-softmax
  - tokenize(lib, vocab, text, buf_size) — tokenize text, return list of ids
"""

import ctypes
import os
import numpy as np

DEFAULT_LIB_PATH = os.path.join(os.path.dirname(__file__), "../../build_release/bin/libllama.so")


# ── Structs ────────────────────────────────────────────────────────────────────

class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices",                      ctypes.c_void_p),
        ("tensor_buft_overrides",        ctypes.c_void_p),
        ("n_gpu_layers",                 ctypes.c_int32),
        ("split_mode",                   ctypes.c_int32),
        ("main_gpu",                     ctypes.c_int32),
        ("tensor_split",                 ctypes.c_void_p),
        ("progress_callback",            ctypes.c_void_p),
        ("progress_callback_user_data",  ctypes.c_void_p),
        ("kv_overrides",                 ctypes.c_void_p),
        ("vocab_only",                   ctypes.c_bool),
        ("use_mmap",                     ctypes.c_bool),
        ("use_direct_io",                ctypes.c_bool),
        ("use_mlock",                    ctypes.c_bool),
        ("check_tensors",                ctypes.c_bool),
        ("use_extra_bufts",              ctypes.c_bool),
        ("no_host",                      ctypes.c_bool),
        ("no_alloc",                     ctypes.c_bool),
    ]


class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx",                ctypes.c_uint32),
        ("n_batch",              ctypes.c_uint32),
        ("n_ubatch",             ctypes.c_uint32),
        ("n_seq_max",            ctypes.c_uint32),
        ("n_threads",            ctypes.c_int32),
        ("n_threads_batch",      ctypes.c_int32),
        ("rope_scaling_type",    ctypes.c_int32),
        ("pooling_type",         ctypes.c_int32),
        ("attention_type",       ctypes.c_int32),
        ("flash_attn_type",      ctypes.c_int32),
        ("rope_freq_base",       ctypes.c_float),
        ("rope_freq_scale",      ctypes.c_float),
        ("yarn_ext_factor",      ctypes.c_float),
        ("yarn_attn_factor",     ctypes.c_float),
        ("yarn_beta_fast",       ctypes.c_float),
        ("yarn_beta_slow",       ctypes.c_float),
        ("yarn_orig_ctx",        ctypes.c_uint32),
        ("defrag_thold",         ctypes.c_float),
        ("cb_eval",              ctypes.c_void_p),
        ("cb_eval_user_data",    ctypes.c_void_p),
        ("type_k",               ctypes.c_int32),
        ("type_v",               ctypes.c_int32),
        ("abort_callback",       ctypes.c_void_p),
        ("abort_callback_data",  ctypes.c_void_p),
        ("embeddings",           ctypes.c_bool),
        ("offload_kqv",          ctypes.c_bool),
        ("no_perf",              ctypes.c_bool),
        ("op_offload",           ctypes.c_bool),
        ("swa_full",             ctypes.c_bool),
        ("kv_unified",           ctypes.c_bool),
        ("samplers",             ctypes.c_void_p),
        ("n_samplers",           ctypes.c_size_t),
        ("split2_mode_init",     ctypes.c_int32),  # 0=verify 1=draft; must set before init for split2 MMVQ debug
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


class LlamaKVLayerInfo(ctypes.Structure):
    """Per-layer KV cache layout returned by llama_get_kv_layer_info().
    k_data / v_data are raw CUDA device pointers (or CPU pointers for CPU backend).
    Wrap with cp.cuda.UnownedMemory for zero-copy CuPy access.
    Requires Flash Attention (v_trans == 0) for clean V layout.
    """
    _fields_ = [
        ("k_data",    ctypes.c_void_p),   # device ptr: K tensor start for stream 0
        ("v_data",    ctypes.c_void_p),   # device ptr: V tensor start for stream 0
        ("n_cells",   ctypes.c_int32),    # tokens currently in the KV cache
        ("k_stride",  ctypes.c_int32),    # bytes between token rows in K (nb[1])
        ("v_stride",  ctypes.c_int32),    # bytes between token rows in V (nb[1])
        ("n_embd_k",  ctypes.c_int32),    # K embedding dim (ne[0])
        ("n_embd_v",  ctypes.c_int32),    # V embedding dim (ne[0])
        ("ggml_type", ctypes.c_int32),    # ggml_type: 1=F16, 30=BF16
        ("v_trans",   ctypes.c_int32),    # 1 if V transposed (FA disabled)
    ]


# ── Opaque pointer types ───────────────────────────────────────────────────────

class _LlamaModel(ctypes.Structure):   pass
class _LlamaContext(ctypes.Structure): pass
class _LlamaVocab(ctypes.Structure):   pass
class _LlamaMemory(ctypes.Structure):  pass

ModelPtr   = ctypes.POINTER(_LlamaModel)
ContextPtr = ctypes.POINTER(_LlamaContext)
VocabPtr   = ctypes.POINTER(_LlamaVocab)
MemoryPtr  = ctypes.POINTER(_LlamaMemory)


# ── Lib setup ─────────────────────────────────────────────────────────────────

def setup_lib(lib):
    """Set all restype/argtypes on a loaded CDLL instance."""
    # Logging
    lib.llama_log_set.restype  = None
    lib.llama_log_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # Backend
    lib.llama_backend_init.restype  = None
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_free.restype  = None
    lib.llama_backend_free.argtypes = []

    # Model
    lib.llama_model_default_params.restype  = LlamaModelParams
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_load_from_file.restype  = ModelPtr
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, LlamaModelParams]
    lib.llama_model_free.restype  = None
    lib.llama_model_free.argtypes = [ModelPtr]
    lib.llama_model_get_vocab.restype  = VocabPtr
    lib.llama_model_get_vocab.argtypes = [ModelPtr]
    lib.llama_model_n_layer.restype  = ctypes.c_int32
    lib.llama_model_n_layer.argtypes = [ModelPtr]

    # Vocab / tokenize
    lib.llama_vocab_n_tokens.restype  = ctypes.c_int32
    lib.llama_vocab_n_tokens.argtypes = [VocabPtr]
    lib.llama_vocab_eos.restype  = ctypes.c_int32
    lib.llama_vocab_eos.argtypes = [VocabPtr]
    lib.llama_vocab_eot.restype  = ctypes.c_int32
    lib.llama_vocab_eot.argtypes = [VocabPtr]
    lib.llama_vocab_is_eog.restype  = ctypes.c_bool
    lib.llama_vocab_is_eog.argtypes = [VocabPtr, ctypes.c_int32]
    lib.llama_tokenize.restype  = ctypes.c_int32
    lib.llama_tokenize.argtypes = [
        VocabPtr, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    lib.llama_detokenize.restype  = ctypes.c_int32
    lib.llama_detokenize.argtypes = [
        VocabPtr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype  = ctypes.c_int32
    lib.llama_token_to_piece.argtypes = [
        VocabPtr, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_bool,
    ]

    # Context
    lib.llama_context_default_params.restype  = LlamaContextParams
    lib.llama_context_default_params.argtypes = []
    lib.llama_init_from_model.restype  = ContextPtr
    lib.llama_init_from_model.argtypes = [ModelPtr, LlamaContextParams]
    lib.llama_free.restype  = None
    lib.llama_free.argtypes = [ContextPtr]

    # Experimental split2 mode switch (0=verify, 1=draft)
    if hasattr(lib, "llama_set_split2_mode"):
        lib.llama_set_split2_mode.restype  = None
        lib.llama_set_split2_mode.argtypes = [ContextPtr, ctypes.c_int32]
    if hasattr(lib, "llama_get_split2_mode"):
        lib.llama_get_split2_mode.restype  = ctypes.c_int32
        lib.llama_get_split2_mode.argtypes = [ContextPtr]

    # Batch / decode
    lib.llama_batch_init.restype  = LlamaBatch
    lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.llama_batch_free.restype  = None
    lib.llama_batch_free.argtypes = [LlamaBatch]
    lib.llama_decode.restype  = ctypes.c_int32
    lib.llama_decode.argtypes = [ContextPtr, LlamaBatch]
    lib.llama_get_logits_ith.restype  = ctypes.POINTER(ctypes.c_float)
    lib.llama_get_logits_ith.argtypes = [ContextPtr, ctypes.c_int32]

    # Memory / KV
    lib.llama_get_memory.restype  = MemoryPtr
    lib.llama_get_memory.argtypes = [ContextPtr]
    lib.llama_memory_clear.restype  = None
    lib.llama_memory_clear.argtypes = [MemoryPtr, ctypes.c_bool]
    lib.llama_memory_seq_rm.restype  = ctypes.c_bool
    lib.llama_memory_seq_rm.argtypes = [MemoryPtr, ctypes.c_int32,
                                        ctypes.c_int32, ctypes.c_int32]

    # GPU-side KV layer info (used by gpu_quant.py)
    lib.llama_get_kv_layer_info.restype  = ctypes.c_int32
    lib.llama_get_kv_layer_info.argtypes = [ContextPtr,
                                             ctypes.POINTER(LlamaKVLayerInfo),
                                             ctypes.c_int32]

    # State seq (used by parse_state)
    lib.llama_state_seq_get_size.restype  = ctypes.c_size_t
    lib.llama_state_seq_get_size.argtypes = [ContextPtr, ctypes.c_int32]
    lib.llama_state_seq_get_data.restype  = ctypes.c_size_t
    lib.llama_state_seq_get_data.argtypes = [ContextPtr, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32]
    lib.llama_state_seq_set_data.restype  = ctypes.c_size_t
    lib.llama_state_seq_set_data.argtypes = [ContextPtr, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32]

    # GGML tensor helpers for cb_eval callbacks (symbols exported from libllama.so)
    lib.ggml_get_name.restype   = ctypes.c_char_p
    lib.ggml_get_name.argtypes  = [ctypes.c_void_p]
    lib.ggml_time_us.restype    = ctypes.c_int64
    lib.ggml_time_us.argtypes   = []
    lib.ggml_nbytes.restype     = ctypes.c_size_t
    lib.ggml_nbytes.argtypes    = [ctypes.c_void_p]
    lib.ggml_nelements.restype  = ctypes.c_int64
    lib.ggml_nelements.argtypes = [ctypes.c_void_p]
    lib.ggml_get_data.restype   = ctypes.c_void_p
    lib.ggml_get_data.argtypes  = [ctypes.c_void_p]

    # Tensor shape/data for eval callbacks (llama_tensor_ne / llama_tensor_data)
    lib.llama_tensor_ne.restype   = ctypes.c_int64
    lib.llama_tensor_ne.argtypes  = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_tensor_data.restype  = ctypes.c_void_p
    lib.llama_tensor_data.argtypes = [ctypes.c_void_p]


_LOG_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

# Eval callback type: bool (*)(struct ggml_tensor * t, bool ask, void * user_data)
# Set via cparams.cb_eval + cparams.cb_eval_user_data before llama_init_from_model.
# Called twice per graph node: ask=True (return True to request data sync to CPU),
# then ask=False (data is readable via llama_tensor_data / ggml_get_data).
EVAL_CB_TYPE = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p)


def load_lib(path=None):
    """Load libllama.so, call setup_lib, and return the CDLL instance."""
    if path is None:
        path = DEFAULT_LIB_PATH
    lib = ctypes.CDLL(path)
    setup_lib(lib)
    # Suppress state_read_meta spam (fired every decode step for per-token quants)
    # while keeping all other log messages including model loading and context setup.
    import sys
    def _log_cb(level, text, user_data):
        s = text.decode("utf-8", errors="replace")
        sl = s.lower()
        # Drop noisy DEBUG lines (GGML CUDA graph warmup, per-step KV meta, etc.)
        if ("state_read_meta" in s or "create_tensor" in s
                or "cuda graph" in sl or "ggml_backend_cuda_graph" in sl):
            return
        sys.stderr.write(s)
    lib._log_cb = _LOG_CB_TYPE(_log_cb)  # keep reference to prevent GC
    lib.llama_log_set(lib._log_cb, None)
    return lib


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over a 1-D array."""
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def detokenize(lib, vocab, tokens: list, remove_special: bool = True) -> str:
    """Convert a list of token ids back to a UTF-8 string."""
    n = len(tokens)
    arr = (ctypes.c_int32 * n)(*tokens)
    buf = ctypes.create_string_buffer(n * 8 + 256)
    length = lib.llama_detokenize(vocab, arr, n, buf, len(buf), remove_special, False)
    if length < 0:
        buf = ctypes.create_string_buffer(-length + 1)
        length = lib.llama_detokenize(vocab, arr, n, buf, len(buf), remove_special, False)
    return buf.raw[:max(length, 0)].decode("utf-8", errors="replace")


def token_to_piece(lib, vocab, token_id: int) -> str:
    """Convert a single token id to its string piece (e.g. ' hello', '▁world')."""
    buf = ctypes.create_string_buffer(64)
    length = lib.llama_token_to_piece(vocab, token_id, buf, len(buf), 0, False)
    if length < 0:
        buf = ctypes.create_string_buffer(-length + 1)
        length = lib.llama_token_to_piece(vocab, token_id, buf, len(buf), 0, False)
    return buf.raw[:max(length, 0)].decode("utf-8", errors="replace")


def tokenize(lib, vocab, text: str, buf_size: int = 400_000) -> list:
    """Tokenize text using llama_tokenize. Returns a list of token ids."""
    text_bytes = text.encode("utf-8")
    buf = (ctypes.c_int32 * buf_size)()
    n = lib.llama_tokenize(vocab, text_bytes, len(text_bytes), buf, buf_size, False, True)
    if n < 0:
        raise RuntimeError(f"tokenize failed, need buffer of {-n}")
    return list(buf[:n])


def _probe_special(lib, vocab, text: str) -> bool:
    """Return True if text tokenizes to exactly one token with parse_special=True."""
    text_bytes = text.encode("utf-8")
    buf = (ctypes.c_int32 * 4)()
    n = lib.llama_tokenize(vocab, text_bytes, len(text_bytes), buf, 4, False, True)
    return n == 1


# Known chat format signatures: (probe_token, prefix, suffix, stop, name)
_CHAT_FORMATS = [
    # Qwen / ChatML
    ("<|im_start|>",
     "<|im_start|>user\n",
     "<|im_end|>\n<|im_start|>assistant\n",
     "<|im_end|>",
     "qwen/chatml"),
    # Llama 4 (uses |header_start| / |header_end| without angle brackets)
    ("<|header_start|>",
     "<|header_start|>user<|header_end|>\n\n",
     "<|eot|>\n<|header_start|>assistant<|header_end|>\n\n",
     "<|eot|>",
     "llama4"),
    # Llama 3
    ("<|start_header_id|>",
     "<|start_header_id|>user<|end_header_id|>\n\n",
     "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
     "<|eot_id|>",
     "llama3"),
    # OpenAI GPT-OSS (gpt-oss-*): <|start|> / <|end|> / channel mechanism
    # Model emits <|channel|>analysis<|message|> (CoT) then <|channel|>final<|message|> (answer).
    # <|end|> is the inter-channel separator (closes analysis, precedes final) — NOT a stop signal.
    # <|return|> is the true EOG token, caught by llama_vocab_is_eog automatically.
    # No explicit stop string needed; pass None so run_sweep does not add one.
    # No system message injected here — model still generates sensible responses without it.
    ("<|start|>",
     "<|start|>user<|message|>",
     "<|end|>\n<|start|>assistant",
     None,
     "gpt-oss"),
]


def detect_chat_format(lib, vocab):
    """Probe the vocabulary to detect the model's chat format.

    Returns (prefix, suffix, stop_string, format_name) or (None, None, None, "unknown").
    Only fires when the probe token is a single special token in the vocabulary.
    """
    for probe, prefix, suffix, stop, name in _CHAT_FORMATS:
        if _probe_special(lib, vocab, probe):
            return prefix, suffix, stop, name
    return None, None, None, "unknown"
