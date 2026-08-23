"""
gpu_quant.py — GPU-side KV cache quantization via CuPy.

Replaces the CPU parse_state path for GPU inference.  Instead of serializing
the full KV state through PCIe on every hook call, this module wraps raw CUDA
device pointers exposed by llama_get_kv_layer_info() as CuPy arrays and
quantizes in-place — zero PCIe transfers.

Requirements:
  - cupy installed (conda install cupy / pip install cupy-cuda12x)
  - Model loaded with n_gpu_layers > 0 and --flash-attn
    (Flash Attention keeps V in un-transposed layout, required for clean indexing)

Speedup vs CPU parse_state:
  per-channel (group=128):  ~16 CuPy kernel launches vs ~16 PCIe round-trips  → similar
  per-token   (group=1):  ~2048 CuPy kernel launches vs ~2048 PCIe round-trips → ~400x faster

Usage (called from run_sweep.py when GPU mode is active):
  hook = make_kv_hook_gpu(lib, ctx, k_name, v_name, n_layer)
  hook(ctx, n_new_k=1, n_new_v=1)   # same interface as parse_state hook
"""

import ctypes
import time

import llama_bindings as llama

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _quantize_inplace_gpu(arr, fn_name, dim_group_size=None):
    """
    Quantize a 2-D CuPy array [n_tokens, n_embd] in-place using uniform symmetric quant.

    fn_name examples: "fp16", "int4_ch", "int4_tok", "int3_ch:int3_tok" (caller splits K/V first).
    dim_group_size: if set, group adjacent dims into groups of this size (sub-token grouping).
    """
    if fn_name == "fp16":
        return   # no-op

    import re

    # ── int3_half_ABCD_ch/tok: 4-level non-uniform quantizer ──────────────
    m_half = re.match(r"int3_half_(\d{4})(?:_(ch|tok))?$", fn_name)
    if m_half:
        digits  = m_half.group(1)
        suffix  = m_half.group(2)   # "ch", "tok", or None
        indices = [int(d) for d in digits]
        levels  = cp.array([i / 7.0 for i in indices], dtype=cp.float32)  # [4]
        f32     = arr.astype(cp.float32)

        if suffix == "ch":
            xmin = f32.min(axis=0)                              # [n_embd]
            xmax = f32.max(axis=0)
            rng  = cp.where(xmax - xmin == 0, cp.float32(1.0), xmax - xmin)
            cb   = xmin[:, None] + levels[None, :] * rng[:, None]  # [n_embd, 4]
            diff = cp.abs(f32[:, :, None] - cb[None, :, :])        # [n_tok, n_embd, 4]
            idx  = diff.argmin(axis=-1)                             # [n_tok, n_embd]
            arr[:] = cb[cp.arange(f32.shape[1])[None, :], idx].astype(cp.float16)
        elif suffix == "tok":
            xmin = f32.min(axis=1)                              # [n_tok]
            xmax = f32.max(axis=1)
            rng  = cp.where(xmax - xmin == 0, cp.float32(1.0), xmax - xmin)
            cb   = xmin[:, None] + levels[None, :] * rng[:, None]  # [n_tok, 4]
            diff = cp.abs(f32[:, :, None] - cb[:, None, :])        # [n_tok, n_embd, 4]
            idx  = diff.argmin(axis=-1)                             # [n_tok, n_embd]
            arr[:] = cb[cp.arange(f32.shape[0])[:, None], idx].astype(cp.float16)
        else:
            xmin  = float(f32.min())
            xmax  = float(f32.max())
            rng   = xmax - xmin if (xmax - xmin) != 0 else 1.0
            cb    = cp.array(xmin, dtype=cp.float32) + levels * cp.float32(rng)
            diff  = cp.abs(f32[..., None] - cb)
            idx   = diff.argmin(axis=-1)
            arr[:] = cb[idx].astype(cp.float16)
        return

    # Parse bits and axis from name: intN_ch → axis=0, intN_tok → axis=1, intN → axis=None
    m = re.match(r"int(\d+)(?:_(ch|tok))?(?:_g\d+)?$", fn_name)
    if m is None:
        raise ValueError(f"gpu_quant: unsupported quant name '{fn_name}'")

    bits = int(m.group(1))
    suffix = m.group(2)   # "ch", "tok", or None

    # Parse dim_group_size from name suffix (e.g. "int4_tok_g32")
    gm = re.search(r"_g(\d+)$", fn_name)
    if gm is not None:
        dim_group_size = int(gm.group(1))

    n_levels = (1 << (bits - 1)) - 1   # int8→127, int4→7, int3→3, int2→1
    f32 = arr.astype(cp.float32)

    if dim_group_size is not None and suffix == "tok":
        # Sub-token group quantization: split n_embd into groups of dim_group_size
        n_tokens, n_embd = f32.shape
        assert n_embd % dim_group_size == 0, \
            f"n_embd {n_embd} not divisible by dim_group_size {dim_group_size}"
        n_groups = n_embd // dim_group_size
        f32_g = f32.reshape(n_tokens, n_groups, dim_group_size)
        amax = cp.abs(f32_g).max(axis=2, keepdims=True)
        amax = cp.where(amax == 0, cp.float32(1.0), amax)
        scale = amax / n_levels
        q = cp.round(f32_g / scale).clip(-n_levels, n_levels)
        arr[:] = (q * scale).reshape(n_tokens, n_embd).astype(cp.float16)
    elif suffix == "ch":
        amax = cp.abs(f32).max(axis=0, keepdims=True)   # [1, n_embd] — per-dim
        amax = cp.where(amax == 0, cp.float32(1.0), amax)
        scale = amax / n_levels
        q = cp.round(f32 / scale).clip(-n_levels, n_levels)
        arr[:] = (q * scale).astype(cp.float16)
    elif suffix == "tok":
        amax = cp.abs(f32).max(axis=1, keepdims=True)   # [n_tokens, 1] — per-token
        amax = cp.where(amax == 0, cp.float32(1.0), amax)
        scale = amax / n_levels
        q = cp.round(f32 / scale).clip(-n_levels, n_levels)
        arr[:] = (q * scale).astype(cp.float16)
    else:
        # per-tensor
        amax = cp.abs(f32).max()
        amax = cp.float32(1.0) if float(amax) == 0 else amax
        scale = amax / n_levels
        q = cp.round(f32 / scale).clip(-n_levels, n_levels)
        arr[:] = (q * scale).astype(cp.float16)


def gpu_kv_cache_supported(lib, ctx, n_layer: int) -> bool:
    """Return True if llama_get_kv_layer_info works for this context.

    The C API only supports the classic ``llama_kv_cache`` backend.  Models that
    use hybrid memory (e.g. Qwen3.5) or ``llama_kv_cache_iswa`` expose a
    different ``llama_memory_i`` implementation, so the cast fails and the
    function returns -1.  Callers must fall back to CPU ``parse_state`` hooks.
    """
    if not HAS_CUPY or n_layer <= 0:
        return False
    infos = (llama.LlamaKVLayerInfo * n_layer)()
    n_filled = lib.llama_get_kv_layer_info(ctx, infos, n_layer)
    return n_filled > 0


def apply_kv_hook_gpu(lib, ctx, n_layer, k_fn_names, v_fn_names,
                      n_new_k=None, n_new_v=None, profile=None):
    """
    Quantize the last n_new_k K cells and n_new_v V cells in-place on the GPU.

    lib         — loaded libllama.so (with llama_get_kv_layer_info bound)
    ctx         — ContextPtr
    n_layer     — number of KV cache layers (model n_layer)
    k_fn_names  — quant name for K: single str (all layers) or list[str] (one per layer)
    v_fn_names  — quant name for V: single str (all layers) or list[str] (one per layer)
    n_new_k     — quantize last N K cells (None = all)
    n_new_v     — quantize last N V cells (None = all)
    profile     — optional kv_profile.KvProfile; accumulates gpu_kv_s per call (after sync).
    """
    if not HAS_CUPY:
        raise RuntimeError("gpu_quant requires CuPy: pip install cupy-cuda12x")

    t0 = time.perf_counter() if profile is not None else None

    infos = (llama.LlamaKVLayerInfo * n_layer)()
    n_filled = lib.llama_get_kv_layer_info(ctx, infos, n_layer)
    if n_filled <= 0:
        raise RuntimeError(f"llama_get_kv_layer_info returned {n_filled} — not a KV cache context?")

    # Normalize to lists
    if isinstance(k_fn_names, str):
        k_fn_names = [k_fn_names] * n_filled
    if isinstance(v_fn_names, str):
        v_fn_names = [v_fn_names] * n_filled

    for i in range(n_filled):
        info = infos[i]

        if info.v_trans:
            raise RuntimeError(
                "GPU quantization requires Flash Attention (--flash-attn). "
                "V tensor is transposed (v_trans=1), layout unsupported.")

        n_cells   = info.n_cells
        n_embd_k  = info.n_embd_k
        n_embd_v  = info.n_embd_v
        k_stride  = info.k_stride
        v_stride  = info.v_stride

        k_name  = k_fn_names[i] if i < len(k_fn_names) else "fp16"
        v_name  = v_fn_names[i] if i < len(v_fn_names) else "fp16"
        k_new_i = n_new_k[i] if isinstance(n_new_k, list) else n_new_k
        v_new_i = n_new_v[i] if isinstance(n_new_v, list) else n_new_v

        # Wrap raw CUDA device pointers as CuPy arrays — zero PCIe transfer
        k_mem = cp.cuda.UnownedMemory(info.k_data, n_cells * k_stride, owner=None)
        v_mem = cp.cuda.UnownedMemory(info.v_data, n_cells * v_stride, owner=None)
        k_arr = cp.ndarray((n_cells, n_embd_k), dtype=cp.float16,
                           memptr=cp.cuda.MemoryPointer(k_mem, 0))
        v_arr = cp.ndarray((n_cells, n_embd_v), dtype=cp.float16,
                           memptr=cp.cuda.MemoryPointer(v_mem, 0))

        if k_name != "fp16" and k_new_i != 0:
            k_start = (n_cells - k_new_i) if k_new_i is not None else 0
            _quantize_inplace_gpu(k_arr[k_start:], k_name)

        if v_name != "fp16" and v_new_i != 0:
            v_start = (n_cells - v_new_i) if v_new_i is not None else 0
            _quantize_inplace_gpu(v_arr[v_start:], v_name)

    cp.cuda.Device().synchronize()

    if profile is not None:
        profile.gpu_kv_s += time.perf_counter() - t0
        profile.n_gpu_kv_hook_calls += 1


def make_kv_hook_gpu(lib, ctx, k_fn_names, v_fn_names, n_layer, profile=None):
    """
    Return a hook(ctx, n_new_k=None, n_new_v=None) callable that quantizes
    KV in-place on the GPU.  Same interface as parse_state.apply_kv_hook.

    k_fn_names / v_fn_names: single str (all layers) or list[str] (one per layer).
    """
    if not HAS_CUPY:
        raise RuntimeError("gpu_quant requires CuPy: pip install cupy-cuda12x")

    def hook(ctx, n_new_k=None, n_new_v=None,
             _lib=lib, _k=k_fn_names, _v=v_fn_names, _nl=n_layer, _prof=profile, **_):
        apply_kv_hook_gpu(_lib, ctx, _nl, _k, _v,
                          n_new_k=n_new_k, n_new_v=n_new_v, profile=_prof)

    return hook
