# GPU-Side KV Quantization — Implementation Plan

## Problem
Current CPU-side quantization via `llama_state_seq_get_data/set_data` serializes the
entire KV state (all 36 layers, all tokens) through PCIe on every hook call.

- Per-channel (group_size=128): 16 hook calls for 2048 decode tokens → ~85s ✓
- Per-token (group_size=1):   2048 hook calls for 2048 decode tokens → ~3800s ✗

PCIe transfer dominates; numpy quantization itself is ~1ms. Fix: quantize in-place on GPU.

## Approach: Expose KV tensor pointers + CuPy

Add one small C function to llama.cpp that returns raw CUDA device pointers to K/V
tensors per layer. Python wraps them as CuPy arrays and quantizes in-place — zero PCIe.

**Requires Flash Attention enabled** (`--flash-attn`) so V is not transposed.

## Internal Data Structures (confirmed from source)

```
llama_kv_cache (src/llama-kv-cache.h):
  layers[]         — vector of kv_layer
  v_trans          — bool: true = V transposed (no FA), false = V normal (FA)
  v_cells[stream]  — llama_kv_cells, get_used() = number of tokens in cache
  seq_to_stream[]  — maps seq_id → stream index

kv_layer:
  k  — ggml_tensor*, shape [n_embd_k_gqa, kv_size, n_stream]
  v  — ggml_tensor*, shape [n_embd_v_gqa, kv_size, n_stream]

ggml_tensor:
  data   — void*: CUDA device pointer when GPU offloaded
  ne[0]  — n_embd_k_gqa (head_dim × n_heads_kv)
  ne[1]  — kv_size (max tokens)
  nb[1]  — bytes per token row (stride)
  nb[2]  — bytes per stream slice

Memory layout (v_trans=false, FA enabled):
  K[stream][token] = k->data + stream*nb[2] + token*nb[1]
  V[stream][token] = v->data + stream*nb[2] + token*nb[1]
  → clean [n_cells, n_embd] layout, directly usable as CuPy array
```

K/V written to cache in `cpy_k()` / `cpy_v()` at `src/llama-kv-cache.cpp:1075-1164`.

## Step 1: Add C API — `include/llama.h`

```c
typedef struct llama_kv_layer_info {
    void    * k_data;     // CUDA device ptr: start of K for stream 0
    void    * v_data;     // CUDA device ptr: start of V for stream 0
    int32_t   n_cells;    // tokens currently in cache
    int32_t   k_stride;   // bytes between token rows in K (nb[1])
    int32_t   v_stride;   // bytes between token rows in V (nb[1])
    int32_t   n_embd_k;   // K embedding dim (ne[0])
    int32_t   n_embd_v;   // V embedding dim (ne[0])
    int32_t   ggml_type;  // 1=F16, 30=BF16
    int32_t   v_trans;    // 1 if V transposed (FA disabled) — not supported
} llama_kv_layer_info;

LLAMA_API int32_t llama_get_kv_layer_info(
        struct llama_context * ctx,
        llama_kv_layer_info  * info,
        int32_t                n_layer);
```

## Step 2: Implement — `src/llama-kv-cache.cpp`

```cpp
int32_t llama_get_kv_layer_info(llama_context * ctx,
                                 llama_kv_layer_info * info,
                                 int32_t n_layer) {
    auto * kv = dynamic_cast<llama_kv_cache *>(ctx->memory.get());
    if (!kv) return -1;

    const uint32_t stream  = kv->seq_to_stream[0];
    const uint32_t n_cells = kv->v_cells[stream].get_used();
    const int32_t  n_fill  = std::min((int32_t)kv->layers.size(), n_layer);

    for (int32_t i = 0; i < n_fill; ++i) {
        ggml_tensor * k = kv->layers[i].k;
        ggml_tensor * v = kv->layers[i].v;

        info[i].k_data    = (char *)k->data + stream * k->nb[2];
        info[i].v_data    = (char *)v->data + stream * v->nb[2];
        info[i].n_cells   = (int32_t)n_cells;
        info[i].k_stride  = (int32_t)k->nb[1];
        info[i].v_stride  = (int32_t)v->nb[1];
        info[i].n_embd_k  = (int32_t)k->ne[0];
        info[i].n_embd_v  = (int32_t)v->ne[0];
        info[i].ggml_type = (int32_t)k->type;
        info[i].v_trans   = (int32_t)kv->v_trans;
    }
    return n_fill;
}
```

Also add `#include "llama-kv-cache.h"` and declaration in `src/llama-context.cpp` near
the other `llama_get_*` functions (around line 3119).

## Step 3: ctypes binding — `research/scripts/llama_bindings.py`

```python
class LlamaKVLayerInfo(ctypes.Structure):
    _fields_ = [
        ("k_data",    ctypes.c_void_p),
        ("v_data",    ctypes.c_void_p),
        ("n_cells",   ctypes.c_int32),
        ("k_stride",  ctypes.c_int32),
        ("v_stride",  ctypes.c_int32),
        ("n_embd_k",  ctypes.c_int32),
        ("n_embd_v",  ctypes.c_int32),
        ("ggml_type", ctypes.c_int32),
        ("v_trans",   ctypes.c_int32),
    ]

# in setup_lib():
lib.llama_get_kv_layer_info.restype  = ctypes.c_int32
lib.llama_get_kv_layer_info.argtypes = [ContextPtr,
                                         ctypes.POINTER(LlamaKVLayerInfo),
                                         ctypes.c_int32]
```

## Step 4: `research/scripts/gpu_quant.py` (new file)

```python
import cupy as cp
import llama_bindings as llama

def apply_kv_hook_gpu(lib, ctx, k_fn_name, v_fn_name,
                      n_layer, n_new_k=None, n_new_v=None):
    infos = (llama.LlamaKVLayerInfo * n_layer)()
    n_filled = lib.llama_get_kv_layer_info(ctx, infos, n_layer)
    assert n_filled > 0

    for info in infos[:n_filled]:
        assert info.v_trans == 0, "requires --flash-attn"
        n_cells = info.n_cells

        k_mem = cp.cuda.UnownedMemory(info.k_data, n_cells * info.k_stride, owner=None)
        v_mem = cp.cuda.UnownedMemory(info.v_data, n_cells * info.v_stride, owner=None)
        k = cp.ndarray((n_cells, info.n_embd_k), dtype=cp.float16,
                       memptr=cp.cuda.MemoryPointer(k_mem, 0))
        v = cp.ndarray((n_cells, info.n_embd_v), dtype=cp.float16,
                       memptr=cp.cuda.MemoryPointer(v_mem, 0))

        k_start = (n_cells - n_new_k) if n_new_k else 0
        v_start = (n_cells - n_new_v) if n_new_v else 0
        _quantize_inplace(k, k_start, k_fn_name)
        _quantize_inplace(v, v_start, v_fn_name)

def _quantize_inplace(arr, start, fn_name):
    chunk = arr[start:]
    bits  = int(''.join(filter(str.isdigit, fn_name.split('_')[0])))
    n_lev = (1 << (bits - 1)) - 1
    if '_ch' in fn_name:
        amax = cp.abs(chunk).max(axis=0, keepdims=True)   # per-channel
    elif '_tok' in fn_name:
        amax = cp.abs(chunk).max(axis=1, keepdims=True)   # per-token
    else:
        amax = cp.abs(chunk).max()                         # per-tensor
    scale = cp.where(amax == 0, 1.0, amax / n_lev)
    arr[start:] = (cp.round(chunk / scale).clip(-n_lev, n_lev) * scale).astype(cp.float16)
```

## Step 5: Wire into strategies.py

In `run_sweep.py`, detect GPU mode:
```python
use_gpu_quant = args.n_gpu_layers > 0 and GPU_QUANT_AVAILABLE
# replace parse_state hook with gpu_quant hook when use_gpu_quant=True
```

Hook signature stays the same (`hook(ctx, n_new_k=None, n_new_v=None)`) — just swap
the implementation.

## Files to Change

| File | Change |
|------|--------|
| `include/llama.h` | Add struct + declaration |
| `src/llama-kv-cache.cpp` | Implement function |
| `src/llama-context.cpp` | Forward declaration if needed (near line 3119) |
| `research/scripts/llama_bindings.py` | Add LlamaKVLayerInfo + ctypes binding |
| `research/scripts/gpu_quant.py` | New file |
| `research/scripts/run_sweep.py` | Auto-select GPU/CPU path |

## Build After Changes
```bash
# On MN5 compute node:
cmake -B build_release -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=OFF
cmake --build build_release --config Release -j $(nproc)
# libllama.so at build_release/bin/libllama.so
```

## Expected Speedup
- Current per-token: ~2048 PCIe round-trips × ~0.4s = ~800s per quant
- GPU in-place: ~2048 CuPy kernel launches × ~0.001s = ~2s per quant
- ~400× speedup for per-token quants
