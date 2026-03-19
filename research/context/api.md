# API Reference & Pitfalls

## Critical: llama_memory_clear
```python
# CORRECT
mem = lib.llama_get_memory(ctx)
lib.llama_memory_clear(mem, True)

# WRONG — silent memory corruption → segfault at next decode
lib.llama_memory_clear(ctx, True)
```

## Struct Layouts (verified on this build)
- `LlamaModelParams`: 72 bytes — 4 bytes padding between `main_gpu` and `tensor_split`
- `LlamaContextParams`: 136 bytes — 2 bytes padding between `kv_unified` and `samplers`
- `LlamaBatch`: 56 bytes — 4 bytes padding between `n_tokens` and `token` pointer

## Key ctypes Bindings
```python
lib.llama_get_memory.restype  = LlamaMemoryPtr
lib.llama_get_memory.argtypes = [LlamaContextPtr]
lib.llama_memory_clear.restype  = None
lib.llama_memory_clear.argtypes = [LlamaMemoryPtr, ctypes.c_bool]
lib.llama_state_seq_get_size.restype  = ctypes.c_size_t
lib.llama_state_seq_get_size.argtypes = [ContextPtr, ctypes.c_int32]
lib.llama_state_seq_get_data.restype  = ctypes.c_size_t
lib.llama_state_seq_set_data.restype  = ctypes.c_size_t
```

## KV State Blob Format
`llama_state_seq_get_data(ctx, buf, size, seq_id=0)` serializes as:
```
uint32  n_stream = 1
uint32  cell_count
[meta per cell: int32 pos + uint32 n_seq_id + int32 seq_id × n]
  (if n_pos_per_embd > 1: extra 16 bytes per cell — M-RoPE models like Qwen-VL)
uint32  v_trans
uint32  n_layer
[K blocks × n_layer: int32 ggml_type + uint64 row_size + raw bytes]
[V blocks × n_layer: same layout, or column-major if v_trans=True]
```
GGML types: 0=F32, 1=F16, 30=BF16. `parse_state.py` handles all three.

## Model Architecture (Qwen3-8B)
- n_layers=36, n_heads=32, n_kv_heads=8 (GQA), head_dim=128
- KV dim per layer = n_kv_heads × head_dim = 1024
- KV cache (FP16): n_ctx × 36 × 2048 bytes ≈ n_ctx × 72 KB
  - n_ctx=128  → ~9 MiB
  - n_ctx=4096 → ~288 MiB per K or V; ~576 MiB total
- n_ctx_train = 40960
