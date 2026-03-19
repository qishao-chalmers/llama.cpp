# KV Cache Quantization Research — Index

Perplexity impact of KV cache compression on Qwen3-8B / Qwen2.5-Coder-7B / Llama-4-Scout-17B.
Code: `research/scripts/` — Python ctypes wrapping `build_release/bin/libllama.so`.

## Sub-files (read on demand)
| File | Contents |
|------|----------|
| `context/api.md` | ctypes bindings, struct layouts, state blob format, critical API pitfalls |
| `context/scripts.md` | All scripts, CLI flags, usage examples |
| `context/strategies.md` | Quantization strategies and group-size semantics |
| `context/results.md` | All experiment results tables and key findings |
| `context/setup.md` | New machine setup (build, deps, model download) |
| `context/gpu_quantization.md` | GPU-side KV quantization implementation plan (next) |

## Critical Pitfall (always remember)
`llama_memory_clear` takes `llama_memory_t`, NOT `llama_context*`:
```python
mem = lib.llama_get_memory(ctx)    # required first step
lib.llama_memory_clear(mem, True)  # WRONG: passing ctx directly → silent corruption → segfault
```

## Current Status
- Script architecture refactored: llama_bindings.py / strategies.py / quant.py / parse_state.py / run_sweep.py
- Quant types: fp16/bf16/fp8/int8/int8_ch/int8_tok/int4/int4_ch/int4_tok/int3/int3_ch/int3_tok/int2/int2_ch/int2_tok/nf4
- K:V split notation supported: `int4_ch:int4_tok` = K per-channel, V per-token
- Separate k_group_size / v_group_size: per-token V quantized every token (correct for autoregressive)
- submit_sweep.sh: job array = models × prefill_tokens; n_ctx = prefill + max(score_windows) auto
- Corpus sampling: strided evenly across full corpus (not sequential from start)
- MN5 setup: conda env + libstdc++ LD_PRELOAD fix confirmed working
- GPU quantization bottleneck identified: PCIe round-trip per hook call (~200s for per-token at 2048 steps)
- Next: implement GPU-side quantization (see context/gpu_quantization.md)

## Key Results (prefill=1024, score_windows=512/1024/2048, Qwen3-8B, 3 chunks)
| Quant | ppl@512 | ppl@1024 | ppl@2048 | time |
|-------|---------|----------|----------|------|
| fp16 | 5.50 | 6.97 | 8.06 | 48s |
| int8_ch | 5.49 | 6.97 | 8.07 | 83s |
| int4_ch:int4_tok | 5.53 | 7.07 | 8.15 | 84s |
| int3_ch | 5.90 | 7.41 | 8.75 | 85s |
| int3_ch:int3_tok | 5.91 | 7.39 | 8.50 | 85s |
| int3_tok | 236 | 303 | 323 | 3816s ← slow: group_size=1 |
| int2_ch | 49.1 | 47.5 | 45.4 | 83s |
| int2_ch:int2_tok | 28.8 | 31.3 | 34.0 | 83s |
