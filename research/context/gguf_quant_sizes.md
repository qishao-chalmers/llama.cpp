# GGUF Quantization Sizes — Qwen3-8B Analysis

## Key Insight: "Q3_K_M" is NOT a pure 3-bit model

From `src/llama-quant.cpp`, Q3_K_M's "M" (medium) blend heavily upgrades
sensitive tensors away from Q3_K:

| Tensor | Q3_K_M | Q4_K_M |
|--------|--------|--------|
| ffn_gate, ffn_up | Q3_K (3.44 bpw) | Q4_K (4.50 bpw) |
| ffn_down | Q4_K (layers ≥2) / Q5_K (first 2) | Q4_K or Q6_K (50% each via `use_more_bits`) |
| attn_q, attn_k | Q3_K | Q4_K |
| attn_v | Q4_K (most) / Q5_K (first 2 layers) | Q6_K (50%) / Q4_K (50%) |
| attn_output | Q4_K | Q4_K |
| token_embd | Q3_K (untied) | Q4_K (untied) |
| output.weight | Q6_K | Q6_K |

The `use_more_bits(i_layer, n_layer)` lambda (line 412):
```cpp
return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
```
For n_layer=36: returns True for layers {0-3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31-35} = 18/36 = 50%.

## Effective bits-per-weight (from README, Llama-3-8B reference)

| Type | bpw | Size (8B model) |
|------|-----|-----------------|
| Q2_K | 3.1593 | ~3.1G |
| Q3_K_S | 3.6429 | ~3.4G |
| **Q3_K_M** | **3.9960** | **~3.7G** |
| Q4_K_S | 4.6672 | ~4.4G |
| **Q4_K_M** | **4.8944** | **~4.6G** |
| Q6_K | 6.5633 | ~6.1G |
| Q8_0 | 8.5008 | ~7.9G |

Q3_K_M at ~4.0 bpw is almost Q4_K territory — the "3" only refers to the
base type for ffn_gate/ffn_up, not the whole model.

## Qwen3-8B specific numbers (verified correct)

```
Q8_0:   8.2G  (source model, 8.50 bpw)
Q3_K_M: 3.9G  ✓ matches calculation (~3.8G expected)
Q4_K_M: 4.8G  ✓ matches calculation (~4.7G expected)
Q2_K:   3.1G  ✓ matches calculation (~3.2G expected)
```

**How to verify a GGUF is valid:**
```bash
python3 -c "import gguf; r=gguf.GGUFReader('models/Qwen3-8B-Q8_0-Q4_K_M.gguf'); [print(t.name, t.tensor_type) for t in r.tensors[:10]]"
# ValueError: GGUF magic invalid → file is corrupt
```

**Regenerate a corrupt draft model:**
```bash
python3 research/scripts/make_draft_model.py models/Qwen3-8B-Q8_0.gguf Q4_K_M --force
python3 research/scripts/make_draft_model.py models/Qwen3-8B-Q8_0.gguf Q2_K   --force
```

Note: early `make_draft_model.py` runs produced corrupt files (GGUF magic
invalid). Always verify size after generation — expected for Qwen3-8B:
- Q2_K ~3.1G, Q3_K_M ~3.9G, Q4_K_M ~4.8G

## Why vocab size matters for "fixed cost"

Qwen3-8B has vocab=151,936.  The `output.weight` tensor
(151936×4096 = 623M params) is stored at Q6_K for BOTH Q3_K_M and Q4_K_M
(by the OUTPUT branch in `llama_tensor_get_type`).  This contributes a fixed
~0.48 GiB regardless of which mix is chosen.

For comparison, Llama-3-8B has vocab≈32K so output.weight is ~5× smaller.
Large-vocab models therefore have a higher irreducible floor that shrinks the
relative size difference between quant types.

## Qwen3-8B architecture reference

```
n_layer = 36,  n_embd = 4096,  n_ff = 12288
n_head = 32,   n_head_kv = 8 (GQA),  n_embd_head = 128
vocab = 151936
tensors in Q8_0: 254 × q8_0 (weights) + 145 × f32 (norms incl. q_norm/k_norm per layer)
separate output.weight: yes (has_tied_embeddings = false)
```

Per-layer param counts:
- ffn_gate, ffn_up, ffn_down: 4096×12288 = 50.33M each
- attn_q, attn_o: 4096×4096 = 16.78M each
- attn_k, attn_v: 4096×1024 = 4.19M each (GQA: 8 heads × 128)
- Total per layer: 192.93M; 36 layers = 6,945M
- Embeddings (token_embd + output.weight): 2 × 623M = 1,246M
- Grand total: ~8.19B params
