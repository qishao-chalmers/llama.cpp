# Selective Weight Precision: N-bit-in-Q8 GGUF Builder

## 1. Goal

Produce a standard Q8_0 GGUF from an input Q8_0 GGUF where specific tensor groups
are re-quantised to lower effective precision (2–8 bits) while remaining stored in
Q8_0 format.  Every tensor type is still `GGML_TYPE_Q8_0`; no new kernels or special
handling is required.  The output loads in any framework that supports Q8_0.

**Primary use case:** ablation / weight-importance research — measure which tensor
groups (attention projections, FFN gate/up, FFN down/V) actually need high precision,
and by how much, without changing the inference stack.

No memory saving, no speed improvement vs Q8_0.  The value is flexibility: arbitrary
per-group, per-layer precision in a single GGUF file.


## 2. Script

`research/scripts/build_selective_precision.py`

Input:  `models/Qwen3-8B-Q8_0.gguf`  (or any Q8_0 GGUF)
Output: `models/Qwen3-8B-Q8_0-attn4-ffn3-ess6.gguf`  (Q8_0 GGUF, same size)


## 3. Tensor Classification

Based on actual Qwen3-8B tensor names.  The same patterns apply to other llama.cpp models.

### 3a. Tensor groups

| Group name | Tensor name patterns | Default bits | Flag |
|-----------|---------------------|-------------|------|
| `attn` | `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_output.weight` | 8 | `--attn-bits` |
| `ffn` | `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight` | 8 | `--ffn-bits` |
| `essential` | `blk.N.attn_v.weight`, `blk.N.ffn_down.weight` | 8 | `--essential-bits` |
| `skip` | everything else | always 8, copied as-is | — |

The `skip` group includes: `token_embd.weight`, `output.weight`, `output_norm.weight`,
`blk.N.attn_norm.weight`, `blk.N.ffn_norm.weight`, `blk.N.attn_k_norm.weight`,
`blk.N.attn_q_norm.weight`, and any tensor that is not Q8_0 (F32/F16 kept as-is).

### 3b. Classification logic

```python
import re

def classify_tensor(name: str) -> str:
    """Return 'attn' | 'ffn' | 'essential' | 'skip'."""
    m = re.match(r"^blk\.(\d+)\.(.*?)\.weight$", name)
    if m is None:
        return "skip"                      # token_embd, output, norms without blk prefix
    role = m.group(2)
    if role in ("attn_q", "attn_k", "attn_output"):
        return "attn"
    if role in ("ffn_gate", "ffn_up"):
        return "ffn"
    if role in ("attn_v", "ffn_down"):
        return "essential"
    return "skip"                          # attn_norm, ffn_norm, attn_k_norm, attn_q_norm, etc.

def layer_index(name: str) -> int | None:
    """Return layer index from tensor name, or None for non-blk tensors."""
    m = re.match(r"^blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None
```


## 4. Layer Range: First / Last N Layers Protected

Any tensor in the first `--first-layers` layers or the last `--last-layers` layers is
treated as `skip` regardless of its group, keeping full Q8_0 precision.

```
layers:  [0, 1, ..., first-1]  → always skip (Q8_0)
layers:  [first, ..., n_layers-1-last]  → quantise according to group bits
layers:  [n_layers-last, ..., n_layers-1]  → always skip (Q8_0)
```

Total layer count is determined by scanning all tensor names for the maximum `blk.N.*`
index before processing begins.

```python
def compute_n_layers(tensors) -> int:
    indices = [layer_index(t.name) for t in tensors]
    return max(i for i in indices if i is not None) + 1

def should_skip_layer(layer: int, n_layers: int, first: int, last: int) -> bool:
    return layer < first or layer >= (n_layers - last)
```

Default: `--first-layers 0 --last-layers 0` (no protection, all layers quantised).
Recommended for Qwen3-8B (36 layers): `--first-layers 2 --last-layers 2`.


## 5. Quantisation Algorithm

### 5a. Asymmetric N-bit within a Q8_0 block

Reuse `_uniform_quant_asym` from `research/scripts/quant.py` (already imports cleanly).
Apply it per 32-element Q8_0 block — fully vectorised, no Python loop over blocks.

```python
import sys
sys.path.insert(0, str(repo_root / "research/scripts"))
from quant import _uniform_quant_asym   # affine: scale=(xmax-xmin)/n_levels, q=round((x-xmin)/scale)
```

`_uniform_quant_asym(arr, bits=N)` with `axis=None` and a shape of `(n_blocks, 32)`
can be called with `axis=1` to get independent min/max per block row:

```python
# (n_blocks, 32) float32 input → (n_blocks, 32) float16 output at N-bit precision
w_q = _uniform_quant_asym(w_blocks, bits=N, axis=1).astype(np.float32)
```

`axis=1` makes each row (= one 32-element block) get its own xmin/xmax — exactly one
asymmetric calibration per Q8_0 block.  `n_levels = 2^N - 1` (e.g. 4-bit → 15 levels).

### 5b. Full requantise_tensor function

```python
Q8_BLOCK_BYTES = 34   # 2 (fp16 scale) + 32 (int8)
Q8_BLOCK_ELEMS = 32

def requantise_q8_to_nbit(q8_flat: np.ndarray, bits: int) -> np.ndarray:
    """
    Re-quantise a Q8_0 tensor to N-bit asymmetric precision, re-encoded as Q8_0.

    q8_flat : uint8 (n_bytes,)  where n_bytes = n_blocks * 34
    bits    : 2–8 target precision (8 = no change, but still round-trips)
    Returns : uint8 (n_bytes,)  same shape, Q8_0 re-encoded
    """
    n_blocks = len(q8_flat) // Q8_BLOCK_BYTES
    blocks   = q8_flat.reshape(n_blocks, Q8_BLOCK_BYTES)

    # Dequantise Q8_0 → float32 (n_blocks, 32)
    d   = blocks[:, :2].view(np.float16).astype(np.float32)   # (n_blocks, 1)
    qs  = blocks[:, 2:].view(np.int8).astype(np.float32)      # (n_blocks, 32)
    w   = qs * d                                                # (n_blocks, 32)

    # Asymmetric N-bit quantisation per block (axis=1: per row)
    w_q = _uniform_quant_asym(w, bits=bits, axis=1).astype(np.float32)  # (n_blocks, 32)

    # Re-encode to Q8_0
    abs_max  = np.abs(w_q).max(axis=1, keepdims=True)          # (n_blocks, 1)
    d_new    = (abs_max / 127.0).astype(np.float16)             # new fp16 scale
    d_new_f32 = d_new.astype(np.float32)
    qs_new   = np.clip(
        np.round(w_q / np.where(d_new_f32 == 0.0, 1.0, d_new_f32)),
        -127, 127,
    ).astype(np.int8)                                           # (n_blocks, 32)

    # Pack: [fp16 scale | int8 × 32] per block
    out = np.zeros((n_blocks, Q8_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d_new.view(np.uint8)
    out[:, 2:] = qs_new.view(np.uint8)
    return out.reshape(-1)
```

For `bits=8` the round-trip is effectively a no-op (127 levels → re-encode to Q8_0 ≡
identity up to fp16 rounding of the scale).  Skipping `bits=8` groups saves time.

### 5c. Effective levels per bit width

| bits | n_levels | distinct float values per 32-elem block | analogous to |
|------|----------|----------------------------------------|-------------|
| 2 | 3 | 4 (0..3) | very coarse |
| 3 | 7 | 8 | Q3_0-quality |
| 4 | 15 | 16 | Q4_0-quality (same as Q4_K_M FFN) |
| 5 | 31 | 32 | Q5-quality |
| 6 | 63 | 64 | Q6-quality (similar to Q6_K essential) |
| 8 | 255 | 256 | full Q8_0 (no change) |


## 6. GGUF Read / Write Pattern

Follow `build_q8_recovered.py` exactly for metadata copy and tensor writing.

### 6a. Metadata copy (same skip list)

```python
_SKIP_KV = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
for kv in reader.fields.values():
    if kv.name in _SKIP_KV or kv.name.startswith("GGUF."):
        continue
    try:
        writer.add_key_value(kv.name, kv.contents(), kv.types[0])
    except Exception:
        pass
```

### 6b. Tensor writing decision tree

```python
for t in reader.tensors:
    ttype    = int(t.tensor_type)
    group    = classify_tensor(t.name)
    layer    = layer_index(t.name)
    target_bits = bits_for_group(group, args)   # see §7

    protected = (layer is None) or should_skip_layer(layer, n_layers, args.first_layers, args.last_layers)
    do_requantise = (not protected) and (group != "skip") and (ttype == GGML_TYPE_Q8_0) and (target_bits < 8)

    if do_requantise:
        q8_flat  = np.asarray(t.data, dtype=np.uint8).reshape(-1)
        new_flat = requantise_q8_to_nbit(q8_flat, bits=target_bits)
        # reshape to numpy byte shape expected by GGUFWriter
        numpy_elem_shape = tuple(reversed(list(t.shape)))
        numpy_byte_shape = quant_shape_to_byte_shape(numpy_elem_shape, GGMLQuantizationType.Q8_0)
        writer.add_tensor(t.name,
                          new_flat.reshape(numpy_byte_shape),
                          raw_dtype=GGMLQuantizationType.Q8_0)
    elif ttype == GGML_TYPE_F32:
        writer.add_tensor(t.name, t.data.view(np.float32))
    elif ttype == GGML_TYPE_F16:
        writer.add_tensor(t.name, t.data.view(np.float16))
    else:
        # Q8_0 copy-as-is, or any other quant type copy-as-is
        writer.add_tensor(t.name, t.data.view(np.uint8),
                          raw_dtype=GGMLQuantizationType(ttype))
```


## 7. CLI Interface

```
python3 research/scripts/build_selective_precision.py \
    --input   models/Qwen3-8B-Q8_0.gguf \
    --output  models/Qwen3-8B-Q8_0-attn4-ffn3-ess6.gguf \
    [--attn-bits      N]   # 2-8, default 8 (no change); group: q/k/o_proj
    [--ffn-bits       N]   # 2-8, default 8 (no change); group: ffn_gate/ffn_up
    [--essential-bits N]   # 2-8, default 8 (no change); group: attn_v/ffn_down
    [--first-layers   N]   # protect first N layers (default 0)
    [--last-layers    N]   # protect last  N layers (default 0)
    [--dry-run]            # print per-tensor actions without writing
    [--verbose]            # print per-tensor stats (bits, RMS error before/after)
```

```python
def bits_for_group(group: str, args) -> int:
    if group == "attn":      return args.attn_bits
    if group == "ffn":       return args.ffn_bits
    if group == "essential": return args.essential_bits
    return 8   # skip group always full precision
```

Auto-generated output filename from args when `--output` is omitted:
`{stem}-attn{A}-ffn{F}-ess{E}[-first{FL}][-last{LL}].gguf`


## 8. Dry-run Output Format

```
Tensor classification (36 layers total, first=2 last=2):
  blk.0.attn_q.weight        [attn]      → Q8_0 SKIP (protected layer 0)
  blk.0.attn_v.weight        [essential] → Q8_0 SKIP (protected layer 0)
  blk.2.attn_q.weight        [attn]      → Q4-in-Q8 (bits=4, 15 levels/block)
  blk.2.attn_k.weight        [attn]      → Q4-in-Q8 (bits=4)
  blk.2.attn_v.weight        [essential] → Q6-in-Q8 (bits=6, 63 levels/block)
  blk.2.ffn_gate.weight      [ffn]       → Q3-in-Q8 (bits=3)
  blk.2.ffn_down.weight      [essential] → Q6-in-Q8 (bits=6)
  blk.35.attn_q.weight       [attn]      → Q8_0 SKIP (protected layer 35)
  token_embd.weight          [skip]      → Q8_0 copy-as-is
  output.weight              [skip]      → Q8_0 copy-as-is
  ...
Summary: 252 tensors requantised (attn:108 ffn:72 essential:72), 144 skipped
```


## 9. Verbose Stats (--verbose)

For each requantised tensor, print:
- RMS error: `||w_q - w_orig||_rms` where w_orig = dequant(original Q8_0), w_q = dequant(re-encoded Q8_0)
- Saturation rate: fraction of elements that hit 0-level or max-level after N-bit quantisation

```
  blk.2.attn_q.weight  [attn  bits=4]  rms_err=3.21e-04  sat=0.00%
  blk.2.ffn_gate.weight [ffn  bits=3]  rms_err=8.45e-04  sat=0.01%
```


## 10. Example Sweep Commands

```bash
# Ablation: degrade attn only, keep everything else at Q8_0
python3 research/scripts/build_selective_precision.py \
    --input models/Qwen3-8B-Q8_0.gguf --attn-bits 4

# Match Q4_K_M quality distribution:
#   attn Q/K/O → Q4, FFN gate/up → Q4, V/down → Q6
python3 research/scripts/build_selective_precision.py \
    --input models/Qwen3-8B-Q8_0.gguf \
    --attn-bits 4 --ffn-bits 4 --essential-bits 6

# Protect first+last 4 layers, degrade middle to Q3
python3 research/scripts/build_selective_precision.py \
    --input models/Qwen3-8B-Q8_0.gguf \
    --attn-bits 3 --ffn-bits 3 --essential-bits 4 \
    --first-layers 4 --last-layers 4

# Extreme: only V and down survive at Q8_0; everything else Q2
python3 research/scripts/build_selective_precision.py \
    --input models/Qwen3-8B-Q8_0.gguf \
    --attn-bits 2 --ffn-bits 2 --essential-bits 8
```

Then run through `run_sweep.py` for PPL or GSM8K accuracy to measure the impact.


## 11. Key Dependencies

| Dependency | Source |
|-----------|--------|
| `_uniform_quant_asym` | `research/scripts/quant.py` |
| `GGUFReader`, `GGUFWriter` | `gguf-py/` |
| `quant_shape_to_byte_shape` | `gguf-py/gguf/quants.py` |
| GGUF write pattern | `research/scripts/build_q8_recovered.py` (copy metadata + tensor loop) |
| Tensor name patterns | verified from `Qwen3-8B-Q8_0.gguf` (§3a) |

Import path for `quant.py`:
```python
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "gguf-py"))
sys.path.insert(0, str(_repo_root / "research" / "scripts"))
from quant import _uniform_quant_asym
```


## 12. Non-Q8_0 Input Tensors

The input GGUF is expected to be pure Q8_0, but some tensors (token embedding,
output projection on some models) may be F32 or F16.  Those are always copied as-is —
no quantisation attempted.  If a non-Q8_0, non-F16, non-F32 tensor is encountered
(e.g. the input is accidentally a Q4_K_M GGUF), emit a warning and copy it unchanged.
