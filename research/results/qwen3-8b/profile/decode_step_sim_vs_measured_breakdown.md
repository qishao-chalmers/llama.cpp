# Decode step: layerwise sim vs measured (Qwen3-8B, KV f16)

**Preset:** `qwen3-8b-q8_0` (weight_bits=8.5) · **KV:** `f16` · **ctx (mid):** 4096
**Hardware:** `h100-sxm` · **η:** `layerwise_eta_h100.json` (all-kv fit) · **attn:** `simple`

**Real step ms** = `measured_ms × B` from `kv_timing_h100.json` (`measured_ms` is decode ms/tok, i.e. wall-clock-per-step / B).

**Sim step ms** = sum of `SimEvent.seconds` × 1000 from `simulate_decode_step` (all layers, serial roofline sum).

---

## Table (corrected — Q8_0 weight preset)

```
┌────┬─────────────┬──────────────┬────────┬──────────────────────────────────────────────────────────────────┐
│ B  │ sim step ms │ real step ms │  ratio │  q_proj  k_proj  v_proj  attn_core  o_proj     FFN   other       │
├────┼─────────────┼──────────────┼────────┼──────────────────────────────────────────────────────────────────┤
│ 1  │       8.562 │        7.599 │  1.127 │   0.722   0.181   0.181      0.911   0.722   5.820   0.025       │
│ 4  │      11.373 │       11.133 │  1.022 │   0.724   0.182   0.182      3.644   0.724   5.831   0.086       │
│ 8  │      15.122 │       14.292 │  1.058 │   0.727   0.184   0.184      7.289   0.727   5.846   0.167       │
│ 16 │      22.619 │       18.295 │  1.236 │   0.732   0.187   0.187     14.578   0.732   5.875   0.328       │
│ 32 │      37.613 │       26.121 │  1.440 │   0.743   0.194   0.194     29.155   0.743   5.934   0.652       │
└────┴─────────────┴──────────────┴────────┴──────────────────────────────────────────────────────────────────┘
```

_(Previous table used `qwen3-8b` preset with weight_bits=16; that caused the 2× overestimate.)_

---

## Root cause analysis

### Bug 1 (fixed): wrong weight_bits

The original script used `cat["presets"]["qwen3-8b"]` which has `weight_bits=16` (fp16),
but `kv_timing_h100.json` was measured with Q8_0 weights (8.5 bpw = 1.0625 bpe).
The sim computed weight bandwidth at 2.0 bpe — 1.88× too high for every GEMM.

Fix: use `qwen3-8b-q8_0` preset (or pass `weight_tag="Q8_0"` explicitly).

Before fix: sim/real ≈ 2.0× across all B.
After fix: sim/real = 1.03–1.44× (acceptable at B≤8; grows at large B).

### Remaining gap: attn_core at large B

Decomposing each step into weight ops vs attention:

| B  | sim wt ms | sim attn ms | real step ms | implied real attn ms | attn overestimate |
|----|----------:|------------:|-------------:|---------------------:|------------------:|
|  1 |     7.651 |       0.911 |        7.599 |               −0.052 |               n/a |
|  4 |     7.729 |       3.644 |       11.133 |                3.404 |             1.07× |
|  8 |     7.832 |       7.289 |       14.292 |                6.459 |             1.13× |
| 16 |     8.040 |      14.578 |       18.295 |               10.255 |             1.42× |
| 32 |     8.455 |      29.155 |       26.121 |               17.666 |             1.65× |

Key insight: at B=1, the weight ops (7.651ms sim) nearly exactly match reality (7.599ms real). The weight GEMV is well-calibrated. All residual error is in `attn_core`.

The `attn_core` in the serial roofline model grows as B × ctx × kv_bytes / BW. In practice, the GPU overlaps KV streaming with GEMM compute in the next layer, so the serial sum overestimates by 1.65× at B=32.

A 3-component fit `real_step ≈ α/B + β_wt × wt_ms + β_attn × attn_ms` gives:

```
β_wt   = 1.365  (weight scale — inflated because η was trained on all kv types)
β_attn = 0.507  (attn scale — real attn ≈ 51% of serial BW model)
RMSE   = 0.252 ms (vs 0.511 ms from η-only prediction)
```

The `β_wt > 1` happens because the all-kv η fit pulled `gemm_bw` down to 0.379 (vs 0.422 in the f16-only refit) to compensate for quantized KV overhead — see below.

### Quantized KV: model form error

For q8_0 and q4_0 KV, the BW-proportional model is fundamentally wrong:

| kv_type | sim change vs f16 (B=1) | measured change vs f16 (B=1) |
|---------|------------------------:|-----------------------------:|
| f16     |                  0.000  |                       0.000  |
| q8_0    |               −0.441 ms |                      +0.632 ms |
| q4_0    |               −0.669 ms |                      +0.613 ms |

The sim predicts quantized KV is *faster* (fewer bytes → less BW → shorter time).
Reality: q8_0/q4_0 KV is *slower* than f16 by ~0.6ms/tok at B=1 (and increasingly slower at larger B).

Cause: the CUDA dequantization kernel adds fixed compute overhead that dominates the BW savings.
This is a model-form error; no η value can fix it for both f16 and quantized KV simultaneously.

The all-kv η fit (used in the current JSON) attempted a compromise:
- Pulled `attn_bw` to 0.283 (vs 0.320 in f16-only fit) to inflate quantized KV predictions
- This overshot f16 attn, creating the growing gap at large B

---

## Commands (run from `research/scripts`)

```bash
python3 << 'PY'
import json
from collections import defaultdict
from pathlib import Path

from layerwise_roofline_sim import (
    HARDWARE_PRESETS,
    load_structure_catalog,
    load_eta_json,
    resolve_kv_quant_key,
    resolve_weight_bits,
    simulate_decode_step,
)

cat = load_structure_catalog("model_structures.json")
# IMPORTANT: use qwen3-8b-q8_0 (weight_bits=8.5), NOT qwen3-8b (weight_bits=16)
model = dict(cat["presets"]["qwen3-8b-q8_0"])
wb, _ = resolve_weight_bits(model, None, None)
wbpe = wb / 8.0
norm_bpe = 16.0 / 8.0
hw = dict(HARDWARE_PRESETS["h100-sxm"])
ctx = 2048 + 4096 // 2
kv_key = resolve_kv_quant_key("f16")
eta = load_eta_json("../results/qwen3-8b/profile/layerwise_eta_h100.json")

kv_path = Path("../results/qwen3-8b/profile/kv_timing_h100.json")
meas_rows = {
    int(r["batch_size"]): float(r["measured_ms"])
    for r in json.loads(kv_path.read_text())["rows"]
    if r.get("model_preset") == "qwen3-8b" and str(r.get("kv_type")).lower() == "f16"
}

def bucket(name):
    if name in ("q_proj", "k_proj", "v_proj", "o_proj"): return name
    if name == "attn_core" or name.startswith("flash_"): return "attn_core"
    if name in ("ffn_gate", "ffn_up", "ffn_down"): return "ffn"
    return "other"

for B in (1, 4, 8, 16, 32):
    _, events = simulate_decode_step(model, B, ctx, hw, eta,
        weight_bpe=wbpe, norm_bpe=norm_bpe, kv_quant_key=kv_key, attn_impl="simple")
    by_b = {k: 0.0 for k in ("q_proj","k_proj","v_proj","attn_core","o_proj","ffn","other")}
    for e in events: by_b[bucket(e.name)] += e.seconds
    ms = sum(e.seconds for e in events) * 1000.0
    real_step = meas_rows[B] * B
    print(B, f"sim={ms:.3f}", f"real={real_step:.3f}", f"ratio={ms/real_step:.3f}",
          f"q={by_b['q_proj']*1000:.3f}", f"attn={by_b['attn_core']*1000:.3f}",
          f"ffn={by_b['ffn']*1000:.3f}")
PY
```
