# Layer-wise roofline simulation — design plan

**Status:** plan only — does **not** modify `research/scripts/perf_model.py` (aggregate roofline / shared presets stay as-is while other work proceeds).

**Goal:** a **separate** estimator that loads a **model structure** first, then runs an **event-ordered, layer-by-layer** decode-step simulation: each operation gets its own **mini-roofline** (compute vs memory) with explicit **batching rules**, optional **per-op-family efficiencies**, and outputs **total ms/tok** plus **per-layer / per-op breakdown**.

---

## 1. Scope of architectures

| Family | Preset examples (existing or to add) | Notes |
|--------|--------------------------------------|--------|
| **Qwen3** | `qwen3-8b`, `qwen3-14b`, future `qwen3-17b`-class | Same block pattern: RMSNorm, SwiGLU FFN, GQA attention. New sizes = **new preset row** in the structure catalog, not a code fork. |
| **Llama / Llama-like** | e.g. `llama4-scout-17b`, future Llama-3.x presets | Same **decoder block skeleton** (norm → attn → residual → norm → FFN → residual); dimensions differ. |
| **Later** | MoE, sliding-window, extra norms | **Out of MVP**; structure schema should allow optional fields later. |

There is **no** “Qwen3-only” hard-code in the simulator core: only **structure-driven** dimensions (`n_layers`, `d_model`, `n_heads`, `n_kv_heads`, `head_dim`, `ffn_dim`, `ffn_style`, …).

---

## 2. Load structure first

**Rule:** the script **always** resolves a **`ModelStructure`** (name below) before any timing math.

**Sources (pick one per run):**

1. **`--preset <name>`** — lookup in a **dedicated** structure catalog shipped with the new tool, e.g. `research/scripts/model_structures.json` or `layerwise_model_presets.py`.  
   - **Do not** require importing `perf_model.MODEL_PRESETS` if that would conflict with “don’t touch perf_model”; **duplicate** the numeric fields for listed presets or maintain one **read-only** JSON that both teams can align manually.

2. **`--structure-json path`** — user-supplied JSON with the same schema (for HuggingFace `config.json`-derived or GGUF metadata).

**Catalog file:** `research/scripts/model_structures.json` (see **Appendix A** for the full schema and field list).

**Validation (loader):** `d_model == n_heads * head_dim` unless explicitly documented exception; `n_kv_heads <= n_heads`; `ffn_style` in allowed set for MVP (`swiglu`).

---

## 3. Operation taxonomy (decoder block)

Order is **fixed** for MVP (one block, repeated `n_layers` times):

1. **Pre-attn RMSNorm** — elementwise (small).
2. **Q projection** — GEMM: `[B, 1, d_model] × W_q` → q dim `n_heads * head_dim` (decode: seq 1 per stream; batched over `B`).
3. **K projection** — GEMM to `n_kv_heads * head_dim` (GQA).
4. **V projection** — same as K.
5. **KV cache write** — **memory event**: bytes written depend on KV dtype × layer × heads (per new token).
6. **Attention core** — **not** a single GEMM: modeled as **KV read** (all layers’ cache read for this step’s context) + **FLOPs** for softmax/accum (simplified tile model in MVP; refinable later).
7. **O projection** — GEMM.
8. **Residual** — elementwise.
9. **Pre-FFN RMSNorm**.
10. **FFN** — SwiGLU: up, gate, down (three GEMMs or fused) with **batched** `B` on the token dimension.
11. **Residual**.

**Norms / activations:** lumped as **elementwise** roofline with high `η_comp`, low bytes unless we want separate `η`.

---

## 4. Batching semantics

| Op kind | How `B` enters FLOPs / bytes | Notes |
|---------|-------------------------------|--------|
| **Weight-aligned GEMM** (Q/K/V/O, FFN) | Linear in `B` for activations; **weights read once** per layer per op (shared across batch). | Simulator attributes **weight bytes** to the op that uses them; **amortization** = `weight_bytes` counted once per step, **not** × `B` for the **weight tensor** (activations scale with `B`). |
| **KV cache read (attention)** | Scales with **`B × ctx_len`** for loaded KV volume (per layer, per head group). | Core of decode scaling. |
| **KV cache write** | **`B`** new tokens per step. | |
| **Attention math** | Parallel over `B` and heads; **time** not `T(B=1)/B` naively — use explicit FLOP + byte model, not “÷ B”. | MVP: one roofline per “attn_core” event. |

The simulator should **tag** each event with `batch_scaling: "linear_B" | "shared_weights" | "attn_kv"` for debugging.

---

## 5. Per-op roofline (time for one event)

For each event `i`:

- `T_comp_i = flops_i / (P_comp * η_comp[family])`
- `T_mem_i  = bytes_i / (P_bw * η_bw[family])`
- `t_i = max(T_comp_i, T_mem_i)` (same spirit as global roofline, **local** to the op).

**Families** (suggested keys for η): `gemm`, `attn_core`, `elementwise`, `kv_rw`.

Hardware: same **peak TFLOPS / GB/s** table as today, **duplicated** in the new module or read from a tiny `hardware_presets.json` to avoid editing `perf_model.py`.

---

## 5b. sim_physics knobs (beyond η)

`η` is meant to represent **family-level hardware efficiency** (compute / BW) and is
ideally portable across nearby workloads. In practice we also need a small set of
“physics knobs” that capture *systematic* effects which are **not** explained by
pure bytes/FLOPs:

- **`kv_attn_byte_mode`**: how to count KV bytes for attention (e.g. storage vs fp16-equivalent).
- **`attn_time_scale`**, **`attn_time_scale_inv_batch`**, `attn_scale_by_batch`:
  simple correction factors for attention kernels (tile/scheduler effects).
- **`weight_time_scale_by_tag`**: map `weight_tag -> multiplier` that scales only the
  **GEMM family** time. This is intended to capture weight-quant kernel regime
  changes (dequant overhead, kernel selection) that a single “effective bpw” cannot.

### Design: `weight_time_scale_by_tag`

Within one simulated decode step, split time into:

\[
T = T_\text{gemm} + T_\text{other}
\]

and apply:

\[
T' = s(\text{weight\_tag}) \cdot T_\text{gemm} + T_\text{other}
\]

where `s(weight_tag)` is read from `sim_physics["weight_time_scale_by_tag"]`.

Key intent:
- Keep `η` stable (family-level), and let `s(weight_tag)` capture quant-kernel
  overhead differences between e.g. `Q8_0` vs `Q4_K_M`.
- This is per-profile/per-model by default (local fit), but can be merged into a
  shared prior later if it generalizes.

### Fitting workflow (implemented)

- `research/scripts/fit_sim_physics_weight_scale.py` fits `weight_time_scale_by_tag`
  from one `kv_timing*.json` plus its `layerwise_eta_*.json`.
- It runs the simulator once per measured row, extracts `gemm_ms` and `other_ms`,
  and solves the least-squares fit:

\[
\min_s \sum_i (s \cdot \text{gemm}_i + \text{other}_i - \text{meas}_i)^2
\]

with clamps (default \(s \in [0.5, 2.0]\)).

This is designed to be a “small knob” fit: if the tag’s behavior is already
explained by η, the fitted scale lands near 1.0.

---

## 6. Event-driven simulation loop

**One decode step:**

```
total_time = 0
for layer in 1 .. n_layers:
    for op in ordered_ops(layer):
        t_i = roofline_op(op, B, ctx_len, structure, hw, eta)
        emit event(layer, op, t_i)
        total_time += t_i   # v1: strictly sequential
ms_per_tok = total_time / B
```

**Outputs:**

- `total_ms`, `ms_per_tok`, `tok_per_s`
- Optional: `breakdown_ms` by layer and by op family
- Optional: export JSON for comparison with `benchmark_kv_timing` / aggregate roofline

**v2 (later):** overlap rules between events — **not** in MVP.

---

## 7. CLI (sketch)

```text
python3 layerwise_roofline_sim.py \
  --preset qwen3-14b \
  --batch-size 8 \
  --ctx-len 4096 \
  --hw h100-sxm \
  --eta-bw-gemm 0.45 --eta-comp-gemm 0.4 \
  --structure-json path/to/custom.json   # optional override
```

---

## 8. Phasing

| Phase | Content |
|-------|---------|
| **MVP** | Structure loader + Qwen3 + Llama-style presets from catalog; sequential sum; SwiGLU + GQA; simplified **attn_core** (KV bytes + simple FLOP estimate). |
| **v2** | Richer attention (tile/block model closer to flash-attn). |
| **v3** | Prefill (`seq > 1`) path; optional per-layer overlap. |
| **v4** | Calibrate η from measured JSON; compare to aggregate `roofline_ms` without replacing it. |

---

## 9. File layout (implementation phase)

| Path | Role |
|------|------|
| `research/scripts/layerwise_roofline_sim.py` | CLI + simulation loop |
| `research/scripts/model_structures.json` | Presets: `schema_version`, `presets` map (qwen3-8b, qwen3-14b, llama4-scout-17b, …) |
| `research/context/layerwise_roofline_sim_plan.md` | This document |
| `research/scripts/fit_layerwise_eta.py` | Fit per-op-family η from measured kv_timing JSON |
| `research/scripts/fit_layerwise_calibration.py` | Fit linear `(t_floor_ms, scale)` calibration on top of layerwise prediction |
| `research/scripts/fit_sim_physics_weight_scale.py` | Fit `weight_time_scale_by_tag` into a sim_physics JSON |
| `research/scripts/report_measured_vs_layerwise.py` | Table/report measured vs predicted across many kv_timing JSONs; supports auto-discovery of per-profile files |

**Constraint:** no edits to `perf_model.py` for this feature set; keep aggregate roofline **unchanged**.

---

## 10. Relation to “Qwen3-17B”

Upstream may ship a **17B** class model not yet in `perf_model`. The plan is: **add one row** to the structure catalog when `config.json` / GGUF metadata is known (`n_layers`, `d_model`, …). The **simulator code** stays the same.

---

## 11. Open questions (to resolve before coding)

1. **Attention core** MVP formula: bytes-only bound vs explicit softmax FLOP term (and coefficients).
2. **Single JSON schema** version field (`"schema_version": 1`).
3. Whether to **import** read-only constants from `perf_model` vs **duplicate** — decision: **duplicate structure table** in the new repo files to avoid coupling and merge conflicts.

---

## Appendix A — `model_structures.json` schema (v1)

**Top-level object**

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `schema_version` | `integer` | yes | Must be `1` for this document. Bump when fields are added or semantics change. |
| `description` | `string` | no | Human-readable note. |
| `presets` | `object` | yes | Map from **preset id** (CLI `--preset` value) to a **model structure** object. |

**Per-preset object (each value in `presets`)**

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `architecture` | `string` | yes | MVP: `"decoder_only"`. |
| `n_layers` | `integer` | yes | Number of transformer blocks. |
| `d_model` | `integer` | yes | Hidden size. |
| `n_heads` | `integer` | yes | Query heads. |
| `n_kv_heads` | `integer` | yes | KV heads (GQA); may be `< n_heads`. |
| `head_dim` | `integer` | yes | Per-head dimension; normally `d_model / n_heads`. |
| `ffn_dim` | `integer` | yes | FFN intermediate (pre-activation); SwiGLU uses three matmuls. |
| `ffn_style` | `string` | yes | MVP: `"swiglu"`. |
| `weight_bits` | `number` | no | **Effective bits/weight for the whole model** (roofline / GEMM bytes — same as `perf_model`; not per-tensor). Use the value from GGUF/metadata or benchmarks. If both `weight_bits` and `weight_tag` are set, **`weight_bits` is authoritative** for numerics. |
| `weight_tag` | `string` | no | e.g. `Q8_0`, `Q4_K_M` — for documentation; if `weight_bits` is omitted, **tag** maps to effective bpw (same rules as `benchmark_kv_timing`). |
| `family` | `string` | no | Hint only: e.g. `"qwen3"`, `"llama"` — for docs / validation messages. |
| `vocab_size` | `integer` | no | If embedding / LM head is modeled later. |
| `rope_theta` | `number` | no | RoPE base; optional for future accuracy tweaks. |

**Optional v2 fields (not in v1 file yet):** `moe_*`, `sliding_window`, etc. — omit for MVP.

**Example — minimal custom file** (user `--structure-json`), same per-preset shape without wrapping `presets` map if the loader accepts “single model” OR use one preset in a file that mirrors the catalog format (loader TBD: either `{ "presets": { "my-run": { ... } } }` or flat `{ ...fields... }` for a single model).

Recommended single-model file for portability:

```json
{
  "schema_version": 1,
  "presets": {
    "custom-from-hf": {
      "family": "llama",
      "architecture": "decoder_only",
      "n_layers": 32,
      "d_model": 4096,
      "n_heads": 32,
      "n_kv_heads": 8,
      "head_dim": 128,
      "ffn_dim": 14336,
      "ffn_style": "swiglu",
      "weight_bits": 16,
      "vocab_size": 128256
    }
  }
}
```

**Adding Qwen3-17B (or similar):** add a new key under `presets` with dimensions from the released `config.json` / GGUF metadata — no code change to the simulator core.

---

*End of plan.*
