# Plan: Reduce quantization overhead in the research KV framework

## Goals

- **Primary:** Lower wall-clock cost of KV quantization on the **hot path** (decode + hook), without changing experimental semantics unless explicitly opted in.
- **Secondary:** Keep **`--profile-kv`** (and JSON output) as the **before/after** yardstick per quant and per mode (CPU vs GPU hook).

---

## Phase 0 — Lock semantics & measurement (1–2 sessions)

| Step | Action |
|------|--------|
| 0.1 | **Document the contract** for one run: hook cadence (group size), CPU vs GPU path, zone hooks (`sink` / `recent`), adaptive sim vs normal generation. |
| 0.2 | **Baseline matrix:** same model, same corpus slice, **fp16** vs **quant** with `--profile-kv`, note `decode_s`, `kv_cpu` vs `kv_gpu`, `n_*_hook_calls`, total wall time. |
| 0.3 | **Sanity check:** confirm GPU quant path (CuPy + flash) is actually used when intended; one mis-flag dominates everything. |

**Exit:** A small table or JSON you trust as “current overhead.”

---

## Phase 1 — Remove accidental overhead (highest ROI / low risk)

| # | Idea | Why |
|---|------|-----|
| 1.1 | **Avoid full `cudaDeviceSynchronize()` when safe** | End-of-hook sync forces a global drain; often the dominant cost if hooks fire often. |
| 1.2 | **Single sync per “batch” of hook work** | If multiple logical updates can share one sync, sync once after all layers / both K+V for that fire. |
| 1.3 | **Reduce hook frequency where experiments allow** | Larger effective group size → fewer hooks → fewer syncs (accuracy trade — only where the study allows). |
| 1.4 | **CPU path:** avoid redundant **get/parse/pack/set** on unchanged cells | If `kv_get` / `kv_pack` dominate in `--profile-kv`, consider smaller views or dirty ranges (bigger change; see Phase 3). |

**Exit:** Re-run Phase 0 matrix; target a meaningful drop in `gpu_kv_s` / wall time without changing quant math.

---

## Phase 2 — GPU hook implementation (medium effort, high payoff on GPU)

| # | Idea | Why |
|---|------|-----|
| 2.1 | **Fuse per-layer quant** | One kernel launch per tensor type (or one per layer pair K+V) vs many tiny launches reduces launch overhead. |
| 2.2 | **Use CUDA events** (optional) | Separate “GPU time” vs “CPU launch overhead” for debugging; guides 2.1. |
| 2.3 | **Stream discipline** | Ensure quant runs on the **same stream** as KV writers (or ordered streams) to avoid accidental cross-stream sync. |
| 2.4 | **Avoid Python loop per layer in the hottest path** | Batch layer pointers into one C++/CUDA entry if profiling shows Python + launch overhead is large. |

**Exit:** `gpu_kv_s` down at the same `n_gpu_kv_hook_calls`; or same cost with fewer effective launches.

---

## Phase 3 — Structural changes (larger effort / may touch semantics)

| # | Idea | Risk |
|---|------|------|
| 3.1 | **Quant only “new” rows** without touching full cache metadata | Must stay consistent with reference `parse_state` behavior — needs careful tests. |
| 3.2 | **Async quant behind decode** | Changes error semantics vs strict “quant after every decode”; only for specific experiments. |
| 3.3 | **Integrate with llama.cpp native KV quant** (if/when available) | Best long-term for production-like perf; research scripts remain for ablations. |

**Exit:** Only after Phase 1–2 plateau.

---

## Phase 4 — Guardrails

- **Regression tests:** Small fixed prompt + seed; compare logits or token ids fp16 vs quant hook schedule (or saved golden hashes) after each optimization.
- **Feature flags:** e.g. `KV_QUANT_SYNC_MODE=full|minimal` so papers can cite exact behavior.

---

## Suggested order of attack

1. **Profile** (Phase 0) → confirm whether **`gpu_kv_s`**, **`decode_s`**, or **sync** dominates.
2. **Phase 1.1–1.2** (sync / batching) if GPU hook + frequent fires.
3. **Phase 2.1** (fuse launches) if profiling shows launch overhead.
4. **Phase 1.4 / 3.1** only if CPU path or blob I/O dominates.

---

## Success criteria

- **Quant / fp16 wall-time ratio** drops on a standard benchmark command (same flags, same machine).
- **`--profile-kv`:** either lower **`gpu_kv_s`** per token (or per hook), or **fewer** `n_gpu_kv_hook_calls` without hurting the research questions.

---

## Related files

- `research/scripts/kv_profile.py` — timing aggregates (`decode_s`, CPU KV phases, `gpu_kv_s`).
- `research/scripts/run_sweep.py` — `--profile-kv`, `--profile-kv-out`.
- `research/scripts/parse_state.py` — CPU serialize/parse/quant/pack path.
- `research/scripts/gpu_quant.py` — CuPy in-place GPU KV quant + sync.
