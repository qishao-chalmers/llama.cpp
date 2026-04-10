# Adaptive-gen bootstrap (`run_sweep.py` — `run_adaptive_gen`)

This document records the **bootstrap** phase behavior for `--adaptive-gen` (structured corpus + `--eval-accuracy`): windowed candidate verification, steer-back on reject, logging, and quant selection.

**Code:** `research/scripts/run_sweep.py` — `run_adaptive_gen`, bootstrap candidate loop over `--quants` entries (excluding fp16 and verifier).

## Glossary

| Term | Meaning |
|------|---------|
| **Bootstrap reference tokens** | Token sequence used to segment the walk: fp16 bootstrap string, or int8 teacher when `--verifier-quant` is not `fp16` (diagnostics / alignment with verifier path). |
| **Chunk** | Up to `W` tokens where `W = --adaptive-window`. The loop advances `off` by `len(draft_tokens)` or `len(fb_toks)` per iteration. |
| **`window_ok`** | Verifier accepts **every** draft token in the chunk (greedy or `--adaptive-verify-top-k` / `--adaptive-verify-top-p`). |

## Verification vs recovery (chunk vs token)

- **Decision (`window_ok`):** **Token-level.** For each position, verifier logits are compared to the draft token; **all** positions must pass for the chunk to count as a full quant accept.
- **Recovery on failure:** **Chunk-level** (same as post-bootstrap **quant rollout**). If **any** token fails, the implementation **does not** commit a partial quant prefix for that chunk; it **restores** the KV checkpoint at the **start** of the chunk and regenerates the **entire** chunk in **fp16** (`kv_hook=None`). This matches the main generation loop’s reject path.

## Steer-back (bootstrap)

If `window_ok` is false for a chunk, bootstrap **steers back** like rollout:

1. Restore boundary: `pre_prime_blob` for the first chunk, `kv_roll` for later chunks.
2. One fp16 `_single_decode` of the boundary (prime) token — **no** draft quant hook.
3. `generate_window(..., kv_hook=None)` for that chunk length.

Then advance `off` and continue; the candidate is **not** dropped solely because one chunk failed verify.

## Reporting-only splits within **failed** chunks

For metrics only (behavior unchanged): on each failed chunk, compute the **longest prefix** where the verifier still accepts the draft (same rules as full `window_ok`). Accumulate:

| Counter | Meaning |
|---------|---------|
| **`full` + verifier name** | Tokens from chunks where **every** token passed verify (`bootstrap_quant_toks`). |
| **`verif-prefix`** | Sum over **steer** chunks of prefix lengths: tokens that matched **before** the first mismatch (`bootstrap_prefix_ok_toks`). |
| **`chunk-tail`** | Sum over steer chunks of `(len(draft_tokens) − prefix)` — remainder of the draft chunk from first failure onward (`bootstrap_mismatch_toks`). |
| **`fp16 steer`** | Total tokens emitted in fp16 steer paths (`bootstrap_fp16_toks`; usually aligns with draft chunk lengths when EOG does not truncate). |
| **`Nq+Ms chunks`** | `N` chunks with full quant+verifier accept; `M` chunks that required fp16 steer. |

Example (all numbers illustrative):

```text
64/64 tok: 48 full+int8_ch + 12 verif-prefix + 4 chunk-tail + 16 fp16 steer | 12q+4s chunks
```

Interpretation: 48 tokens from fully accepted quant windows; across four failed chunks, 12 tokens still agreed with the verifier before the first disagreement, 4 tokens fell in the “tail” after that; 16 fp16 tokens from steer (here 12+4=16).

## Post-bootstrap **probe** (optional)

After a candidate finishes the bootstrap walk, the script can run **probe windows**: short rollout-style draft+verify trials from the **fp16** `kv_boundary` after the initial fp16 bootstrap (not from the candidate’s mixed KV).

| CLI | Default | Effect |
|-----|---------|--------|
| `--bootstrap-probe-windows N` | **`0`** | Number of probe rollout windows. **`0` = disabled** (no probe loop; probe code remains in place). |
| `--bootstrap-pick-epsilon E` | **`0.02`** | With probe disabled: from candidates within **E** of the best **agree rate**, pick **lowest bits** (see below). |

When **disabled** (`N = 0`):

- Log shows `→ probe disabled`.
- **`bootstrap_pick`:** **Agree rate** = `(full-window tokens + verif-prefix tokens) / bootstrap reference length`. Take the **best** agree rate among candidates; keep any candidate within **`--bootstrap-pick-epsilon`** of that best; among those, pick **lowest bit width**. If the best rate is unique (no one else within ε), that candidate wins; if several are close (e.g. 0.78 vs 0.80 with ε = 0.02), the band can include multiple quants and **min bits** breaks the tie toward a more aggressive quant.

When **enabled** (`N > 0`, e.g. `4`):

- Records `probe_accepted / probe_attempted` and compares to an internal minimum rate (e.g. 0.90) for PASS/FAIL in the log.
- Pick prefers candidates that **pass** the probe threshold, then **lowest bits**.

## Relation to main rollout

After `bootstrap_pick`, generation uses the chosen draft hook; quant rollout windows use the same verify / fp16-fallback pattern as documented in `scripts.md` (adaptive gen section).

## See also

- `research/context/scripts.md` — CLI table and adaptive flags (short form).
- `research/context/left_tasks.md` — glossary: bootstrap window vs quant rollout window; broader design notes.
