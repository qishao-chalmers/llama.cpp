# Left tasks (KV / adaptive research)

## Asymmetric window gen/verify (speculative-style, quant acceptance)

**Recorded:** 2026-04-09T10:05:42+02:00 (when this entry was last updated)

### Glossary (canonical terms — use in markdown and code comments)

| Term | Meaning |
|------|---------|
| **Bootstrap window** | The **first** generation segment: fp16 proposes **W** tokens (ground-truth trajectory for that block). |
| **Quant rollout window** | Each **later** segment: **quant** (draft KV) proposes **W** tokens, then verifier replay checks them. After the bootstrap window, generation proceeds as a sequence of quant rollout windows (plus fp16 fallback when verify fails). |

Related names: **bootstrap-and-rollout** = overall pattern; **re-anchor** = periodic fp16 verification on quant output (see section C).

### Target design (check again)

Work in **windows** of size **W** (see `--adaptive-window`). The loop is **not** “fp16 proposes every window” — only the **bootstrap window** uses fp16 to **propose** first; then **quant rollout windows** use **quant** to generate.

**A. Bootstrap window**  
1. **Propose at full precision** — Decode **W** new tokens with **verifier precision** (fp16 / teacher KV). That fixes the candidate token ids.  
2. **Verify under quant — must be fast (parallel)** — Replay those ids on the **quant** KV path with **parallel** scoring (speculative-style), not Θ(window) serial decodes. Cheap accept/reject vs the expensive proposal.

**B. Quant rollout windows (after the bootstrap window)**  
**Switch** to **quant** (draft / worker quant mode) to generate each **next** segment of **W** tokens. This is the **cheap** path: most forward work is **quant** generation, not fp16.

**C. Re-anchor with full precision**  
On a schedule (e.g. every *k* windows or at boundaries), run **fp16 verification** on quant-generated segments so errors do not compound without a fp16 ground truth. That check should also aim for **parallel** verify where possible.

**Asymmetry vs classic speculative decoding**  
- **Classic speculative:** small **draft** proposes; large **target** verifies.  
- **Here:** **fp16** proposes the **initial** window; **quant** verifies in parallel. Then **quant** generates the **long** middle; **fp16** returns periodically to **verify** quant output, not to propose every token.

**Suggested short names (for readers)**  
- **Bootstrap-and-rollout:** one **bootstrap window** (fp16) + many **quant rollout windows** (quant).  
- **Anchor-propose / quant-carry** (informal): bootstrap window **anchor-proposes**; **quant carries** until the next fp16 re-anchor.

**Why this can be fast in practice**

- **Bootstrap window:** one fp16 proposal + cheap parallel quant verify.  
- **Steady state:** **quant rollout windows** emit **long stretches** of tokens; fp16 is **occasional** (re-anchor), not per-token.  
- **Verification:** parallel under quant (and ideally under fp16 when re-checking) keeps verify from dominating.

### Current scripts (gap)

- `run_adaptive_gen` today: **draft** (`--quants`) generates the window first, then **verifier** replays for `verify_window` (**sequential** per-token decode).  
- It does **not** implement: **bootstrap window** (fp16 → parallel quant verify) → **quant rollout windows** → periodic **fp16** re-verify (see glossary).

**Relevant code**

- `research/scripts/strategies.py` — `verify_window`, `_single_decode`
- `research/scripts/run_sweep.py` — `run_adaptive_gen`

**Implementation notes (open)**

- **Parallel verify is load-bearing:** without batched scoring of the window’s token ids under quant KV, verification cost scales with window length and undermines the design. Target: **few** `llama_decode` steps (or equivalent) for the whole window, analogous to speculative verify.
- Acceptance rule must be defined (strict greedy match vs softer threshold).

---

## Legacy note — old “parallel verification” ask

Earlier wording focused only on speeding up **sequential** `verify_window` under the **verifier** path. The fuller picture: **fp16 proposes only the bootstrap window**, **quant verifies in parallel**, then **quant generates** a long run of tokens, with **periodic fp16** re-verification.

---

*Add further “left tasks” below as needed.*

---

## Combined Weight + KV Cache Quantization

**Recorded:** 2026-04-10

Full design in [`combined_weight_kv_quant.md`](combined_weight_kv_quant.md).

### Core idea

Extend adaptive-gen to also quantize **model weights** (not just KV cache).
Draft uses lower-precision weights (e.g. Q4_K_M) + int2/int3 KV.
Verifier uses higher-precision weights (Q8_0) + fp16/int8 KV.
If acceptance rate stays high, speedup compounds from both axes.

Longer-term: **base+delta weight storage** (W_fp16 = W_q4 + sparse delta) so
both draft and verifier share the same weight memory with no 2× storage cost.

### Near-term implementation

1. **Make draft model** (one-time):
   ```bash
   python3 research/scripts/make_draft_model.py \
       models/Qwen3-8B-Q8_0.gguf Q4_K_M
   # → models/Qwen3-8B-Q8_0-Q4_K_M.gguf
   ```
   Try: Q4_K_M, Q3_K_M, Q2_K.

2. **`--draft-model PATH`** flag in `run_sweep.py` — loads second llama context.
   KV blobs are layout-compatible between Q8_0 and Q4_K_M (same architecture).

3. **KV sync**: at each window boundary, save verifier's KV → restore into
   draft_ctx, then apply int2 hook. Same invariant as current adaptive-gen.

### Open questions

- Does weight quant + KV quant compound errors or stay roughly independent?
- Q4_K_M vs Q3_K_M operating point?
- Is imatrix quantization needed or does magnitude-only suffice at Q4_K_M?
