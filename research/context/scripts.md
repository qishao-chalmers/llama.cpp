# Scripts Reference

All scripts live in `research/scripts/`. Data in `research/data/`. Results in `research/results/`.

## Module Architecture (dependency order, no cycles)
```
llama_bindings.py  →  strategies.py  →  run_sweep.py
quant.py           ↗
parse_state.py     ↗
gpu_quant.py       ↗  (future: GPU path)
```

## llama_bindings.py
Structs, pointer types, load_lib(), tokenize(), log_softmax().
- `load_lib(path=None)` — loads build_release/bin/libllama.so, sets all restype/argtypes
  - Installs a log callback filtering `state_read_meta` spam (fired every decode step for per-token quants); all other INFO-level logs (model loading, KV cache setup, context construction) pass through unchanged
- `detokenize(lib, vocab, tokens, remove_special=True) -> str` — list of token ids → UTF-8 string
- `token_to_piece(lib, vocab, token_id) -> str` — single token id → piece string (e.g. `' hello'`)
- `LlamaKVLayerInfo` — struct for GPU quantization (k_data, v_data, n_cells, k/v_stride, n_embd_k/v, ggml_type, v_trans)

## quant.py
Simulated quantization functions. All: `(np.ndarray float16) → np.ndarray float16`.

Available formats:
```
fp16, bf16, fp8_e4m3, fp8_e5m2
int8, int8_ch, int8_tok, int8_tok_g16, int8_tok_g32, int8_tok_g64
int4, int4_ch, int4_tok, int4_tok_g16, int4_tok_g32, int4_tok_g64
int3, int3_ch, int3_tok, int3_tok_g16, int3_tok_g32, int3_tok_g64
int2, int2_ch, int2_tok
nf4
```
K:V split: `int4_ch:int4_tok` = K uses int4_ch, V uses int4_tok (colon splits K-spec from V-spec).
`PER_TOKEN_QUANTS = {"int8_tok", "int4_tok", "int3_tok", "int2_tok"}` — auto v_group_size=1.

### Per-layer quant specs
- `parse_layer_spec(spec, n_layers) -> list[str]` — parses a layer range spec into a per-layer list
- `resolve_quant_layers(spec, n_layers) -> (k_names, v_names)` — splits on `:` then calls `parse_layer_spec` for each side

Layer range syntax (supports multiple segments separated by `/`):
```
int8_ch                       # all layers: int8_ch
int8_ch@0-15/int4_ch@16-31    # layers 0-15: int8_ch, layers 16-31: int4_ch
fp16@0-7/int4_ch@8-31         # first 8 layers unquantized, rest int4_ch
```
K:V split with layer ranges: `int8_ch@0-15/int4_ch@16-31:int4_tok`
— colon separates K-spec from V-spec; each parsed independently.

## parse_state.py
CPU-side KV quantization via state blob (slow for large contexts).
- `apply_kv_hook(lib, ctx, CP, k_fn, v_fn, seq_id, n_pos_per_embd, n_new_k, n_new_v)`
  - `n_new_k`: quantize last N K cells (None = all)
  - `n_new_v`: quantize last N V cells (None = all) — can differ from n_new_k

## strategies.py
Returns `RunResult(log_probs, log_dists, kl_divs, top1s, diags)` namedtuple (last four default to None).

- `run_chunk_batch_prefill(lib, ctx, tokens, n_vocab, kv_hook, n_prompt, k_group_size, v_group_size, return_log_dists=False, base_log_dists=None, return_top1=False, return_diagnostics=False)`
  - Batch prefill n_prompt tokens → quantize all at once → decode with per-group quantization
  - K quantized every k_group_size tokens; V quantized every v_group_size tokens
  - `return_top1=True`: populates `RunResult.top1s` (argmax token id per decode step)
  - `return_diagnostics=True`: populates `RunResult.diags = {"H": [], "p_max": [], "self_surp": []}`
    - `H`: output entropy in nats — no ground truth needed; rises when KV corruption causes confusion
    - `p_max`: probability of top-1 prediction (model's own confidence)
    - `self_surp`: log-prob of previous step's top-1 under current step's distribution (NaN step 0);
      drops when model's own continuations become incoherent
- `run_chunk_token_by_token(...)` — same interface, no batch prefill
- `run_structured(...)` — wraps run_chunk_batch_prefill for (prompt_tokens, completion_tokens) pairs
- `_collect_logits(...)` — returns current top1 so run functions track prev_top1 for self_surp
- `run_generate(lib, ctx, prompt_tokens, n_vocab, kv_hook=None, max_new_tokens=512, eos_token_id=None, k_group_size=128, v_group_size=128)`
  - Greedy autoregressive generation (for accuracy evaluation, not PPL)
  - Batch prefills prompt with logits only for last prompt token (`batch.logits[n_prompt-1]=1`)
  - First generated token: `llama_get_logits_ith(ctx, n_prompt-1)` — NOT index 0 (common pitfall)
  - Subsequent tokens: single-token decode batches, index 0 is correct
  - Returns list of generated token ids (prompt not included); stops at max_new_tokens or EOS

## run_sweep.py
```bash
# Multi-window mode (primary use):
python3 research/scripts/run_sweep.py model.gguf corpus.txt \
    --prefill-tokens 4096    \  # batch prefill this many tokens
    --score-windows 512 1024 2048 \  # decode 2048, compute PPL at each cutoff
    --n-chunks 5             \  # strided across corpus (not sequential)
    --n-threads 8            \
    --n-gpu-layers 99        \
    --quants fp16 int8_ch int4_ch:int4_tok int3_ch int3_ch:int3_tok \
    --quant-k int8_ch@0-15/int4_ch@16-31 \  # optional: per-layer K spec (merged into --quants)
    --quant-v int4_tok               \  # optional: per-layer V spec (merged into --quants)
    --quant-group-size 128   \  # K group size (V per-token uses 1 automatically)
    --show-text              \  # print prompt+completion + post-run token prediction table
    --out results.json

# Structured mode (prompt/completion JSONL):
python3 research/scripts/run_sweep.py model.gguf data/code_longcode.jsonl \
    --corpus-mode structured \
    --n-ctx 8192             \  # cap per example; prompt truncated from left if needed
    --n-chunks 10            \  # max examples (0 = all)
    --n-gpu-layers 99        \
    --n-threads 8            \
    --quants fp16 int8_ch int4_ch int4_ch:int4_tok \
    --show-text              \
    --out results.json

# Standard flat mode:
python3 research/scripts/run_sweep.py model.gguf corpus.txt \
    --n-ctx 512 --n-chunks 20 --n-threads 8 \
    --n-prompt 400 --prefill-batch \
    --quants fp16 int8_ch int4_ch \
    --out results.json
```

### --show-text output
Prints at the start: detokenized prompt tail + decode target for first chunk/example.
Prints at the end: full token prediction table across all decode steps.
```
  step  actual        fp16                                int8_ch
                      token        lp    H    p    ss     token        lp    H    p    ss
  ─────────────────────────────────────────────────────────────────────────────────────
     0  ' The'        -           +0.0  1.2  .87   nan    -           +0.0  1.2  .87   nan
     1  ' cat'        -           -1.2  0.9  .72  -1.2    ' a'        -2.8  2.1  .45  -2.3
```
5 sub-columns per quant:
- `token`: `-` = correct prediction; predicted piece = wrong
- `lp`: log-prob of the **correct** token (always shown). Near 0 = confident; -5 or lower = off-track.
- `H`: output entropy (nats) — low=confident, high=confused; no ground truth needed
- `p`: top-1 prediction probability
- `ss`: self-surprisal — log-prob of previous step's top-1 under current distribution (NaN step 0)

After the table: summary block with mean of all 5 metrics split by matched vs unmatched tokens.

### --quant-k / --quant-v flags
```bash
--quant-k int8_ch@0-15/int4_ch@16-31    # K: int8_ch for layers 0-15, int4_ch for 16-31
--quant-v int4_tok                       # V: int4_tok for all layers
```
- Coexist with `--quants`; the composed `k_spec:v_spec` is merged into the quant list automatically
- Layer ranges use `name@lo-hi` notation, multiple segments separated by `/`
- Segments without `@lo-hi` apply to all layers: `int8_ch@0-15/int4_ch` sets remaining layers to int4_ch
- If `--quant-k` is set without `--quant-v` (or vice versa), the missing side defaults to `fp16`

### Per-layer group size handling
Mixed per-channel + per-token layers on the same side are handled correctly:
- Per-channel layers (group_size=128): accumulate 128 tokens before firing — no incorrect single-token scales
- Per-token layers (group_size=1): fire every token
- Stateful per-layer pending counters inside `make_kv_hook`; uniform case (all same group_size) uses the fast non-stateful path

### --eval-accuracy (structured mode only)
```bash
python3 research/scripts/run_sweep.py model.gguf data/gsm8k_test.jsonl \
    --corpus-mode structured \
    --n-ctx 2048 --n-chunks 10 \
    --n-gpu-layers 99 --flash-attn --n-threads 20 \
    --quants fp16 int8_ch int4_ch:int4_tok int2_ch \
    --eval-accuracy --max-gen-tokens 512 \
    --answer-regex '####\s*([\d,]+)' \
    --out results.json
```
- Runs greedy generation on each example per quant, extracts answer with `--answer-regex`
- Prints per-example gold/predicted, reports accuracy (% correct)
- Results JSON includes `accuracy`, `n_correct`, `n_total` fields
- n_ctx auto-widened to cover `max(prompt) + max_gen_tokens`
- Default regex `####\s*([\d,]+)` matches GSM8K format `#### 42`

### Incremental JSON saving + log file
`save_results()` is called after each quant finishes — partial results are readable mid-sweep.
Output log automatically saved to `<out>.log` (alongside `<out>.json`), capturing both stdout and stderr.

## submit_sweep.sh
SLURM job array for MareNostrum 5 (H100, acc partition).
```bash
bash research/scripts/submit_sweep.sh \
    --account bsc93 --qos acc_bscls \
    --models "models/Qwen3-8B-Q8_0.gguf" \
    --prefill-tokens "1024 4096 16384" \  # one job per prefill size
    --score-windows "512 1024 2048"    \  # decoded once, PPL at each cutoff
    --quants "fp16 int8_ch int4_ch:int4_tok int3_ch int3_ch:int3_tok int2_ch int2_ch:int2_tok" \
    --quant-group-size 128 \
    --corpus research/data/wikitext2_test.txt \
    --n-chunks 5 --time 08:00:00
```
- Job array = models × prefill_tokens (NOT × score_windows — those are free)
- n_ctx per job = prefill_tokens + max(score_windows), set automatically
- Output: `research/results/results_<model>_prefill<N>_<timestamp>.json`
- Chunks strided evenly across corpus

## visualize_kv_quant.py
Text visualization of KV cache quantization effects. Runs a real model forward pass, extracts a K/V slice, and shows how each quant scheme transforms it.

```bash
# Compare multiple quants side-by-side:
python3 research/scripts/visualize_kv_quant.py model.gguf \
    --quants int8_ch int4_ch int2_ch int4_ch:int4_tok \
    --text-file research/data/wikitext2_test.txt \
    --layer 16 --head 0 --head-dim 128 \
    --n-tokens 128 \
    --verbose --verbose-size 8 \
    --out kv_viz.txt
less -S kv_viz.txt

# Single quant (shorthand):
python3 research/scripts/visualize_kv_quant.py model.gguf \
    --quant int4_ch --n-threads 8 --out kv_viz.txt
```

### Output structure
For each K/V side:
1. **fp16 baseline** — shown once; Unicode block encoding (` ░▒▓█`), anchored to fp16 [min, max]
2. Per quant block:
   - Quantized matrix — same encoding as fp16 (direct visual comparison)
   - Per-dim scales (per-channel quants): block row + first 16 values
   - Per-token scales (per-token quants): appended as `| sc=...` on each row
   - Error matrix `|quant−fp16|` — encoding anchored to [0, max_err]; RMS + max error
3. **Verbose sub-block** (`--verbose`): actual float values for first `--verbose-size` tokens × dims;
   all quants shown together so differences are directly comparable

### Key flags
| Flag | Default | Description |
|------|---------|-------------|
| `--quants` | required | One or more quant specs |
| `--quant` | None | Single quant shorthand (appended to --quants) |
| `--layer` | n_layer//2 | Which transformer layer to visualize |
| `--head` | 0 | KV head index |
| `--head-dim` | 128 | Head dimension (dims head*head_dim : (head+1)*head_dim) |
| `--n-tokens` | 128 | Prefill length; shows min(n_tokens, 128) rows |
| `--verbose` | off | Dump actual floats for n_verbose × n_verbose sub-block |
| `--verbose-size` | 8 | Size of verbose sub-block |

## Corpus Files (research/data/)
- `wikitext2_test.txt`  — 245K tokens
- `wikitext103_test.txt` — 314K tokens
- `c4_val.txt`          — 500K tokens
- `code_longcode.jsonl` — 143 LongCodeArena examples, prompt~4200 tok, completion~600 tok
- `gsm8k_test.jsonl`    — GSM8K test set (1319 examples), 8-shot format (fetch_gsm8k.py)
