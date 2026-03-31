# Task Guide — What to Run for Each Evaluation

Quick reference: pick your task, get the commands.

## Benchmark Overview

| Benchmark | Prompt length | Output length | Evaluation | What can fail |
|-----------|--------------|---------------|------------|---------------|
| WikiText2 | 2048 tok chunks | n/a (teacher-forced) | Perplexity | Distribution shift |
| GSM8K | ~1500 tok | ~50–300 tok | Exact number match | Reasoning chain breaks |
| NIAH | ~128K tok (H100) | ~5 tok | F1 on passphrase | Retrieval from middle context |
| LongBench qasper | ~8192 tok | ~50 tok (fp16) / loops (int2) | F1 on answer | Comprehension + generation collapse |

### WikiText2 / C4 — Flat Perplexity
- **Prompt/output**: No split — continuous text; model predicts each next token (teacher-forced).
- **Focus**: General language modeling quality. Does quantization make the model "worse at language" on average?
- **Evaluation**: Perplexity (lower = better) + KL divergence vs fp16.

### GSM8K — Math Reasoning Accuracy
- **Prompt**: 8 worked examples (few-shot) + a new math word problem. ~1000–2000 tokens.
- **Output**: Step-by-step reasoning chain ending in "The answer is 42." ~50–300 tokens.
- **Focus**: Does quantization break multi-step arithmetic reasoning chains?
- **Evaluation**: Regex extracts final number; exact match against gold. Score = 0 or 1.

### NIAH — Needle in a Haystack
- **Prompt**: ~128K tokens of C4 filler with one passphrase inserted at a controlled position (0%–100%).
- **Output**: Model asked to recall the magic word. 1–5 tokens expected.
- **Focus**: Long-context retrieval. Does quantization amplify the "lost in the middle" effect?
- **Evaluation**: F1 against passphrase words. 110 examples = 11 positions × 10 replicates.

### LongBench Qasper — Long-Document QA
- **Prompt**: Full scientific paper (up to ~6000 tokens) + question, wrapped in chat format. Up to 8192 tokens.
- **Output**: 1–2 sentence answer (~20–100 tokens fp16; repetition loops when int2 fails).
- **Focus**: Does quantization impair reading comprehension over long, information-dense documents?
- **Evaluation**: F1 between generated answer and gold phrases (multiple valid answers allowed).

---

---

## 1. Wikitext PPL Sweep (flat text, teacher-forced)

**Purpose**: measure perplexity degradation across quant types at different decode lengths.
**Ground truth**: none needed — model predicts next token given all previous.
**Key signal**: PPL, KL divergence vs fp16.

```bash
# Fetch dataset (one-time)
python3 research/scripts/fetch_wikitext.py

# Run sweep (multi-window mode — one prefill, PPL at 512/1024/2048)
python3 research/scripts/run_sweep.py \
    /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
    research/data/wikitext2_test.txt \
    --prefill-tokens 1024 \
    --score-windows 512 1024 2048 \
    --n-chunks 5 --n-threads 20 \
    --n-gpu-layers 99 --flash-attn \
    --quants fp16 int8_ch int4_ch int3_ch int2_ch \
    --save-diags research/results/wiki_diags.json \
    --out research/results/wiki_results.json
```

**Analysis**:
```bash
# Plot entropy vs position — where does int2 diverge?
python3 research/scripts/plot_entropy.py research/results/wiki_diags.json \
    --metrics H p_max --diff --smooth 30 --out wiki_entropy.png

# Correlation: do H/p_max predict KL? (deployable alarm signal)
python3 research/scripts/plot_entropy.py research/results/wiki_diags.json \
    --corr --out wiki_corr.png
```

**Note**: PPL is meaningful here. Do NOT use `--skip-ppl`.

---

## 2. GSM8K Reasoning Accuracy

**Purpose**: measure accuracy on math word problems; test if quantization breaks reasoning chains.
**Ground truth**: numeric answer extracted by regex from model generation.
**Key signal**: accuracy, gen_len (longer = more uncertainty), n_hedges (self-doubt count).

```bash
# Fetch dataset (one-time, 8-shot format)
python3 research/scripts/fetch_gsm8k.py

# Run sweep
python3 research/scripts/run_sweep.py \
    /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
    research/data/gsm8k_test.jsonl \
    --corpus-mode structured \
    --eval-accuracy \
    --n-ctx 4096 --n-chunks 100 \
    --n-gpu-layers 99 --flash-attn --n-threads 20 \
    --quants fp16 int8_ch int4_ch int3_ch int2_ch \
    --max-gen-tokens 512 \
    --stop-strings $'\n\nQuestion:' \
    --save-per-example research/results/gsm8k_per_ex.json \
    --out research/results/gsm8k_results.json
```

**Analysis**:
```bash
# Alarm signal analysis: does H/p_max/rep_rate predict wrong answers?
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json --summary
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json \
    --metric H --sweep    # full ROC table
```

**Pitfalls**:
- PPL ≠ accuracy: teacher-forced PPL can be low even when generation fails
- Use `--stop-strings $'\n\nQuestion:'` to stop before next question prefix
- Default regex matches both `#### 42` and `The answer is 42`
- `inconclusive`: truncated while model was still doubting → scored 0

---

## 3. Needle-In-A-Haystack (NIAH) — Lost-in-the-Middle

**Purpose**: measure whether quantization amplifies the U-shaped accuracy drop at middle positions.
**Ground truth**: F1 against passphrase word verbatim in context.
**Key signal**: accuracy vs needle position curve (U-shape?), gen_len.

```bash
# Build dataset (one-time, uses model for exact token counts)
python3 research/scripts/build_niah_dataset.py \
    /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
    research/data/c4_val.txt \
    --out research/data/niah_4096.jsonl \
    --target-ctx 4096 --n-positions 11 --n-needles 10

# Run sweep (--skip-ppl: ppl≈1.0 always since passphrase is verbatim in context)
python3 research/scripts/run_sweep.py \
    /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
    research/data/niah_4096.jsonl \
    --corpus-mode structured \
    --eval-accuracy --eval-metric f1 \
    --skip-ppl \
    --n-ctx 4160 --n-chunks 0 \
    --n-gpu-layers 99 --flash-attn --n-threads 20 \
    --quants fp16 int8_ch int4_ch int3_ch int2_ch \
    --max-gen-tokens 20 --stop-strings $'\n' \
    --save-per-example research/results/niah_per_ex.json \
    --out research/results/niah_results.json

# Submit to cluster (MN5)
sbatch research/scripts/submit_niah.sh
```

**Analysis**:
```bash
# U-curve: accuracy vs needle position per quant
python3 research/scripts/plot_niah.py research/results/niah_per_ex.json \
    --jsonl research/data/niah_4096.jsonl --show-gen-len --out niah_plot.png

# Alarm signals (rep_rate most useful here — short generation, so loops = obvious fail)
python3 research/scripts/analyze_alarms.py research/results/niah_per_ex.json \
    --metric rep_rate --sweep
```

**Dataset details**: 110 examples = 11 positions × 10 replicates, ~4092 tokens each, C4 haystack.
**If fp16 = 100% everywhere**: context too easy → rebuild at 8192 tokens.
**Reduce examples**: `--n-needles 3` → 33 examples; `--n-positions 7 --n-needles 3` → 21 examples.

---

## 4. Code Completion (LongCodeArena)

**Purpose**: measure PPL/KL on long-context code completions.
**Ground truth**: next tokens of a real codebase file.
**Key signal**: PPL, KL vs fp16.

```bash
# Fetch dataset (one-time)
python3 research/scripts/fetch_code.py

# Run sweep
python3 research/scripts/run_sweep.py \
    /home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf \
    research/data/code_longcode.jsonl \
    --corpus-mode structured \
    --n-ctx 8192 --n-chunks 20 \
    --n-gpu-layers 99 --flash-attn --n-threads 20 \
    --quants fp16 int8_ch int4_ch int3_ch int2_ch \
    --save-diags research/results/code_diags.json \
    --out research/results/code_results.json
```

---

## Choosing `--skip-ppl`

| Task | Use `--skip-ppl`? | Reason |
|------|-------------------|--------|
| Wikitext PPL | **No** | PPL is the whole point |
| Code PPL | **No** | PPL is the whole point |
| GSM8K accuracy-only | **Yes** | Saves ~half the compute; PPL doesn't predict accuracy |
| NIAH | **Always** | ppl≈1.0 (passphrase is verbatim → trivially predicted) |
| Any `--eval-accuracy` where you want speed | **Yes** | Skip teacher-forced pass |

---

## Choosing `--save-per-example` vs `--save-diags`

| Flag | Mode | What it saves | Analysis tool |
|------|------|---------------|---------------|
| `--save-diags` | flat or structured (PPL pass) | per-token H/lp/p_max/self_surp/kl vs position | `plot_entropy.py` |
| `--save-per-example` | structured + `--eval-accuracy` | per-example score/gold/pred/gen_len + per-step gen_diags | `analyze_alarms.py`, `plot_niah.py` |

Both can be set simultaneously for structured accuracy runs.
