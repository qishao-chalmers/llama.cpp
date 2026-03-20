# Analysis Tools Reference

All analysis scripts read output files from `run_sweep.py`. None require re-running the model.

---

## plot_entropy.py — Per-token signal vs position (flat/PPL mode)

**Input**: `--save-diags FILE` JSON from `run_sweep.py`
**Use for**: wikitext, code — anywhere you have teacher-forced PPL with per-token diagnostics.

```bash
# Time-series: entropy per quant vs decode position
python3 research/scripts/plot_entropy.py research/results/wiki_diags.json \
    --quants fp16 int3_ch int2_ch --metric H --smooth 50 --out entropy.png

# Delta (quant − fp16): where does divergence concentrate?
python3 research/scripts/plot_entropy.py research/results/wiki_diags.json \
    --metric H --diff --smooth 30 --out entropy_diff.png

# Multi-panel: H + p_max + ppl_curve together
python3 research/scripts/plot_entropy.py research/results/wiki_diags.json \
    --metrics H p_max ppl_curve --smooth 50 --out multi.png

# Correlation mode: do H/p_max/self_surp predict KL? → deployable alarm signal
python3 research/scripts/plot_entropy.py research/results/wiki_diags.json \
    --corr --kl-threshold 0.1 --out corr_plot.png
```

**Metrics**: `H` (entropy, nats), `lp` (log-prob correct token), `p_max` (top-1 prob),
`self_surp` (log-prob of prev top-1 under current dist), `kl` (vs fp16), `ppl_curve`.

**`--corr` mode**: answers "can I use H as an alarm in deployment without a fp16 reference?"
- Scatter plot: signal vs KL per token
- Pearson/Spearman correlation table (stdout)
- ROC curves: TPR/FPR as alarm threshold is swept per signal × quant

---

## analyze_alarms.py — Alarm signal evaluation (accuracy mode)

**Input**: `--save-per-example FILE` JSON from `run_sweep.py --eval-accuracy`
**Use for**: GSM8K, NIAH — anywhere you have per-example correct/wrong labels + gen_diags.

```bash
# Value distributions: what does H look like for correct vs wrong examples?
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json --summary

# TPR/FPR table at default thresholds for all 4 metrics
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json

# Sweep thresholds for one metric (ROC table: TPR/FPR per row)
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json \
    --metric H --sweep

# Focus on specific quants and metric with custom threshold
python3 research/scripts/analyze_alarms.py research/results/gsm8k_per_ex.json \
    --quants int3_ch int2_ch --metric rep_rate --threshold 0.3
```

**Output columns**:
| Column | Meaning |
|--------|---------|
| `n_wrong` / `n_correct` | examples with wrong / correct answers |
| `TPR` | fraction of wrong examples where alarm fires at least once |
| `FPR` | fraction of correct examples where alarm fires (false alarm rate) |
| `alarm@step` | mean step where alarm first fires (in wrong examples) |
| `gen_len` | mean total generation length (wrong examples) |
| `lead_time` | `gen_len − alarm@step` — steps of advance warning |

**Metrics and alarm directions**:
| Metric | Alarm when | Meaning |
|--------|-----------|---------|
| `H` | H > threshold | High entropy = model confused |
| `p_max` | p_max < threshold | Low top-1 prob = uncertain |
| `rep_rate` | rep_rate > threshold | >X% repeated 3-grams in last 20 tokens = loop |
| `self_surp` | self_surp < threshold | Model's own prev prediction is now unlikely = incoherent |

**Good alarm signal**: high TPR, low FPR, positive lead_time.
`rep_rate` is most reliable (unambiguous loop = definite failure).
`H` is more sensitive but has more false alarms.

---

## plot_niah.py — NIAH U-curve (needle position analysis)

**Input**: `--save-per-example FILE` JSON from `run_sweep.py --eval-accuracy`
**Use for**: NIAH only — plots accuracy vs needle position.

```bash
python3 research/scripts/plot_niah.py research/results/niah_per_ex.json \
    --jsonl research/data/niah_4096.jsonl \
    --show-gen-len \
    --out niah_plot.png
```

Plots accuracy ± SE vs position (0.0=start, 1.0=end) per quant.
U-shape = lost-in-the-middle: model handles start/end better than middle.
`--show-gen-len`: adds subplot of mean gen length vs position.

---

## plot_results.py — Summary bar chart across quants

**Input**: `--out FILE` results JSON from `run_sweep.py`

```bash
python3 research/scripts/plot_results.py research/results/wiki_results.json \
    --metric ppl --out results_bar.png
```

---

## Combining tools (typical workflow)

```
run_sweep.py  ──┬── --save-diags ──────► plot_entropy.py (--corr for alarm correlation)
                │
                └── --save-per-example ─► analyze_alarms.py (TPR/FPR/lead-time)
                                        ► plot_niah.py      (NIAH U-curve)
```
