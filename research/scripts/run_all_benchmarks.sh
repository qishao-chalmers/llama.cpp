#!/usr/bin/env bash
# run_all_benchmarks.sh — Full KV cache quantization benchmark suite
#
# Usage:
#   bash research/scripts/run_all_benchmarks.sh
#   bash research/scripts/run_all_benchmarks.sh --models qwen3-8b qwen3-14b
#   bash research/scripts/run_all_benchmarks.sh --benchmarks wikitext2 gsm8k
#   bash research/scripts/run_all_benchmarks.sh --skip niah-128k humaneval
#
# Available benchmarks:
#   wikitext2   WikiText2 flat perplexity
#   gsm8k       Math reasoning accuracy (8-shot)
#   longbench   Long-document QA (qasper F1)
#   niah-4k     Needle-in-a-haystack at 4K context
#   niah-16k    Needle-in-a-haystack at 16K context
#   niah-32k    Needle-in-a-haystack at 32K context
#   niah-128k   Needle-in-a-haystack at 128K context
#   humaneval   Code completion pass@1
#   aime        AIME math competition accuracy
#
# Results saved to: research/results/{model_name}/

set -euo pipefail
cd "$(dirname "$0")/../.."   # run from llama.cpp root

# ── Model registry ────────────────────────────────────────────────────────────
# Add / remove models here. Fields: name  path  n_gpu_layers  n_threads
declare -A MODEL_PATH=(
    [qwen3-8b]="models/Qwen3-8B-Q8_0.gguf"
    [qwen3-14b]="models/Qwen3-14B-Q8_0.gguf"
    [llama4-scout-17b]="models/Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf"
    [gpt-oss-20b]="models/gpt-oss-20b-Q8_0.gguf"
)
declare -A MODEL_GPU_LAYERS=(
    [qwen3-8b]=99
    [qwen3-14b]=99
    [llama4-scout-17b]=99
    [gpt-oss-20b]=99
)
declare -A MODEL_THREADS=(
    [qwen3-8b]=20
    [qwen3-14b]=20
    [llama4-scout-17b]=20
    [gpt-oss-20b]=20
)

# Chat template for LongBench only — needs a "Answer briefly" system prompt.
# GSM8K, NIAH, AIME, HumanEval all use auto-detection or no prefix (see below).
declare -A MODEL_LONGBENCH_PREFIX=(
    [qwen3-8b]=$'<|im_start|>system\nAnswer briefly in 1-2 sentences based only on the provided text.<|im_end|>\n<|im_start|>user\n'
    [qwen3-14b]=$'<|im_start|>system\nAnswer briefly in 1-2 sentences based only on the provided text.<|im_end|>\n<|im_start|>user\n'
    [llama4-scout-17b]=$'<|header_start|>system<|header_end|>\nAnswer briefly in 1-2 sentences based only on the provided text.<|eot|>\n<|header_start|>user<|header_end|>\n'
    # gpt-oss-20b: built dynamically via _gpt_oss_lb_prefix $GPT_OSS_EFFORT_LB in the model loop
)
declare -A MODEL_LONGBENCH_SUFFIX=(
    [qwen3-8b]=$'<|im_end|>\n<|im_start|>assistant\n'
    [qwen3-14b]=$'<|im_end|>\n<|im_start|>assistant\n'
    [llama4-scout-17b]=$'<|eot|>\n<|header_start|>assistant<|header_end|>\n'
    [gpt-oss-20b]=$'<|end|>\n<|start|>assistant'
)
declare -A MODEL_LONGBENCH_STOP=(
    [qwen3-8b]="<|im_end|>"
    [qwen3-14b]="<|im_end|>"
    [llama4-scout-17b]="<|eot|>"
    # gpt-oss: no explicit stop — <|end|> is inter-channel separator (analysis→final),
    # not EOG. <|return|> (real EOG) is caught by llama_vocab_is_eog.
    [gpt-oss-20b]=""
)

# ── GPT-OSS task-specific reasoning effort ────────────────────────────────────
# The model reads "Reasoning: low/medium/high" from its system message and adjusts
# CoT depth accordingly. Defaults below; override all with --effort LEVEL.
GPT_OSS_EFFORT_AIME="high"     # hard math competition — full CoT needed
GPT_OSS_EFFORT_GSM8K="medium"  # grade school math + 8-shot examples
GPT_OSS_EFFORT_NIAH="low"      # pure retrieval — long CoT loses focus
GPT_OSS_EFFORT_LB="low"        # retrieval QA — "Answer briefly" already set

# LongBench needs task-specific content ("Answer briefly") in the system message,
# so we build it explicitly here. Other tasks pass --effort to run_sweep.py directly
# and let it inject the system message via auto-detection.
_gpt_oss_lb_prefix() {
    printf '<|start|>system<|message|>Reasoning: %s\n\nAnswer briefly in 1-2 sentences based only on the provided text.\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\n<|start|>user<|message|>' "$1"
}

# ── Quant sets ─────────────────────────────────────────────────────────────────
QUANTS="fp16 int8_ch int4_ch int3_ch int3_half_1357_ch int3_half_1357_ch:int3_half_1357_tok int2_ch"
QUANTS_EXPLICIT=0      # set to 1 when --quants is passed; triggers auto-suffix

# ── Common flags ───────────────────────────────────────────────────────────────
COMMON="--asym --sink-tokens 32 --recent-tokens 32 --flash-attn --show-prompt --show-gen"

# ── Output suffix ──────────────────────────────────────────────────────────────
# Auto-derived from --quants when specified (e.g. "int8_ch" -> aime_int8_ch.json).
# Override with --out-suffix "name" or suppress with --out-suffix "".
OUT_SUFFIX=""

# ── CLI options ────────────────────────────────────────────────────────────────
ALL_BENCHMARKS=(wikitext2 gsm8k longbench niah-4k humaneval aime)
# niah-16k/32k/128k excluded by default: lost-in-middle is already severe at 4k.
# Re-enable with: --benchmarks niah-16k niah-32k niah-128k
RUN_MODELS=("qwen3-8b" "qwen3-14b" "llama4-scout-17b" "gpt-oss-20b")
RUN_BENCHMARKS=("${ALL_BENCHMARKS[@]}")   # default: all

ALL_QUANTS=($QUANTS)
ALL_MODELS=("${!MODEL_PATH[@]}")

should_run() { [[ " ${RUN_BENCHMARKS[*]} " == *" $1 "* ]]; }

show_help() {
    cat <<'HEADER'
run_all_benchmarks.sh — Full KV cache quantization benchmark suite

Usage:
    bash research/scripts/run_all_benchmarks.sh [OPTIONS]

Options:
    --models MODEL [MODEL ...]        Models to run (default: all)
    --benchmarks BENCH [BENCH ...]    Benchmarks to run (default: all)
    --skip BENCH [BENCH ...]          Benchmarks to skip
    --effort LEVEL                    GPT-OSS reasoning effort (low/medium/high)
    --help                            Show this help message

HEADER
    printf "Available models:\n"
    for m in "${ALL_MODELS[@]}"; do
        printf "    %-25s %s\n" "$m" "${MODEL_PATH[$m]}"
    done

    printf "\nAvailable benchmarks:\n"
    printf "    %-15s %s\n" "wikitext2"  "WikiText2 flat perplexity"
    printf "    %-15s %s\n" "gsm8k"      "Math reasoning accuracy (8-shot, n-ctx=8192)"
    printf "    %-15s %s\n" "longbench"  "Long-document QA / qasper F1 (n-ctx=16384)"
    printf "    %-15s %s\n" "niah-4k"    "Needle-in-a-haystack at 4K context"
    printf "    %-15s %s\n" "niah-16k"   "Needle-in-a-haystack at 16K context"
    printf "    %-15s %s\n" "niah-32k"   "Needle-in-a-haystack at 32K context"
    printf "    %-15s %s\n" "niah-128k"  "Needle-in-a-haystack at 128K context"
    printf "    %-15s %s\n" "humaneval"  "Code completion pass@1 (n-ctx=4096)"
    printf "    %-15s %s\n" "aime"       "AIME math competition accuracy (n-ctx=32768)"

    printf "\nAvailable quantizations:\n    "
    printf "%s  " "${ALL_QUANTS[@]}"
    printf "\n"

    printf "\nDefault benchmarks: %s\n" "${ALL_BENCHMARKS[*]}"
    printf "Default models:     %s\n" "${RUN_MODELS[*]}"

    printf "\nExamples:\n"
    printf "    bash research/scripts/run_all_benchmarks.sh --models qwen3-8b --benchmarks gsm8k\n"
    printf "    bash research/scripts/run_all_benchmarks.sh --skip niah-128k humaneval\n"
    printf "    bash research/scripts/run_all_benchmarks.sh --models gpt-oss-20b --effort high\n"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help ;;
        --models)
            shift; RUN_MODELS=()
            while [[ $# -gt 0 && $1 != --* ]]; do RUN_MODELS+=("$1"); shift; done ;;
        --benchmarks)
            shift; RUN_BENCHMARKS=()
            while [[ $# -gt 0 && $1 != --* ]]; do RUN_BENCHMARKS+=("$1"); shift; done ;;
        --skip)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                RUN_BENCHMARKS=("${RUN_BENCHMARKS[@]/$1}")
                shift
            done ;;
        --quants)
            QUANTS="$2"; QUANTS_EXPLICIT=1; shift 2 ;;
        --out-suffix)
            OUT_SUFFIX="$2"; shift 2 ;;
        --effort)
            shift
            GPT_OSS_EFFORT_AIME="$1"
            GPT_OSS_EFFORT_GSM8K="$1"
            GPT_OSS_EFFORT_NIAH="$1"
            GPT_OSS_EFFORT_LB="$1"
            shift ;;
        *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

# Auto-derive OUT_SUFFIX from quant list when --quants was given explicitly.
# spaces -> _, colons -> - (e.g. "int8_ch int4_ch" -> "int8_ch_int4_ch")
if [[ $QUANTS_EXPLICIT -eq 1 && -z "$OUT_SUFFIX" ]]; then
    OUT_SUFFIX=$(echo "$QUANTS" | tr ' ' '_' | tr ':' '-')
fi
# FSUFFIX: "_int8_ch" when set, "" when not — appended to every output filename stem.
FSUFFIX="${OUT_SUFFIX:+_${OUT_SUFFIX}}"

echo "Models:     ${RUN_MODELS[*]}"
echo "Benchmarks: ${RUN_BENCHMARKS[*]}"
echo "Quants:     $QUANTS"
[[ -n "$FSUFFIX" ]] && echo "Out suffix: $FSUFFIX"
echo "GPT-OSS effort: aime=$GPT_OSS_EFFORT_AIME  gsm8k=$GPT_OSS_EFFORT_GSM8K  niah=$GPT_OSS_EFFORT_NIAH  lb=$GPT_OSS_EFFORT_LB"
echo

# ── Helpers ────────────────────────────────────────────────────────────────────
PY="python3"
SWEEP="$PY research/scripts/run_sweep.py"
ALARMS="$PY research/scripts/analyze_alarms.py"
EVAL_CODE="$PY research/scripts/eval_code.py"

log() { echo; echo "━━━ $* ━━━"; echo; }

run_benchmark() {
    local name="$1"; shift
    log "$name"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

diagnose_alarms() {
    local per_ex="$1"
    local label="$2"
    if [[ -f "$per_ex" ]]; then
        log "ALARMS — $label"
        $ALARMS "$per_ex" --summary 2>&1 | tee -a "$LOG_FILE" || true
        $ALARMS "$per_ex" 2>&1 | tee -a "$LOG_FILE" || true
    else
        echo "  [skip alarms — $per_ex not found]"
    fi
}

# ── Main loop ──────────────────────────────────────────────────────────────────
for MODEL in "${RUN_MODELS[@]}"; do
    MODEL_FILE="${MODEL_PATH[$MODEL]}"
    GPU="${MODEL_GPU_LAYERS[$MODEL]}"
    THREADS="${MODEL_THREADS[$MODEL]}"
    OUTDIR="research/results/$MODEL"
    mkdir -p "$OUTDIR"
    LOG_FILE="$OUTDIR/run.log"

    echo "================================================================"
    echo "  MODEL: $MODEL  →  $MODEL_FILE"
    echo "  Results: $OUTDIR"
    echo "================================================================"
    > "$LOG_FILE"   # reset log

    GPU_FLAGS="--n-gpu-layers $GPU --n-threads $THREADS"

    # LongBench variables are only needed when running longbench.
    # Use ${..:-} to avoid unbound-variable errors under set -u.
    LB_PREFIX="${MODEL_LONGBENCH_PREFIX[$MODEL]:-}"
    LB_SUFFIX="${MODEL_LONGBENCH_SUFFIX[$MODEL]:-}"
    LB_STOP="${MODEL_LONGBENCH_STOP[$MODEL]:-}"
    if [[ "$MODEL" == "gpt-oss-20b" ]]; then
        LB_PREFIX="$(_gpt_oss_lb_prefix "$GPT_OSS_EFFORT_LB")"
    fi

    # ── 1. WikiText2 PPL ───────────────────────────────────────────────────────
    if should_run wikitext2; then
        run_benchmark "WikiText2 PPL — $MODEL" \
            $SWEEP "$MODEL_FILE" research/data/wikitext2_test.txt \
            --prefill-tokens 1024 --score-windows 512 1024 2048 \
            --n-chunks 20 $GPU_FLAGS $COMMON \
            --quants $QUANTS \
            --out "$OUTDIR/wikitext2${FSUFFIX}.json"
    fi

    # ── 2. GSM8K Accuracy ─────────────────────────────────────────────────────
    # 8-shot few-shot is inside the user turn. No system prompt — the examples
    # already demonstrate the Q/A format and CoT style.
    # n-ctx 4096: 8-shot prompt ~500 tok + question ~100 tok + 1024 gen = ~1624 min.
    # prefix/suffix/stop: auto-detected from model vocab.
    # \n\nQuestion: stop: auto-added by task rule (gsm8k keyword).
    if should_run gsm8k; then
        # prefix/suffix: auto-detected from vocab; gpt-oss also gets --effort for system message.
        EFFORT_ARGS=()
        [[ "$MODEL" == "gpt-oss-20b" ]] && EFFORT_ARGS=(--effort "$GPT_OSS_EFFORT_GSM8K")
        run_benchmark "GSM8K — $MODEL" \
            $SWEEP "$MODEL_FILE" research/data/gsm8k_test.jsonl \
            --corpus-mode structured --eval-accuracy \
            --n-ctx 8192 --n-chunks 20 $GPU_FLAGS $COMMON \
            --quants $QUANTS \
            --max-gen-tokens 4096 \
            "${EFFORT_ARGS[@]}" \
            --out "$OUTDIR/gsm8k${FSUFFIX}.json"

        diagnose_alarms "$OUTDIR/gsm8k${FSUFFIX}_per_example.json" "GSM8K $MODEL"
    fi

    # ── 3. LongBench Qasper ───────────────────────────────────────────────────
    # Explicit prefix — includes "Answer briefly" system prompt needed for
    # retrieval tasks. Auto-detect is skipped when prefix is set explicitly.
    if should_run longbench; then
        # gpt-oss: <|end|> would cut off at analysis channel before final answer.
        # Rely on <|return|> EOG instead; $'\nHuman:' guard not needed for gpt-oss format.
        if [[ "$MODEL" == "gpt-oss-20b" ]]; then
            LB_STOP_ARGS=()
        else
            LB_STOP_ARGS=(--stop-strings "$LB_STOP" --stop-strings $'\nHuman:')
        fi
        run_benchmark "LongBench Qasper — $MODEL" \
            $SWEEP "$MODEL_FILE" research/data/LongBench/qasper.jsonl \
            --corpus-mode structured --eval-accuracy --eval-metric f1 \
            --n-ctx 16384 --n-chunks 20 $GPU_FLAGS $COMMON \
            --quants $QUANTS \
            --max-gen-tokens 4096 \
            --prompt-prefix "$LB_PREFIX" --prompt-suffix "$LB_SUFFIX" \
            "${LB_STOP_ARGS[@]}" \
            --out "$OUTDIR/longbench_qasper${FSUFFIX}.json"

        diagnose_alarms "$OUTDIR/longbench_qasper${FSUFFIX}_per_example.json" "LongBench $MODEL"
    fi

    # ── 4–6. NIAH (all sizes) ─────────────────────────────────────────────────
    # Metric: exact match with regex extracting the compound passphrase word
    # (e.g. "goldennexus" — all-lowercase, 6-20 chars).
    # F1 gives misleading partial credit when the model wraps the answer in
    # extra text like "The passphrase is X." — exact + last-match regex fixes this.
    # prefix/suffix: auto-detected from vocab; gpt-oss also gets --effort for system message.
    NIAH_EFFORT_ARGS=()
    [[ "$MODEL" == "gpt-oss-20b" ]] && NIAH_EFFORT_ARGS=(--effort "$GPT_OSS_EFFORT_NIAH")
    for niah_size in "4k:4096:niah_4096.jsonl:8192" \
                     "16k:16K:niah_16k.jsonl:24576" \
                     "32k:32K:niah_32k.jsonl:49152" \
                     "128k:128K:niah_128k.jsonl:180224"; do
        IFS=: read -r bench_key data_key data_file n_ctx_val <<< "$niah_size"
        bench_name="niah-$bench_key"
        data_path="research/data/$data_file"

        # \n stop: auto-added by task rule (niah keyword).
        if should_run "$bench_name" && [[ -f "$data_path" ]]; then
            run_benchmark "NIAH ${data_key} — $MODEL" \
                $SWEEP "$MODEL_FILE" "$data_path" \
                --corpus-mode structured --eval-accuracy --eval-metric exact \
                --answer-regex '([a-z]{6,20})' \
                --n-ctx "$n_ctx_val" --n-chunks 0 $GPU_FLAGS $COMMON \
                --quants $QUANTS \
                --max-gen-tokens 2048 \
                "${NIAH_EFFORT_ARGS[@]}" \
                --out "$OUTDIR/niah_${bench_key}${FSUFFIX}.json"
        fi
    done

    # ── 7. HumanEval Code ─────────────────────────────────────────────────────
    # No chat format: HumanEval is a code completion task. The prompt is a
    # Python function signature + docstring; the model continues writing code.
    # Chat format would shift the model into conversation mode — wrong for this.
    # --prompt-prefix "" suppresses auto-detection.
    # Stop strings: auto-added by task rule (humaneval keyword).
    if should_run humaneval && [[ -f research/data/humaneval.jsonl ]]; then
        run_benchmark "HumanEval — $MODEL" \
            $SWEEP "$MODEL_FILE" research/data/humaneval.jsonl \
            --corpus-mode structured --eval-accuracy --eval-metric code \
            --n-ctx 4096 --n-chunks 20 $GPU_FLAGS $COMMON \
            --quants $QUANTS \
            --max-gen-tokens 2048 \
            --prompt-prefix "" --prompt-suffix "" \
            --out "$OUTDIR/humaneval${FSUFFIX}.json"

        log "HumanEval pass@1 — $MODEL"
        $EVAL_CODE "$OUTDIR/humaneval${FSUFFIX}_per_example.json" \
            --jsonl research/data/humaneval.jsonl \
            --out "$OUTDIR/humaneval_pass1${FSUFFIX}.json" \
            2>&1 | tee -a "$LOG_FILE" || true
    fi

    # ── 8. AIME ───────────────────────────────────────────────────────────────
    # 0-shot: reasoning models need no format demonstration.
    # Data: research/data/aime_2022_2024.jsonl (73 problems, AIME 2022–2024).
    # Regenerate: python3 research/scripts/fetch_aime.py \
    #     --out research/data/aime_2022_2024.jsonl --year-min 2022
    # prefix/suffix: auto-detected for non-gpt-oss; \n\nQuestion: stop: auto-added by task rule.
    # prefix/suffix: auto-detected from vocab; gpt-oss also gets --effort for system message.
    AIME_EFFORT_ARGS=()
    [[ "$MODEL" == "gpt-oss-20b" ]] && AIME_EFFORT_ARGS=(--effort "$GPT_OSS_EFFORT_AIME")
    if should_run aime && [[ -f research/data/aime_2022_2024.jsonl ]]; then
        run_benchmark "AIME — $MODEL" \
            $SWEEP "$MODEL_FILE" research/data/aime_2022_2024.jsonl \
            --corpus-mode structured --eval-accuracy \
            --answer-regex 'boxed{(\d+)}' \
            --n-ctx 32768 --n-chunks 20 $GPU_FLAGS $COMMON \
            --quants $QUANTS \
            --max-gen-tokens 32000 \
            "${AIME_EFFORT_ARGS[@]}" \
            --out "$OUTDIR/aime${FSUFFIX}.json"
    elif should_run aime && [[ -f research/data/aime_2024.jsonl ]]; then
        # Fallback to 2024-only if 2022-2024 dataset not yet generated
        run_benchmark "AIME 2024 — $MODEL" \
            $SWEEP "$MODEL_FILE" research/data/aime_2024.jsonl \
            --corpus-mode structured --eval-accuracy \
            --answer-regex 'boxed{(\d+)}' \
            --n-ctx 32768 --n-chunks 20 $GPU_FLAGS $COMMON \
            --quants $QUANTS \
            --max-gen-tokens 32000 \
            "${AIME_EFFORT_ARGS[@]}" \
            --out "$OUTDIR/aime_2024${FSUFFIX}.json"
    fi

    echo
    echo "✓ $MODEL done — results in $OUTDIR/"
    echo "  Log: $LOG_FILE"
    echo
done

echo "All models complete."
