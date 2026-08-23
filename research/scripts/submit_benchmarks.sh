#!/usr/bin/env bash
# submit_benchmarks.sh — Submit KV cache benchmark suite as individual SLURM jobs.
#
# One job per (model × benchmark). Each job runs within the 2-hour acc_debug limit.
# Results saved to: research/results/{model_name}/
#
# Usage:
#   bash research/scripts/submit_benchmarks.sh --account bsc93
#   bash research/scripts/submit_benchmarks.sh --account bsc93 --models qwen3-8b gpt-oss-20b
#   bash research/scripts/submit_benchmarks.sh --account bsc93 --benchmarks gsm8k aime
#   bash research/scripts/submit_benchmarks.sh --account bsc93 --skip niah-128k
#   bash research/scripts/submit_benchmarks.sh --account bsc93 --effort high
#   bash research/scripts/submit_benchmarks.sh --account bsc93 --dry-run
#
# Available benchmarks:
#   wikitext2   WikiText2 perplexity (PPL at 512/1024/2048)
#   gsm8k       Math reasoning accuracy (8-shot)
#   longbench   Long-document QA (qasper F1)
#   niah-4k     Needle-in-a-haystack at 4K context
#   niah-16k    Needle-in-a-haystack at 16K context
#   niah-32k    Needle-in-a-haystack at 32K context
#   niah-128k   Needle-in-a-haystack at 128K context
#   humaneval   Code completion pass@1
#   aime        AIME math competition (0-shot)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/research/scripts"
DATA_DIR="$REPO_ROOT/research/data"

# ── Model registry ─────────────────────────────────────────────────────────────
declare -A MODEL_PATH=(
    # Q8_0 / high-precision baselines
    [qwen3-8b]="models/Qwen3-8B-Q8_0.gguf"
    [qwen3-8b-q4km]="models/Qwen3-8B-Q8_0-Q4_K_M.gguf"
    [qwen3-14b]="models/Qwen3-14B-Q8_0.gguf"
    [qwen3-14b-q4km]="models/Qwen3-14B-Q8_0-Q4_K_M.gguf"
    [qwen3-27b]="models/Qwen3.5-27B-Q8_0.gguf"
    [qwen3-27b-q4km]="models/Qwen3.5-27B-Q8_0-Q4_K_M.gguf"
    [gpt-oss-20b]="models/gpt-oss-20b-Q8_0.gguf"
    [gpt-oss-20b-q4km]="models/gpt-oss-20b-Q8_0-Q4_K_M.gguf"
    [llama4-scout-17b]="models/Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf"
    [llama4-scout-6e]="models/Llama-4-Scout-17B-6E-Instruct.Q6_K.gguf"
    [llama4-scout-6e-q4km]="models/Llama-4-Scout-17B-6E-Instruct.Q6_K-Q4_K_M.gguf"
    # Q4_K_RES recovered: dequant(base_Q4_K)+dequant(residual_Q4_K) re-quant to Q8_0
    # Built with: python3 research/scripts/build_q8_recovered.py models/<name>_q4k_res.gguf
    [qwen3-8b-q8rec]="models/Qwen3-8B-Q8_0-Q4_K_M_recovered.gguf"
    [qwen3-14b-q8rec]="models/Qwen3-14B-Q8_0-Q4_K_M_recovered.gguf"
    [qwen3-27b-q8rec]="models/Qwen3.5-27B-Q8_0-Q4_K_M_recovered.gguf"
    [gpt-oss-20b-q8rec]="models/gpt-oss-20b-Q8_0-Q4_K_M_recovered.gguf"
    [llama4-scout-6e-q8rec]="models/Llama-4-Scout-17B-6E-Instruct.Q6_K-Q4_K_M_recovered.gguf"
)

# LongBench only — needs explicit "Answer briefly" system prompt.
# GSM8K, NIAH, AIME, HumanEval use auto-detection (no explicit prefix needed).
_QWEN_CHATML_LB_PFX=$'<|im_start|>system\nAnswer briefly in 1-2 sentences based only on the provided text.<|im_end|>\n<|im_start|>user\n'
_QWEN_CHATML_LB_SFX=$'<|im_end|>\n<|im_start|>assistant\n'
_LLAMA4_LB_PFX=$'<|header_start|>system<|header_end|>\nAnswer briefly in 1-2 sentences based only on the provided text.<|eot|>\n<|header_start|>user<|header_end|>\n'
_LLAMA4_LB_SFX=$'<|eot|>\n<|header_start|>assistant<|header_end|>\n'

declare -A MODEL_LONGBENCH_PREFIX=(
    [qwen3-8b]="$_QWEN_CHATML_LB_PFX"
    [qwen3-8b-q4km]="$_QWEN_CHATML_LB_PFX"
    [qwen3-14b]="$_QWEN_CHATML_LB_PFX"
    [qwen3-14b-q4km]="$_QWEN_CHATML_LB_PFX"
    [qwen3-27b]="$_QWEN_CHATML_LB_PFX"
    [qwen3-27b-q4km]="$_QWEN_CHATML_LB_PFX"
    [llama4-scout-17b]="$_LLAMA4_LB_PFX"
    [llama4-scout-6e]="$_LLAMA4_LB_PFX"
    [llama4-scout-6e-q4km]="$_LLAMA4_LB_PFX"
    # gpt-oss-20b: built dynamically via _gpt_oss_lb_prefix $GPT_OSS_EFFORT_LB below
)
declare -A MODEL_LONGBENCH_SUFFIX=(
    [qwen3-8b]="$_QWEN_CHATML_LB_SFX"
    [qwen3-8b-q4km]="$_QWEN_CHATML_LB_SFX"
    [qwen3-14b]="$_QWEN_CHATML_LB_SFX"
    [qwen3-14b-q4km]="$_QWEN_CHATML_LB_SFX"
    [qwen3-27b]="$_QWEN_CHATML_LB_SFX"
    [qwen3-27b-q4km]="$_QWEN_CHATML_LB_SFX"
    [llama4-scout-17b]="$_LLAMA4_LB_SFX"
    [llama4-scout-6e]="$_LLAMA4_LB_SFX"
    [llama4-scout-6e-q4km]="$_LLAMA4_LB_SFX"
    [gpt-oss-20b]=$'<|end|>\n<|start|>assistant'
    [gpt-oss-20b-q4km]=$'<|end|>\n<|start|>assistant'
)
declare -A MODEL_LONGBENCH_STOP=(
    [qwen3-8b]="<|im_end|>"
    [qwen3-8b-q4km]="<|im_end|>"
    [qwen3-14b]="<|im_end|>"
    [qwen3-14b-q4km]="<|im_end|>"
    [qwen3-27b]="<|im_end|>"
    [qwen3-27b-q4km]="<|im_end|>"
    [llama4-scout-17b]="<|eot|>"
    [llama4-scout-6e]="<|eot|>"
    [llama4-scout-6e-q4km]="<|eot|>"
    # gpt-oss: no explicit stop — <|end|> is inter-channel separator (analysis→final),
    # not EOG. <|return|> (real EOG) is caught by llama_vocab_is_eog.
    [gpt-oss-20b]=""
    [gpt-oss-20b-q4km]=""
)

# ── GPT-OSS task-specific reasoning effort ────────────────────────────────────
# Defaults below; override all with --effort LEVEL.
GPT_OSS_EFFORT_AIME="high"     # hard math competition — full CoT needed
GPT_OSS_EFFORT_GSM8K="medium"  # grade school math + 8-shot examples
GPT_OSS_EFFORT_NIAH="low"      # pure retrieval — long CoT loses focus
GPT_OSS_EFFORT_LB="low"        # retrieval QA — "Answer briefly" already set

# LongBench prefix includes task content, so built explicitly rather than via --effort.
_gpt_oss_lb_prefix() {
    printf '<|start|>system<|message|>Reasoning: %s\n\nAnswer briefly in 1-2 sentences based only on the provided text.\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\n<|start|>user<|message|>' "$1"
}

# ── Defaults ───────────────────────────────────────────────────────────────────
ALL_BENCHMARKS=(wikitext2 gsm8k longbench niah-4k humaneval aime)
# niah-16k/32k/128k excluded by default: lost-in-middle is already severe at 4k.
# Re-enable with: --benchmarks niah-16k niah-32k niah-128k
RUN_MODELS=(qwen3-8b qwen3-8b-q4km qwen3-14b qwen3-14b-q4km qwen3-27b qwen3-27b-q4km gpt-oss-20b gpt-oss-20b-q4km llama4-scout-6e llama4-scout-6e-q4km)
RUN_BENCHMARKS=("${ALL_BENCHMARKS[@]}")
QUANTS="fp16 int8_ch int4_ch int3_ch int3_half_1357_ch int3_half_1357_ch:int3_half_1357_tok int2_ch"
COMMON_FLAGS="--asym --sink-tokens 32 --recent-tokens 32 --flash-attn --show-prompt --show-gen"
OUT_SUFFIX=""          # auto-derived from --quants when set; override with --out-suffix
SLURM_ACCOUNT=""
SLURM_QOS="acc_bsccs"     # acc_bsccs = project QOS; acc_debug = 2h limit for quick tests
SLURM_TIME="01:55:00"     # just under 2h
N_GPUS=1
N_CHUNKS=0                # --n-chunks: max examples per job (0 = all); appended to output filename when >0
DRY_RUN=0
CHAIN=0                   # --chain: each job waits for the previous via --dependency=afterany

# ── Parse CLI ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)       SLURM_ACCOUNT="$2"; shift 2 ;;
        --qos)           SLURM_QOS="$2";     shift 2 ;;
        --time)          SLURM_TIME="$2";    shift 2 ;;
        --quants)        QUANTS="$2"; shift 2 ;;
        --out-suffix)    OUT_SUFFIX="$2";    shift 2 ;;
        --n-gpus)        N_GPUS="$2";        shift 2 ;;
        --n-chunks)      N_CHUNKS="$2";      shift 2 ;;
        --dry-run)       DRY_RUN=1;          shift ;;
        --chain)         CHAIN=1;            shift ;;
        --effort)
            shift
            GPT_OSS_EFFORT_AIME="$1"
            GPT_OSS_EFFORT_GSM8K="$1"
            GPT_OSS_EFFORT_NIAH="$1"
            GPT_OSS_EFFORT_LB="$1"
            shift ;;
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
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$SLURM_ACCOUNT" ]]; then
    echo "ERROR: --account is required."
    echo "  Usage: bash submit_benchmarks.sh --account bsc93 [options]"
    exit 1
fi

N_CPUS=$(( N_GPUS * 20 ))   # MN5 rule: 20 CPUs per GPU
N_THREADS=$N_CPUS

# Effective chunk counts per benchmark type.
# When --n-chunks N is given, N overrides all benchmarks.
# When --n-chunks 0 (default = all), fixed-size benchmarks fall back to 20;
# open-ended ones (niah, aime) stay 0 (= all examples).
NC_FIXED=$(( N_CHUNKS > 0 ? N_CHUNKS : 20 ))  # wikitext2 / gsm8k / longbench / humaneval
NC_OPEN=${N_CHUNKS}                            # niah / aime  (0 = all)

# Build OUT_SUFFIX from the quant list (always, unless --out-suffix was given explicitly).
# Single quant  → short: "fp16", "int4_ch"
# Multiple quants → joined: "fp16_int8_ch_int4_ch"
# K:V split quants → colon replaced by dash: "int4_ch-int4_tok"
if [[ -z "$OUT_SUFFIX" ]]; then
    OUT_SUFFIX=$(echo "$QUANTS" | tr ' ' '_' | tr ':' '-')
fi
# Append chunk count when --n-chunks > 0.
FSUFFIX="_${OUT_SUFFIX}"
[[ $N_CHUNKS -gt 0 ]] && FSUFFIX="${FSUFFIX}_${N_CHUNKS}chunks"

echo "================================================================"
echo "  Models:     ${RUN_MODELS[*]}"
echo "  Benchmarks: ${RUN_BENCHMARKS[*]}"
echo "  Quants:     $QUANTS"
[[ -n "$FSUFFIX" ]] && echo "  Out suffix: $FSUFFIX"
echo "  Account:    $SLURM_ACCOUNT  QOS: $SLURM_QOS  Time: $SLURM_TIME"
echo "  GPT-OSS effort: aime=$GPT_OSS_EFFORT_AIME  gsm8k=$GPT_OSS_EFFORT_GSM8K  niah=$GPT_OSS_EFFORT_NIAH  lb=$GPT_OSS_EFFORT_LB"
echo "  N-chunks:   $N_CHUNKS (0=all; fixed-benchmarks default to 20)"
echo "  Dry-run:    $DRY_RUN  Chain: $CHAIN"
echo "================================================================"
echo

# ── MN5 module setup (embedded in each job) ────────────────────────────────────
MN5_SETUP=$(cat << 'SETUP_EOF'
source /apps/GPP/ANACONDA/2024.02/etc/profile.d/conda.sh
conda activate llama
# libstdc++ override: conda's version conflicts with llama.cpp shared libs
GCC_LIBSTDCXX="/apps/GPP/GCC/12.3.0/lib64/libstdc++.so.6"
if [[ -f "$GCC_LIBSTDCXX" ]]; then
    export LD_PRELOAD="$GCC_LIBSTDCXX${LD_PRELOAD:+:$LD_PRELOAD}"
fi
SETUP_EOF
)

# ── Submit one job ─────────────────────────────────────────────────────────────
# Stdout: the submitted job ID (so callers can capture it: jid=$(submit_job ...))
# Stderr: human-readable status lines
submit_job() {
    local model="$1"
    local benchmark="$2"
    local body="$3"        # the actual sweep command(s)
    local outdir="$4"

    local job_name="${model}-${benchmark}"
    local log_out="$outdir/${benchmark}${FSUFFIX}_%j.out"
    local log_err="$outdir/${benchmark}${FSUFFIX}_%j.err"

    local script
    script=$(mktemp /tmp/job_XXXXXX.sh)

    cat > "$script" << SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_out}
#SBATCH --error=${log_err}
#SBATCH --partition=acc
#SBATCH --qos=${SLURM_QOS}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=${SLURM_TIME}
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_CPUS}

set -euo pipefail
cd "${REPO_ROOT}"

echo "=== \${SLURM_JOB_NAME} job \${SLURM_JOB_ID} ==="
echo "Node: \$(hostname)  GPUs: \${CUDA_VISIBLE_DEVICES:-none}"

${MN5_SETUP}

${body}

echo "=== DONE ==="
SBATCH_EOF

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] $job_name" >&2
        echo "  ----" >&2
        grep -v '^#SBATCH\|^set -\|^cd \|^echo\|MN5_SETUP\|DONE' "$script" | grep -v '^\s*$' | head -20 >&2
        echo >&2
        echo "0"  # fake JID for dry-run
    else
        local jid
        jid=$(sbatch --parsable "$script")
        echo "  Submitted $job_name → job $jid" >&2
        echo "    Log: $outdir/${benchmark}${FSUFFIX}_${jid}.out" >&2
        echo "$jid"  # return JID to caller
    fi

    rm -f "$script"
}

should_run() { [[ " ${RUN_BENCHMARKS[*]} " == *" $1 "* ]]; }

# ── Chain mode helpers ────────────────────────────────────────────────────────
# wait_for_job: poll squeue every 60s until the job disappears (done or failed).
wait_for_job() {
    local jid="$1"
    echo "  [chain] waiting for job $jid to finish..." >&2
    while squeue -j "$jid" -h &>/dev/null; do
        sleep 60
    done
    echo "  [chain] job $jid done." >&2
}

# chain_submit: submit a job; when --chain, block until it finishes before returning.
# Usage:  chain_submit MODEL BENCH BODY OUTDIR
chain_submit() {
    local jid
    jid=$(submit_job "$1" "$2" "$3" "$4")
    if [[ $CHAIN -eq 1 && -n "$jid" ]]; then
        wait_for_job "$jid"
    fi
}

# ── Loop: model × benchmark ───────────────────────────────────────────────────
TOTAL=0

for MODEL in "${RUN_MODELS[@]}"; do
    MODEL_FILE="${MODEL_PATH[$MODEL]}"
    OUTDIR="$REPO_ROOT/research/results/$MODEL"
    mkdir -p "$OUTDIR"

    GPU_FLAGS="--n-gpu-layers 99 --n-threads ${N_THREADS}"

    # LongBench prefix/suffix/stop (explicit; auto-detect skipped when prefix set)
    LB_PREFIX="${MODEL_LONGBENCH_PREFIX[$MODEL]:-}"
    LB_SUFFIX="${MODEL_LONGBENCH_SUFFIX[$MODEL]:-}"
    LB_STOP="${MODEL_LONGBENCH_STOP[$MODEL]:-}"
    if [[ "$MODEL" == "gpt-oss-20b" ]]; then
        LB_PREFIX="$(_gpt_oss_lb_prefix "$GPT_OSS_EFFORT_LB")"
    fi

    # gpt-oss effort flags (passed to run_sweep.py --effort; auto-detect injects system msg)
    EFFORT_GSM8K=""; [[ "$MODEL" == gpt-oss-20b* ]] && EFFORT_GSM8K="--effort $GPT_OSS_EFFORT_GSM8K"
    EFFORT_NIAH="";  [[ "$MODEL" == gpt-oss-20b* ]] && EFFORT_NIAH="--effort $GPT_OSS_EFFORT_NIAH"
    EFFORT_AIME="";  [[ "$MODEL" == gpt-oss-20b* ]] && EFFORT_AIME="--effort $GPT_OSS_EFFORT_AIME"

    # gpt-oss LongBench stop: <|end|> is inter-channel separator, not EOG — omit it
    if [[ "$MODEL" == gpt-oss-20b* || -z "$LB_STOP" ]]; then
        LB_STOP_FLAGS=""
    else
        LB_STOP_FLAGS="--stop-strings '${LB_STOP}' --stop-strings $'\\nHuman:'"
    fi

    echo "── $MODEL ──────────────────────────────────────"

    # 1. WikiText2
    if should_run wikitext2; then
        chain_submit "$MODEL" "wikitext2" \
"python3 ${SCRIPTS_DIR}/run_sweep.py '${MODEL_FILE}' ${DATA_DIR}/wikitext2_test.txt \\
    --prefill-tokens 1024 --score-windows 512 1024 2048 \\
    --n-chunks ${NC_FIXED} ${GPU_FLAGS} ${COMMON_FLAGS} \\
    --quants ${QUANTS} \\
    --out '${OUTDIR}/wikitext2${FSUFFIX}.json'" \
            "$OUTDIR"
        TOTAL=$((TOTAL+1))
    fi

    # 2. GSM8K — prefix/suffix auto-detected; gpt-oss gets --effort
    if should_run gsm8k; then
        chain_submit "$MODEL" "gsm8k" \
"python3 ${SCRIPTS_DIR}/run_sweep.py '${MODEL_FILE}' ${DATA_DIR}/gsm8k_test.jsonl \\
    --corpus-mode structured --eval-accuracy \\
    --n-ctx 8192 --n-chunks ${NC_FIXED} ${GPU_FLAGS} ${COMMON_FLAGS} \\
    --quants ${QUANTS} \\
    --max-gen-tokens 4096 ${EFFORT_GSM8K} \\
    --out '${OUTDIR}/gsm8k${FSUFFIX}.json'

python3 ${SCRIPTS_DIR}/analyze_alarms.py '${OUTDIR}/gsm8k${FSUFFIX}_per_example.json' \\
    --summary 2>/dev/null || true
python3 ${SCRIPTS_DIR}/analyze_alarms.py '${OUTDIR}/gsm8k${FSUFFIX}_per_example.json' \\
    2>/dev/null || true" \
            "$OUTDIR"
        TOTAL=$((TOTAL+1))
    fi

    # 3. LongBench — explicit prefix with "Answer briefly"; gpt-oss stop via EOG only
    if should_run longbench && [[ -f "${DATA_DIR}/LongBench/qasper.jsonl" ]]; then
        chain_submit "$MODEL" "longbench" \
"python3 ${SCRIPTS_DIR}/run_sweep.py '${MODEL_FILE}' ${DATA_DIR}/LongBench/qasper.jsonl \\
    --corpus-mode structured --eval-accuracy --eval-metric f1 \\
    --n-ctx 16384 --n-chunks ${NC_FIXED} ${GPU_FLAGS} ${COMMON_FLAGS} \\
    --quants ${QUANTS} \\
    --max-gen-tokens 4096 \\
    --prompt-prefix '${LB_PREFIX}' --prompt-suffix '${LB_SUFFIX}' \\
    ${LB_STOP_FLAGS} \\
    --out '${OUTDIR}/longbench_qasper${FSUFFIX}.json'

python3 ${SCRIPTS_DIR}/analyze_alarms.py '${OUTDIR}/longbench_qasper${FSUFFIX}_per_example.json' \\
    --summary 2>/dev/null || true
python3 ${SCRIPTS_DIR}/analyze_alarms.py '${OUTDIR}/longbench_qasper${FSUFFIX}_per_example.json' \\
    2>/dev/null || true" \
            "$OUTDIR"
        TOTAL=$((TOTAL+1))
    fi

    # 4–6. NIAH — exact match metric; prefix/suffix auto-detected; gpt-oss gets --effort
    for niah_size in "niah-4k:niah_4096.jsonl:8192" \
                     "niah-16k:niah_16k.jsonl:24576" \
                     "niah-32k:niah_32k.jsonl:49152" \
                     "niah-128k:niah_128k.jsonl:180224"; do
        IFS=: read -r bench_name data_file n_ctx_val <<< "$niah_size"
        data_path="${DATA_DIR}/${data_file}"
        out_key="${bench_name/niah-/niah_}"

        if should_run "$bench_name" && [[ -f "$data_path" ]]; then
            chain_submit "$MODEL" "$bench_name" \
"python3 ${SCRIPTS_DIR}/run_sweep.py '${MODEL_FILE}' ${data_path} \\
    --corpus-mode structured --eval-accuracy --eval-metric exact \\
    --answer-regex '([a-z]{6,20})' \\
    --n-ctx ${n_ctx_val} --n-chunks ${NC_OPEN} ${GPU_FLAGS} ${COMMON_FLAGS} \\
    --quants ${QUANTS} \\
    --max-gen-tokens 2048 ${EFFORT_NIAH} \\
    --out '${OUTDIR}/${out_key}${FSUFFIX}.json'" \
                "$OUTDIR"
            TOTAL=$((TOTAL+1))
        fi
    done

    # 7. HumanEval — code completion; suppress chat format with empty prefix/suffix
    if should_run humaneval && [[ -f "${DATA_DIR}/humaneval.jsonl" ]]; then
        chain_submit "$MODEL" "humaneval" \
"python3 ${SCRIPTS_DIR}/run_sweep.py '${MODEL_FILE}' ${DATA_DIR}/humaneval.jsonl \\
    --corpus-mode structured --eval-accuracy --eval-metric code \\
    --n-ctx 4096 --n-chunks ${NC_FIXED} ${GPU_FLAGS} ${COMMON_FLAGS} \\
    --quants ${QUANTS} \\
    --max-gen-tokens 2048 \\
    --prompt-prefix '' --prompt-suffix '' \\
    --out '${OUTDIR}/humaneval${FSUFFIX}.json'

python3 ${SCRIPTS_DIR}/eval_code.py '${OUTDIR}/humaneval${FSUFFIX}_per_example.json' \\
    --jsonl ${DATA_DIR}/humaneval.jsonl \\
    --out '${OUTDIR}/humaneval_pass1${FSUFFIX}.json' || true" \
            "$OUTDIR"
        TOTAL=$((TOTAL+1))
    fi

    # 8. AIME — 0-shot; prefix/suffix auto-detected; gpt-oss gets --effort high
    # n-ctx=69632 = ~4K prompt + 64K gen budget.
    # rep-rate-threshold=0.8: 3-gram rate > 80% for 8 steps.
    # rep-period-stop: also catches longer spirals (period 8–160, 3 repeats, 92% similarity).
    # Use aime_2022_2024.jsonl (73 examples) if available, else aime_2024.jsonl (14 examples).
    for aime_file in "aime_2022_2024.jsonl:aime" "aime_2024.jsonl:aime_2024"; do
        IFS=: read -r aime_data aime_out <<< "$aime_file"
        if should_run aime && [[ -f "${DATA_DIR}/${aime_data}" ]]; then
            chain_submit "$MODEL" "aime" \
"python3 ${SCRIPTS_DIR}/run_sweep.py '${MODEL_FILE}' ${DATA_DIR}/${aime_data} \\
    --corpus-mode structured --eval-accuracy --skip-ppl \\
    --n-ctx 69632 --n-chunks ${NC_OPEN} ${GPU_FLAGS} ${COMMON_FLAGS} \\
    --quants ${QUANTS} \\
    --max-gen-tokens 65536 --rep-rate-threshold 0.8 --rep-rate-consecutive 8 \\
    --rep-period-stop --rep-period-min 8 --rep-period-max 160 \\
    --rep-period-repeats 3 --rep-period-similarity 0.92 --rep-min-tokens 256 \\
    ${EFFORT_AIME} \\
    --out '${OUTDIR}/${aime_out}${FSUFFIX}.json'" \
                "$OUTDIR"
            TOTAL=$((TOTAL+1))
            break   # use first dataset found (prefer 2022-2024)
        fi
    done

    echo
done

echo "================================================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] Would submit $TOTAL jobs"
else
    echo "  Submitted $TOTAL jobs"
    echo "  Monitor: squeue -u \$USER"
    echo "  Results: research/results/{model}/"
fi
echo "================================================================"
