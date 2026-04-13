#!/usr/bin/env bash
# submit_kv_timing.sh — Submit SLURM jobs to run benchmark_kv_timing.py (native KV types).
#
# One SLURM job per registry key in RUN_MODELS. Each job runs benchmark_kv_timing.py
# once with that GGUF (KV types + batch sizes are swept inside that single process).
#
# Registry keys → GGUF paths + perf_model preset (see MODEL_PATH / MODEL_PRESET).
# Default: --suite qwen3-8b → four jobs (Q8_0 + Q2_K + Q3_K_M + Q4_K_M for that family).
# Logs + JSON: research/results/{model_key}/profile/
#
# Usage:
#   bash research/scripts/submit_kv_timing.sh --account bsc93
#   bash research/scripts/submit_kv_timing.sh --account bsc93 --suite qwen3-14b
#   bash research/scripts/submit_kv_timing.sh --account bsc93 --suite qwen3-both
#   bash research/scripts/submit_kv_timing.sh --account bsc93 --suite bases
#   bash research/scripts/submit_kv_timing.sh --account bsc93 --models qwen3-8b llama4-scout-17b
#   bash research/scripts/submit_kv_timing.sh --account bsc93 --dry-run
#
#   bash research/scripts/submit_kv_timing.sh --help
#
# Override workload / lib (after defaults):
#   bash research/scripts/submit_kv_timing.sh --account bsc93 \
#       --extra "--prompt-lens 4096 --decode-len 512 2048 4096 --decode-bucket-size 64 --prefill-bucket-size 64"

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/research/scripts"

# ── Model registry (same paths as submit_benchmarks.sh; adjust on your cluster) ─
# NOTE: Do not put `#` at the start of a [key]= line inside declare -A — bash treats the
# whole line as a comment, so that key is never defined (you will see SKIP: no MODEL_PATH).

declare -A MODEL_PATH=(
    [qwen3-8b]="models/Qwen3-8B-Q8_0.gguf"
    [qwen3-14b]="models/Qwen3-14B-Q8_0.gguf"
    [llama4-scout-17b]="models/Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf"
    [gpt-oss-20b]="models/gpt-oss-20b-Q8_0.gguf"
    [qwen3-8b-q2k]="models/Qwen3-8B-Q8_0-Q2_K.gguf"
    [qwen3-8b-q3km]="models/Qwen3-8B-Q8_0-Q3_K_M.gguf"
    [qwen3-8b-q4km]="models/Qwen3-8B-Q8_0-Q4_K_M.gguf"
    [qwen3-14b-q2k]="models/Qwen3-14B-Q8_0-Q2_K.gguf"
    [qwen3-14b-q3km]="models/Qwen3-14B-Q8_0-Q3_K_M.gguf"
    [qwen3-14b-q4km]="models/Qwen3-14B-Q8_0-Q4_K_M.gguf"
)

declare -A MODEL_PRESET=(
    [qwen3-8b]="qwen3-8b"
    [qwen3-14b]="qwen3-14b"
    [llama4-scout-17b]="llama4-scout-17b"
    [gpt-oss-20b]="gpt-oss-20b"
    [qwen3-8b-q2k]="qwen3-8b"
    [qwen3-8b-q3km]="qwen3-8b"
    [qwen3-8b-q4km]="qwen3-8b"
    [qwen3-14b-q2k]="qwen3-14b"
    [qwen3-14b-q3km]="qwen3-14b"
    [qwen3-14b-q4km]="qwen3-14b"
)

# Subfolder under research/results/{model}/ for logs and JSON (override: --profile-dir)
PROFILE_SUBDIR="profile"

# Filled after parsing (--suite or explicit --models). Empty until apply_suite_or_models runs.
RUN_MODELS=()
MODELS_EXPLICIT=0
SUITE="qwen3-8b"
SLURM_ACCOUNT=""
SLURM_QOS="acc_debug"
SLURM_TIME="01:55:00"
N_GPUS=1
DRY_RUN=0
HW_PRESET="h100"
# Default workload: long decode + bucketed prefill/decode stats (space-separated = one benchmark sweep)
PROMPT_LENS="2048"
DECODE_LENS="4096"
BATCH_SIZES="1 4"
KV_TYPES="f16 q8_0 q4_0"
N_GPU_LAYERS=99
FLASH_ATTN="--flash-attn"
DECODE_BUCKET_SIZE=64
PREFILL_BUCKET_SIZE=64
EXTRA_FLAGS=""
LIB_PATH=""   # empty → benchmark default (build_release/bin/libllama.so)

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)    SLURM_ACCOUNT="$2"; shift 2 ;;
        --qos)        SLURM_QOS="$2";     shift 2 ;;
        --time)       SLURM_TIME="$2";    shift 2 ;;
        --n-gpus)     N_GPUS="$2";        shift 2 ;;
        --hw)         HW_PRESET="$2";     shift 2 ;;
        --prompt-lens) PROMPT_LENS="$2";  shift 2 ;;
        --decode-len)
            shift
            DECODE_LENS=""
            while [[ $# -gt 0 && $1 != --* ]]; do DECODE_LENS="$DECODE_LENS $1"; shift; done
            DECODE_LENS="${DECODE_LENS# }"
            ;;
        --batch-sizes) BATCH_SIZES="$2";  shift 2 ;;
        --kv-types)    KV_TYPES="$2";     shift 2 ;;
        --decode-bucket-size)  DECODE_BUCKET_SIZE="$2";  shift 2 ;;
        --prefill-bucket-size) PREFILL_BUCKET_SIZE="$2"; shift 2 ;;
        --lib)        LIB_PATH="$2";      shift 2 ;;
        --extra)      EXTRA_FLAGS="$2";   shift 2 ;;
        --profile-dir) PROFILE_SUBDIR="$2"; shift 2 ;;
        --suite)      SUITE="$2";         shift 2 ;;
        --dry-run)    DRY_RUN=1;          shift ;;
        --help|-h)
            cat << 'HELP_EOF'
submit_kv_timing.sh — SLURM jobs for benchmark_kv_timing.py (native KV cache types).

Required: --account NAME

Model selection (first match wins after parsing):
  --suite NAME     preset group of registry keys (default: qwen3-8b)
  --models K K ... explicit registry keys (overrides --suite)

  --decode-len N [N ...]   timed decode length(s); pass multiple to sweep (default: 4096)

  --suite values:
    qwen3-8b       Qwen3-8B  Q8_0 + Q2_K + Q3_K_M + Q4_K_M  (4 jobs)
    qwen3-14b      Qwen3-14B same four weight GGUFs          (4 jobs)
    qwen3-both     all eight Qwen3 8B + 14B K-quant jobs
    bases          one job each: qwen3-8b, qwen3-14b, llama4-scout-17b, gpt-oss-20b (base GGUFs)

  Add new models: extend MODEL_PATH and MODEL_PRESET above, then pass the key via --models.

Other options: --hw, --prompt-lens, --decode-len, --batch-sizes, --kv-types,
  --decode-bucket-size, --prefill-bucket-size, --lib, --extra, --profile-dir,
  --qos, --time, --n-gpus, --dry-run
HELP_EOF
            exit 0 ;;
        --models)
            shift; RUN_MODELS=(); MODELS_EXPLICIT=1
            while [[ $# -gt 0 && $1 != --* ]]; do RUN_MODELS+=("$1"); shift; done ;;
        *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

if [[ $MODELS_EXPLICIT -eq 0 ]]; then
    case $SUITE in
        qwen3-8b)
            RUN_MODELS=(qwen3-8b qwen3-8b-q2k qwen3-8b-q3km qwen3-8b-q4km) ;;
        qwen3-14b)
            RUN_MODELS=(qwen3-14b qwen3-14b-q2k qwen3-14b-q3km qwen3-14b-q4km) ;;
        qwen3-both|qwen3-all)
            RUN_MODELS=(
                qwen3-8b qwen3-8b-q2k qwen3-8b-q3km qwen3-8b-q4km
                qwen3-14b qwen3-14b-q2k qwen3-14b-q3km qwen3-14b-q4km
            ) ;;
        bases)
            RUN_MODELS=(qwen3-8b qwen3-14b llama4-scout-17b gpt-oss-20b) ;;
        *)
            echo "ERROR: unknown --suite '$SUITE' (try: qwen3-8b, qwen3-14b, qwen3-both, bases)"
            exit 1 ;;
    esac
fi

if [[ ${#RUN_MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no models selected (use --suite or --models)."
    exit 1
fi

if [[ -z "$SLURM_ACCOUNT" ]]; then
    echo "ERROR: --account is required."
    echo "  Usage: bash research/scripts/submit_kv_timing.sh --account YOUR_ACCOUNT [options]"
    exit 1
fi

N_CPUS=$(( N_GPUS * 20 ))
N_THREADS=$N_CPUS

echo "================================================================"
if [[ $MODELS_EXPLICIT -eq 1 ]]; then
    echo "  Selection:  explicit --models"
else
    echo "  Suite:      $SUITE"
fi
echo "  Models:     ${RUN_MODELS[*]}"
echo "  HW preset:  $HW_PRESET"
echo "  Prompt:     $PROMPT_LENS   decode_lens: $DECODE_LENS   batches: $BATCH_SIZES"
echo "  KV types:   $KV_TYPES"
echo "  Buckets:    prefill=$PREFILL_BUCKET_SIZE  decode=$DECODE_BUCKET_SIZE"
echo "  Account:    $SLURM_ACCOUNT  QOS: $SLURM_QOS  Time: $SLURM_TIME"
echo "  Out dir:    research/results/{model}/$PROFILE_SUBDIR/"
echo "  Dry-run:    $DRY_RUN"
[[ -n "$EXTRA_FLAGS" ]] && echo "  Extra:      $EXTRA_FLAGS"
echo "================================================================"
echo

# MN5-style env (comment out or replace on non-BSC systems)
MN5_SETUP=$(cat << 'SETUP_EOF'
source /apps/GPP/ANACONDA/2024.02/etc/profile.d/conda.sh
conda activate llama
GCC_LIBSTDCXX="/apps/GPP/GCC/12.3.0/lib64/libstdc++.so.6"
if [[ -f "$GCC_LIBSTDCXX" ]]; then
    export LD_PRELOAD="$GCC_LIBSTDCXX${LD_PRELOAD:+:$LD_PRELOAD}"
fi
SETUP_EOF
)

submit_kv_job() {
    local model="$1"
    local model_file="${MODEL_PATH[$model]:-}"
    local preset="${MODEL_PRESET[$model]:-$model}"

    if [[ -z "$model_file" ]]; then
        echo "  SKIP: no MODEL_PATH for '$model' — add [${model}]=\"models/....gguf\" in MODEL_PATH, or use --models to select keys that exist."
        return 1
    fi

    local outdir="$REPO_ROOT/research/results/$model/${PROFILE_SUBDIR}"
    mkdir -p "$outdir"

    local job_name="kv-${model}"
    local log_out="$outdir/kv_timing_%j.out"
    local log_err="$outdir/kv_timing_%j.err"

    local gguf_path="$REPO_ROOT/$model_file"
    local json_out="$outdir/kv_timing_${HW_PRESET}.json"
    local LIB_ARG=""
    [[ -n "$LIB_PATH" ]] && LIB_ARG="--lib '${LIB_PATH}'"

    local script
    script=$(mktemp /tmp/kv_timing_XXXXXX.sh)

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

python3 ${SCRIPTS_DIR}/benchmark_kv_timing.py '${gguf_path}' --hw ${HW_PRESET} --model-preset ${preset} --n-gpu-layers ${N_GPU_LAYERS} --n-threads ${N_THREADS} ${FLASH_ATTN} --prompt-lens ${PROMPT_LENS} --decode-len ${DECODE_LENS} --batch-sizes ${BATCH_SIZES} --kv-types ${KV_TYPES} --decode-bucket-size ${DECODE_BUCKET_SIZE} --prefill-bucket-size ${PREFILL_BUCKET_SIZE} --out '${json_out}' ${LIB_ARG} ${EXTRA_FLAGS}

echo "=== DONE ==="
SBATCH_EOF

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] $job_name"
        grep '^python3' "$script" || true
        echo
    else
        local jid
        jid=$(sbatch --parsable "$script")
        echo "  Submitted $job_name → job $jid"
        echo "    Log: $outdir/kv_timing_${jid}.out"
    fi

    rm -f "$script"
    return 0
}

TOTAL=0
for MODEL in "${RUN_MODELS[@]}"; do
    echo "── $MODEL ──────────────────────────────────────"
    if submit_kv_job "$MODEL"; then
        TOTAL=$((TOTAL + 1))
    fi
    echo
done

echo "================================================================"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] Would submit $TOTAL job(s)"
else
    echo "  Submitted $TOTAL job(s)"
    echo "  Monitor: squeue -u \$USER"
    echo "  Results: research/results/{model}/${PROFILE_SUBDIR}/kv_timing_*.json"
fi
echo "================================================================"
