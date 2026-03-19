#!/usr/bin/env bash
# submit_sweep.sh — Submit a KV-cache quantization sweep as SLURM job array.
#
# Tuned for MareNostrum 5 (BSC) ACC partition:
#   - Partition: acc
#   - GPUs: H100 (4 per node)
#   - Rule: 20 CPUs required per GPU
#   - Mandatory: --account AND --qos
#
# Each array job = one (model, prefill_tokens) combination.
# All quant types run sequentially within one job (shared model load).
# n_ctx is set automatically to prefill_tokens + max(score_windows).
#
# Usage:
#   bash research/scripts/submit_sweep.sh --account <your_account> [options]
#
# Options:
#   --account         SLURM account (REQUIRED on MN5)
#   --qos             acc_debug (2h), acc_bscls (2d), acc_ehpc (3d) (default: acc_bscls)
#   --models          Space-separated model paths (default: Qwen3-8B-Q8_0.gguf)
#   --prefill-tokens  Space-separated prefill sizes — one job per value
#                     (default: "1024 4096 16384")
#                     n_ctx is set to prefill + max(score_windows) per job.
#   --score-windows   Space-separated decode lengths (default: "512 1024 2048")
#                     Decode max(windows) tokens once; compute PPL at each cutoff.
#   --quants          Space-separated quant names (default: fp16 int8_ch int4_ch)
#   --corpus          Path to corpus file (.txt or .jsonl)
#   --corpus-mode     flat | structured (default: flat)
#   --n-chunks        Chunks per job (default: 5); strided evenly across corpus
#   --n-ctx           Override n_ctx for structured mode (default: 0 = auto)
#   --quant-group-size  Tokens per quantization group (default: 128)
#   --n-gpus          GPUs per job, 1-4 (default: 1)
#   --time            SLURM time limit HH:MM:SS (default: 08:00:00)
#   --dry-run         Print jobs without submitting
#
# Examples:
#
#   # Prefill sweep: short/medium/long context, score at 512/1024/2048
#   bash research/scripts/submit_sweep.sh \
#       --account bsc93 --qos acc_bscls \
#       --prefill-tokens "1024 4096 16384 32768" \
#       --score-windows "512 1024 2048" \
#       --quants "fp16 int8_ch int4_ch:int4_tok int3_ch int3_ch:int3_tok int2_ch int2_ch:int2_tok" \
#       --corpus research/data/wikitext2_test.txt \
#       --n-chunks 5 --time 08:00:00
#
#   # Quick smoke test (debug queue)
#   bash research/scripts/submit_sweep.sh \
#       --account bsc93 --qos acc_debug \
#       --prefill-tokens "1024 4096" \
#       --score-windows "512 1024 2048" \
#       --quants "fp16 int8_ch int4_ch:int4_tok" \
#       --corpus research/data/wikitext2_test.txt \
#       --n-chunks 3 --time 01:30:00
#
#   # Coding-agent structured sweep
#   bash research/scripts/submit_sweep.sh \
#       --account bsc93 \
#       --models "models/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf" \
#       --corpus research/data/code_longcode.jsonl \
#       --corpus-mode structured \
#       --n-ctx 5120 --n-chunks 0 --time 04:00:00
#
#   # Dry run
#   bash research/scripts/submit_sweep.sh --account x --dry-run \
#       --prefill-tokens "1024 4096 16384"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/research/scripts"
RESULTS_DIR="$REPO_ROOT/research/results"
mkdir -p "$RESULTS_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODELS="models/Qwen3-8B-Q8_0.gguf"
PREFILL_TOKENS="1024 4096 16384"
SCORE_WINDOWS="512 1024 2048"
QUANTS="fp16 int8_ch int4_ch"
CORPUS="research/data/wikitext2_test.txt"
CORPUS_MODE="flat"
N_CHUNKS=5
N_CTX_OVERRIDE=0        # 0 = auto (prefill + max_window); set for structured mode
QUANT_GROUP_SIZE=128
N_GPUS=1
SLURM_TIME="08:00:00"
SLURM_ACCOUNT=""
SLURM_QOS="acc_bscls"
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)          SLURM_ACCOUNT="$2";    shift 2 ;;
        --qos)              SLURM_QOS="$2";        shift 2 ;;
        --models)           MODELS="$2";           shift 2 ;;
        --prefill-tokens)   PREFILL_TOKENS="$2";   shift 2 ;;
        --score-windows)    SCORE_WINDOWS="$2";    shift 2 ;;
        --quants)           QUANTS="$2";           shift 2 ;;
        --corpus)           CORPUS="$2";           shift 2 ;;
        --corpus-mode)      CORPUS_MODE="$2";      shift 2 ;;
        --n-chunks)         N_CHUNKS="$2";         shift 2 ;;
        --n-ctx)            N_CTX_OVERRIDE="$2";   shift 2 ;;
        --quant-group-size) QUANT_GROUP_SIZE="$2"; shift 2 ;;
        --n-gpus)           N_GPUS="$2";           shift 2 ;;
        --time)             SLURM_TIME="$2";       shift 2 ;;
        --dry-run)          DRY_RUN=1;             shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# MN5 rule: 20 CPUs per GPU
N_CPUS=$(( N_GPUS * 20 ))
N_THREADS=$N_CPUS

if [[ -z "$SLURM_ACCOUNT" ]]; then
    echo "ERROR: --account is required on MareNostrum 5."
    echo "  Usage: bash submit_sweep.sh --account bsc_your_project ..."
    exit 1
fi

# Compute max score window
MAX_WINDOW=$(echo $SCORE_WINDOWS | tr ' ' '\n' | sort -n | tail -1)

# ── Build job list: (model, prefill_tokens) pairs ─────────────────────────────
MODELS_ARR=($MODELS)
PREFILL_ARR=($PREFILL_TOKENS)
N_JOBS=$(( ${#MODELS_ARR[@]} * ${#PREFILL_ARR[@]} ))

echo "=== Sweep configuration (MareNostrum 5) ==="
echo "  Account:       $SLURM_ACCOUNT  (qos: $SLURM_QOS)"
echo "  Models:        ${MODELS_ARR[*]}"
echo "  Prefill sizes: ${PREFILL_ARR[*]}"
echo "  Score windows: $SCORE_WINDOWS  (decode $MAX_WINDOW tokens per chunk)"
echo "  n_ctx per job: prefill + $MAX_WINDOW  (auto)"
echo "  Quants:        $QUANTS"
echo "  Corpus:        $CORPUS ($CORPUS_MODE)"
echo "  N_CHUNKS:      $N_CHUNKS  (strided across corpus)"
echo "  Group size:    $QUANT_GROUP_SIZE"
echo "  GPUs/job:      $N_GPUS  (→ $N_CPUS CPUs, $N_THREADS threads)"
echo "  Time limit:    $SLURM_TIME"
echo "  Total jobs:    $N_JOBS (array 0-$((N_JOBS-1)))"
echo ""

# Param file: IDX  MODEL  PREFILL  MODEL_STEM
PARAM_FILE_COPY="$RESULTS_DIR/sweep_params_$(date +%Y%m%d_%H%M%S).txt"

IDX=0
for MODEL in "${MODELS_ARR[@]}"; do
    MODEL_STEM="$(basename "$MODEL" .gguf)"
    for PREFILL in "${PREFILL_ARR[@]}"; do
        N_CTX_JOB=$(( PREFILL + MAX_WINDOW ))
        echo "$IDX $MODEL $PREFILL $MODEL_STEM" >> "$PARAM_FILE_COPY"
        echo "  job $IDX: $MODEL_STEM  prefill=$PREFILL  n_ctx=$N_CTX_JOB"
        IDX=$((IDX + 1))
    done
done
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "(dry-run: param file written to $PARAM_FILE_COPY)"
    echo "(dry-run: not submitting)"
    exit 0
fi

# ── Generate and submit the array job script ──────────────────────────────────
CORPUS_ABS="$REPO_ROOT/$CORPUS"
[[ "$CORPUS" = /* ]] && CORPUS_ABS="$CORPUS"

JOB_SCRIPT="$(mktemp /tmp/sweep_job.XXXXXX.sh)"
trap "rm -f $JOB_SCRIPT" EXIT

cat > "$JOB_SCRIPT" << SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=kv_sweep
#SBATCH --output=$RESULTS_DIR/slurm_%A_%a.out
#SBATCH --error=$RESULTS_DIR/slurm_%A_%a.err
#SBATCH --array=0-$((N_JOBS-1))
#SBATCH --partition=acc
#SBATCH --qos=$SLURM_QOS
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --time=$SLURM_TIME
#SBATCH --gres=gpu:$N_GPUS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$N_CPUS

set -euo pipefail
cd "$REPO_ROOT"

echo "=== Job \$SLURM_ARRAY_JOB_ID[\$SLURM_ARRAY_TASK_ID] ==="
echo "Node: \$(hostname)"
echo "GPUs: \$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Load required modules
module load GCC/12.3.0
module load cuda/12.1 2>/dev/null || true
module load anaconda/2024.02
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama
# Anaconda Python has RPATH pointing to its own libstdc++ (lacks GLIBCXX_3.4.30).
# LD_PRELOAD overrides RPATH — force GCC 12's libstdc++ to load first.
GCC12_LIBSTDCXX="/apps/GPP/GCC/12.3.0/lib64/libstdc++.so.6"
if [[ -f "\$GCC12_LIBSTDCXX" ]]; then
    export LD_PRELOAD="\$GCC12_LIBSTDCXX\${LD_PRELOAD:+:\$LD_PRELOAD}"
    echo "LD_PRELOAD: \$GCC12_LIBSTDCXX"
    strings "\$GCC12_LIBSTDCXX" | grep 'GLIBCXX_3.4.3' | tail -1
else
    echo "WARNING: GCC 12 libstdc++ not found at \$GCC12_LIBSTDCXX"
fi

# Read this job's parameters (IDX MODEL PREFILL MODEL_STEM)
LINE=\$(awk "NR==\$((SLURM_ARRAY_TASK_ID+1))" "$PARAM_FILE_COPY")
MODEL=\$(     echo \$LINE | awk '{print \$2}')
PREFILL=\$(   echo \$LINE | awk '{print \$3}')
MODEL_STEM=\$(echo \$LINE | awk '{print \$4}')
N_CTX=\$(( PREFILL + $MAX_WINDOW ))

echo "Model: \$MODEL_STEM  prefill=\$PREFILL  n_ctx=\$N_CTX  score_windows=$SCORE_WINDOWS"

OUT="$RESULTS_DIR/results_\${MODEL_STEM}_prefill\${PREFILL}_\$(date +%Y%m%d_%H%M%S).json"

if [[ "$CORPUS_MODE" == "flat" ]]; then
    python3 "$SCRIPTS_DIR/run_sweep.py" "\$MODEL" "$CORPUS_ABS" \\
        --corpus-mode flat \\
        --n-chunks $N_CHUNKS \\
        --n-threads $N_THREADS \\
        --n-gpu-layers 99 \\
        --prefill-tokens \$PREFILL \\
        --score-windows $SCORE_WINDOWS \\
        --quants $QUANTS \\
        --quant-group-size $QUANT_GROUP_SIZE \\
        --out "\$OUT"
else
    # Structured mode: use --n-ctx override or auto from JSONL lengths
    N_CTX_STRUCTURED=${N_CTX_OVERRIDE}
    python3 "$SCRIPTS_DIR/run_sweep.py" "\$MODEL" "$CORPUS_ABS" \\
        --corpus-mode structured \\
        \$([ \$N_CTX_STRUCTURED -gt 0 ] && echo "--n-ctx \$N_CTX_STRUCTURED") \\
        --n-chunks $N_CHUNKS \\
        --n-threads $N_THREADS \\
        --n-gpu-layers 99 \\
        --quants $QUANTS \\
        --quant-group-size $QUANT_GROUP_SIZE \\
        --out "\$OUT"
fi

echo "Saved: \$OUT"
SBATCH_EOF

echo "Submitting array job (0-$((N_JOBS-1)))..."
sbatch "$JOB_SCRIPT"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $RESULTS_DIR/slurm_<jobid>_<taskid>.out"
