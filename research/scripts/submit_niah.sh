#!/bin/bash
#SBATCH --job-name=niah_sweep
#SBATCH --account=bsc93
#SBATCH --qos=acc_bscls
#SBATCH --partition=acc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=research/results/niah_%j.log
#SBATCH --error=research/results/niah_%j.log

# ── Environment ───────────────────────────────────────────────────────────────
module load GCC/12.3.0
module load anaconda/2024.02
export LD_PRELOAD="/apps/GPP/GCC/12.3.0/lib64/libstdc++.so.6"
conda activate llama

cd /home/qshao/Project/Fun/llama.cpp

# ── Config ────────────────────────────────────────────────────────────────────
MODEL=${MODEL:-/home/qshao/Project/Fun/models/Qwen3-8B-Q8_0.gguf}
JSONL=${JSONL:-research/data/niah_4096.jsonl}
OUT=${OUT:-research/results/niah_results.json}
PER_EX=${PER_EX:-research/results/niah_per_ex.json}
QUANTS=${QUANTS:-"fp16 int8_ch int4_ch int3_ch int2_ch"}

mkdir -p research/results

echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Quants: $QUANTS"
echo "Started: $(date)"

python3 research/scripts/run_sweep.py "$MODEL" "$JSONL" \
    --corpus-mode structured \
    --eval-accuracy --eval-metric f1 \
    --skip-ppl \
    --n-ctx 4160 --n-chunks 0 \
    --n-gpu-layers 99 --flash-attn \
    --n-threads 20 \
    --quants $QUANTS \
    --max-gen-tokens 20 \
    --stop-strings $'\n' \
    --save-per-example "$PER_EX" \
    --out "$OUT"

echo "Done: $(date)"
