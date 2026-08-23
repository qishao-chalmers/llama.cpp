#!/bin/bash
# Run bench_nibble_gemv on the MN5 cluster (H100 SXM node).
# Usage: sbatch run_bench_nibble_gemv.sh   OR   bash run_bench_nibble_gemv.sh (interactive)
#SBATCH --job-name=bench_nibble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=../results/bench_nibble_gemv_%j.log

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../results"
mkdir -p "$OUT_DIR"

# Detect GPU arch
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "GPU compute capability: $ARCH"
echo "Building bench_nibble_gemv (arch=sm_${ARCH})..."

nvcc -O3 -arch=sm_${ARCH} --compiler-options -Wall \
    -o "$OUT_DIR/bench_nibble_gemv" \
    "$SCRIPT_DIR/bench_nibble_gemv.cu" -lm

echo "Running..."
"$OUT_DIR/bench_nibble_gemv" | tee "$OUT_DIR/bench_nibble_gemv_results.txt"

echo ""
echo "Results saved to: $OUT_DIR/bench_nibble_gemv_results.txt"
