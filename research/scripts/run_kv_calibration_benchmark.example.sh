#!/usr/bin/env bash
# Example: capture cluster timings for multiple decode lengths (one JSON per decode_len).
# Requires GGUF path(s) and a working llama.cpp build (libllama). Not run in CI.
#
#   MODEL=/path/to/Qwen3-8B-Q8_0.gguf
#   bash research/scripts/run_kv_calibration_benchmark.example.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/research/scripts:${PYTHONPATH:-}"

MODEL="${MODEL:?Set MODEL to a GGUF path}"
OUTDIR="${OUTDIR:-${ROOT}/research/results/manual_kv_sweep}"

mkdir -p "$OUTDIR"
for DECODE in 512 2048 4096; do
  python3 "${ROOT}/research/scripts/benchmark_kv_timing.py" "$MODEL" \
    --hw h100-sxm \
    --model-preset qwen3-8b \
    --n-gpu-layers 99 \
    --flash-attn \
    --prompt-lens 2048 \
    --decode-len "$DECODE" \
    --batch-sizes 1 4 \
    --kv-types f16 q8_0 q4_0 \
    --out "${OUTDIR}/kv_timing_h100_d${DECODE}.json"
done

echo "Done. Point roofline_layer.py --calibration-json at the JSON that matches your --n-decode."
