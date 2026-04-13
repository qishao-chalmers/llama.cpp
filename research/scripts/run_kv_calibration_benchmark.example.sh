#!/usr/bin/env bash
# Example: one benchmark_kv_timing run with multiple decode lengths in a single JSON.
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
OUTJSON="${OUTJSON:-${OUTDIR}/kv_timing_h100.json}"

mkdir -p "$OUTDIR"
python3 "${ROOT}/research/scripts/benchmark_kv_timing.py" "$MODEL" \
  --hw h100-sxm \
  --model-preset qwen3-8b \
  --n-gpu-layers 99 \
  --flash-attn \
  --prompt-lens 2048 \
  --decode-len 512 2048 4096 \
  --batch-sizes 1 4 \
  --kv-types f16 q8_0 q4_0 \
  --decode-bucket-size 64 \
  --prefill-bucket-size 64 \
  --out "${OUTJSON}"

echo "Wrote ${OUTJSON} (rows for each weight × kv × prompt × decode_len × batch)."
