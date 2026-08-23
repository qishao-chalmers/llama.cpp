#!/usr/bin/env bash
# profile_nsys_sweep.sh — nsys-profile llama-batched-bench across models × quants × npl.
#
# Workload terms (llama-batched-bench):
#   NPP (-npp)  — prompt tokens per sequence (prefill / prompt processing).
#   NTG (-ntg)  — tokens generated per sequence after the prompt (decode / TG).
#   NPL (-npl)  — number of parallel sequences (batch of independent prompts).
#   Context -c  — set to (NPP + NTG) * NPL so the KV cache fits the full run.
#
# Workload (fixed): NPP=4096, NTG=64, graphs disabled; NPL swept over 1 4 8 16.
# Grid: 4 models × 3 weight quants × 4 npl = 48 runs.
#
# Usage:
#   bash research/scripts/profile_nsys_sweep.sh --dry-run
#   bash research/scripts/profile_nsys_sweep.sh
#   bash research/scripts/profile_nsys_sweep.sh --models qwen3-8b --quants Q8_0 --npl 1 4
#   bash research/scripts/profile_nsys_sweep.sh --bin ./build_release/bin/llama-batched-bench
#
# Outputs: research/results/nsys/{model}_{quant}_npp4096_ntg64_npl{N}_nograph.nsys-rep
# Skips missing GGUFs and existing .nsys-rep files. Continues on run failure (e.g. OOM).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
BIN="${BIN:-$REPO_ROOT/build_release/bin/llama-batched-bench}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/research/results/nsys}"

NPP=4096   # prompt tokens per sequence (prefill)
NTG=64     # generated tokens per sequence (decode)
N_BATCH=2048
N_GPU_LAYERS=99
DRY_RUN=0

# Default full grid
MODELS=(qwen3-8b qwen3-14b gemma3-12b gemma3-27b)
QUANTS=(Q2_K Q3_K_M Q8_0)
NPLS=(1 4 8 16)   # parallel sequences (-npl)

# model_key → basename stem before quant suffix
# Q8_0 files use the stem as-is; Q2_K / Q3_K_M append -<QUANT>
declare -A MODEL_STEM=(
    [qwen3-8b]="Qwen3-8B-Q8_0"
    [qwen3-14b]="Qwen3-14B-Q8_0"
    [gemma3-12b]="gemma-3-12b-it-q8_0"
    [gemma3-27b]="gemma-3-27b-it-q8_0"
)

gguf_path() {
    local model="$1" quant="$2"
    local stem="${MODEL_STEM[$model]}"
    if [[ -z "$stem" ]]; then
        echo ""
        return
    fi
    if [[ "$quant" == "Q8_0" ]]; then
        echo "$MODELS_DIR/${stem}.gguf"
    else
        echo "$MODELS_DIR/${stem}-${quant}.gguf"
    fi
}

usage() {
    cat <<'EOF'
profile_nsys_sweep.sh — nsys profile llama-batched-bench (npp=4096, ntg=64)

Workload terms:
  NPP  prompt tokens per sequence (prefill)
  NTG  generated tokens per sequence (decode)
  NPL  parallel sequences (batch size for this sweep)

Options:
  --dry-run              Print commands only
  --models K [K ...]     Subset of: qwen3-8b qwen3-14b gemma3-12b gemma3-27b
  --quants Q [Q ...]     Subset of: Q2_K Q3_K_M Q8_0
  --npl N [N ...]        Subset of parallel seq counts (default: 1 4 8 16)
  --bin PATH             llama-batched-bench binary
  --models-dir DIR       GGUF directory (default: <repo>/models)
  --out-dir DIR          nsys output directory (default: research/results/nsys)
  --help, -h             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --bin) BIN="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --models)
            shift; MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
            ;;
        --quants)
            shift; QUANTS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do QUANTS+=("$1"); shift; done
            ;;
        --npl)
            shift; NPLS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do NPLS+=("$1"); shift; done
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

if [[ ! -x "$BIN" && "$DRY_RUN" -eq 0 ]]; then
    echo "ERROR: llama-batched-bench not executable: $BIN" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "BIN=$BIN"
echo "MODELS_DIR=$MODELS_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "Workload: npp=$NPP ntg=$NTG npl=${NPLS[*]} models=${MODELS[*]} quants=${QUANTS[*]}"
echo

ran=0
skipped=0
failed=0
missing=0

for model in "${MODELS[@]}"; do
    if [[ -z "${MODEL_STEM[$model]+x}" ]]; then
        echo "SKIP unknown model key: $model"
        ((skipped++)) || true
        continue
    fi
    for quant in "${QUANTS[@]}"; do
        model_path="$(gguf_path "$model" "$quant")"
        if [[ ! -f "$model_path" ]]; then
            echo "MISSING: $model_path"
            ((missing++)) || true
            continue
        fi
        for npl in "${NPLS[@]}"; do
            n_ctx=$(( (NPP + NTG) * npl ))
            out_base="${OUT_DIR}/${model}_${quant}_npp${NPP}_ntg${NTG}_npl${npl}_nograph"
            if [[ -f "${out_base}.nsys-rep" ]]; then
                echo "EXISTS: ${out_base}.nsys-rep — skip"
                ((skipped++)) || true
                continue
            fi

            cmd=(
                env GGML_CUDA_DISABLE_GRAPHS=1
                nsys profile
                --trace=cuda,nvtx,osrt
                -o "$out_base"
                "$BIN"
                -m "$model_path"
                -fa 1
                -npp "$NPP"
                -ntg "$NTG"
                -b "$N_BATCH"
                -ngl "$N_GPU_LAYERS"
                -c "$n_ctx"
                -npl "$npl"
            )

            echo "=== ${model} ${quant} npl=${npl} c=${n_ctx} ==="
            printf '  %q' "${cmd[@]}"
            echo

            if [[ "$DRY_RUN" -eq 1 ]]; then
                ((ran++)) || true
                continue
            fi

            if "${cmd[@]}"; then
                ((ran++)) || true
            else
                echo "FAILED: ${model} ${quant} npl=${npl} (exit $?)" >&2
                ((failed++)) || true
            fi
        done
    done
done

echo
echo "Done. ran=$ran skipped=$skipped missing=$missing failed=$failed"
if [[ "$failed" -gt 0 ]]; then
    exit 1
fi
