# New Machine Setup

## Directory Structure
```
research/
  scripts/   — all Python scripts
  data/      — corpus text files (gitignored; regenerate with fetch_*.py)
  results/   — JSON results + PNG figures
  context/   — this documentation
```

---

## MareNostrum 5 (BSC) — Full Setup Procedure

### Step 1: Load modules (every session, in this order)
```bash
module load GCC/12.3.0       # MUST come before anaconda
module load anaconda/2024.02

# Fix libstdc++ conflict — anaconda RPATH bakes in old version; GCC's overrides it
export LD_PRELOAD="/apps/GPP/GCC/12.3.0/lib64/libstdc++.so.6"
```
**This export must be repeated every new shell session.**  Consider adding to `~/.bashrc` or your job scripts.

### Step 2: Conda environment (create once)
```bash
conda create -n llama python=3.11 numpy matplotlib --offline -y
conda activate llama
# ml_dtypes not available offline on MN5 — skip fp8 quants there
# cupy: install if internet available on login node, else skip GPU quant
pip install cupy-cuda12x   # for GPU-side KV quantization
```

From then on, each session:
```bash
conda activate llama
```

### Step 3: Build — MUST run on a compute node
Login node has a different CPU (no AVX512-VNNI) — binaries built or run there may crash.

```bash
# Get an interactive GPU session
srun --partition=acc --account=bsc93 --qos=acc_bscls \
     --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --pty bash

# Inside the compute node: re-load environment
module load GCC/12.3.0
module load anaconda/2024.02
export LD_PRELOAD="/apps/GPP/GCC/12.3.0/lib64/libstdc++.so.6"
conda activate llama

cd ~/llama.cpp   # or wherever repo lives

CC=gcc CXX=g++ cmake -B build_release \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="90" \
    -DLLAMA_BUILD_SERVER=OFF

cmake --build build_release --config Release -j $(nproc)
# → build_release/bin/libllama.so
```

GPU is H100 (sm_90) — `CMAKE_CUDA_ARCHITECTURES="90"` is required.
`LLAMA_BUILD_SERVER=OFF` avoids OpenSSL version conflict on MN5.

### Step 4: Download models (login node, wget — no internet on compute nodes)
```bash
# Qwen3-8B
wget -O models/Qwen3-8B-Q8_0.gguf \
    "https://huggingface.co/bartowski/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf"

# Llama-4-Scout-17B (Q4_K_M, ~55GB — fits in H100 80GB)
wget -O models/Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf \
    "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/..."
# Or pipe directly if disk is tight:
# curl -L <url> | ssh mn5 'cat > models/model.gguf'
```

### Step 5: Download corpora (login node — needs internet)
```bash
# wikitext2, wikitext103, c4 — requires datasets package
pip install datasets huggingface_hub
python3 research/scripts/fetch_wikitext.py --dataset wikitext2 wikitext103 c4
# → research/data/wikitext2_test.txt (~245K tokens)
# → research/data/wikitext103_test.txt (~314K tokens)
# → research/data/c4_val.txt (~500K tokens)
```

### Step 6: Smoke test (on compute node)
```bash
python3 research/scripts/run_sweep.py models/Qwen3-8B-Q8_0.gguf \
    research/data/wikitext2_test.txt \
    --n-gpu-layers 99 --flash-attn \
    --prefill-tokens 1024 --score-windows 512 \
    --n-chunks 1 --n-threads 20 \
    --quants fp16 \
    --out /tmp/smoke.json
# Expected: ppl@512 ≈ 5.5
```

### Step 7: Full sweep (via SLURM job array)
```bash
bash research/scripts/submit_sweep.sh \
    --account bsc93 --qos acc_bscls \
    --models "models/Qwen3-8B-Q8_0.gguf" \
    --prefill-tokens "1024 4096 16384" \
    --score-windows "512 1024 2048" \
    --quants "fp16 int8_ch int4_ch:int4_tok int3_ch int3_ch:int3_tok int2_ch int2_ch:int2_tok" \
    --quant-group-size 128 \
    --corpus research/data/wikitext2_test.txt \
    --n-chunks 5 --time 08:00:00
```

---

## Generic Machine Setup

### 1. Build libllama.so
```bash
# CPU only
cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build_release --config Release -j $(nproc)

# With CUDA (specify your GPU arch)
CC=gcc CXX=g++ cmake -B build_release \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build_release --config Release -j $(nproc)

# Result: build_release/bin/libllama.so
```

### 1b. Profiling builds (`build_nvtx` / `build_perf`)

Separate build trees, because both options add per-op overhead that must not
contaminate the timing runs done with `build_release`. Same source, different
flags — keep all three trees side by side.

```bash
# NVTX ranges: one range per graph node, named "<tensor> [ne0xne1xne2xne3] <op>".
# Needed for nsys traces where kernels must be attributed back to layers/ops.
CC=gcc CXX=g++ cmake -B build_nvtx \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DGGML_CUDA_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" -DGGML_NATIVE=OFF
cmake --build build_nvtx --config Release -j $(nproc)

# Per-op cudaEvent timing: prints [SUMMARY] / [Kernel sum] / [GPU wall] /
# [Launch/sync overhead] / [tok/s] per phase at context teardown.
# This is what separates kernel time from the ~2.2 ms/step pipeline bubble.
CC=gcc CXX=g++ cmake -B build_perf \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DGGML_CUDA_PERF=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" -DGGML_NATIVE=OFF
cmake --build build_perf --config Release -j $(nproc)
```

`CMAKE_CUDA_ARCHITECTURES`: **86** on the local RTX box, **90** on MN5 (H100).
`GGML_NATIVE=OFF` keeps the binary runnable on a login node with a different CPU.

Consumers of these trees (defaults already point at them):

| Build | Used by | Notes |
|-------|---------|-------|
| `build_nvtx` | `profile.sh`, `sweep_batched_bench.py --bin`, `profile_ops_sweep.py --bin` | pair with `nsys profile --trace=cuda,nvtx,osrt` |
| `build_perf` | direct runs of `llama-batched-bench` / `llama-cli` | timing prints go to stderr on teardown |

Add `GGML_CUDA_DISABLE_GRAPHS=1` to the environment when tracing decode — CUDA
graph replay collapses the per-node ranges/events into one opaque launch.

### 2. Python dependencies
```bash
pip install numpy matplotlib ml_dtypes
pip install cupy-cuda12x   # GPU-side KV quantization (requires CUDA 12)
```

### 3. LIB_PATH
Scripts resolve the library path relative to their own location:
```python
LIB_PATH = os.path.join(SCRIPT_DIR, "../../build_release/bin/libllama.so")
```
This works as long as scripts live in `research/scripts/` (2 levels from repo root).

---

## GPU-Side KV Quantization

Enabled automatically when `--n-gpu-layers > 0` and `--flash-attn` and CuPy is installed.
Flash Attention is required to keep V un-transposed (clean `[n_cells, n_embd]` layout).

Without `--flash-attn`: falls back to CPU parse_state (slow for per-token quants: ~3800s).
With `--flash-attn` + CuPy: GPU in-place quantization (~2s for per-token, ~400× speedup).

```bash
# GPU path (fast)
python3 research/scripts/run_sweep.py model.gguf corpus.txt \
    --n-gpu-layers 99 --flash-attn \
    --quants int4_ch:int4_tok int3_tok_g32 ...

# CPU path (fallback, slow for per-token)
python3 research/scripts/run_sweep.py model.gguf corpus.txt \
    --n-gpu-layers 0 \
    --quants int4_ch int4_ch:int4_tok ...
```
