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

### Step 3b: MN5 profiling tree (`build_nvtx`)

The two MN5 trees as actually configured on 2026-05-04, recovered from their
CMakeCache files:

```bash
# build_release -- CUDA 12.3 from the module tree (/apps/ACC/CUDA/12.3/bin/nvcc).
# NOTE: NVTX is ON here too, so this tree is already nsys-traceable; that is why
# profile.sh points nsys at build_release rather than build_nvtx.
module load CUDA/12.3
CC=gcc CXX=g++ cmake -B build_release \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DGGML_CUDA_NVTX=ON \
    -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build_release --config Release -j $(nproc)

# build_nvtx -- adds per-op timing and disables graphs. Picked up nvcc from
# /usr/local/cuda-12.2 because the accel-sim exports (CUDA_INSTALL_PATH,
# PATH) from ~/.Command.md were active in that shell. Harmless, but pass
# -DCMAKE_CUDA_COMPILER explicitly if you want it pinned to the module.
cmake -B build_nvtx \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NVTX=ON -DGGML_CUDA_PERF=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build_nvtx --config Release -j $(nproc)
```

Both trees are sm_90 with `GGML_NATIVE=OFF` so the binaries stay runnable on a
login node whose CPU differs from the compute nodes.

Contradiction with Step 3 above: the real `build_release` has
`LLAMA_BUILD_SERVER=ON` and `LLAMA_OPENSSL=ON`, not the `-DLLAMA_BUILD_SERVER=OFF`
that Step 3 recommends. The OpenSSL conflict that motivated that flag no longer
bites -- leave the server on unless a configure error says otherwise.

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

### 1b. Profiling builds (`build_nvtx`)

Reconstructed from the CMakeCache of the MN5 trees (configured 2026-05-04); no
cmake line survives in shell history, so these are recovered from the cache
entries that differ from upstream defaults plus the `:UNINITIALIZED=` marker,
which is set only for `-D` values passed on the command line.

```bash
# Local box (RTX, sm_86). Profiling tree: NVTX ranges + per-op cudaEvent timing,
# CUDA graphs OFF so every node keeps its own range and event pair.
cmake -B build_nvtx \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NVTX=ON -DGGML_CUDA_PERF=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build_nvtx --config Release -j $(nproc)
```

`GGML_CUDA_GRAPHS=OFF` at configure time is what makes this tree usable: graph
replay collapses every per-node range and event into one opaque launch. Setting
`GGML_CUDA_DISABLE_GRAPHS=1` in the environment does the same thing at runtime
and is what `profile.sh` uses against a graphs-enabled tree.

The local `build_perf` tree (PERF on, NVTX off) is a split of the same idea and
is not mirrored on MN5 — MN5 puts both options in `build_nvtx`.

Consumers (defaults already point at these paths):

| Build | Used by |
|-------|---------|
| `build_nvtx` | `profile.sh`, `sweep_batched_bench.py --bin`, `profile_ops_sweep.py --bin` |
| `build_perf` (local only) | direct `llama-batched-bench` / `llama-cli` runs; timing prints to stderr on teardown |

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
