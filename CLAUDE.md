# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- KV cache research context (cross-machine) -->
@research/CONTEXT.md

> [!IMPORTANT]
> Read [AGENTS.md](AGENTS.md) before beginning any work. This project does **not** accept pull requests that are fully or predominantly AI-generated.

## Build Commands

```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release -j $(nproc)

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run tests
cd build && ctest --output-on-failure

# Run a single test binary
./build/bin/test-tokenizer-0

# Run full CI locally (CPU-only)
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Architecture Overview

The project consists of two layered components:

**GGML** (`ggml/`) — A standalone tensor computation library:
- `ggml/include/ggml.h` — Core tensor operations API (103 KB)
- `ggml/include/ggml-backend.h` — Backend abstraction for hardware accelerators
- `ggml/src/ggml.c` — Core implementation
- `ggml/src/ggml-backend.cpp` — Backend dispatch system
- `ggml/src/ggml-quants.c` — Quantization functions
- `ggml/src/ggml-cpu/` — CPU backend with SIMD (AVX2/AVX512/NEON/SVE/SME)
- `ggml/src/ggml-cuda/` — NVIDIA GPU backend (200+ CUDA kernels)
- `ggml/src/ggml-metal/` — Apple Metal backend
- `ggml/src/ggml-vulkan/` — Vulkan cross-platform GPU backend
- Other backends: `ggml-sycl/`, `ggml-hip/`, `ggml-opencl/`, `ggml-webgpu/`, `ggml-cann/`, etc.

**Llama** (`src/`) — LLM inference built on top of GGML:
- `include/llama.h` — Public C API
- `src/llama-model.cpp` — Model loading and structure
- `src/llama-context.cpp` — Context management and inference execution
- `src/llama-arch.cpp` — Architecture-specific operations for different model types
- `src/llama-kv-cache.cpp` — KV cache management
- `src/llama-graph.cpp` — Computation graph building
- `src/llama-sampler.cpp` — Token sampling
- `src/llama-vocab.cpp` — Tokenization (SPM/BPE/WPM/UGM/RWKV)
- `src/llama-grammar.cpp` — GBNF grammar-constrained decoding
- `src/llama-adapter.cpp` — LoRA adapter support

**Common utilities** (`common/`) — Shared across all tools: argument parsing, chat templates, Jinja templating.

**Tools** (`tools/`) — CLI binaries built on the llama library:
- `tools/server/` — OpenAI-compatible HTTP server with web UI (multiple frontends in `public/`, `webui/`)
- `tools/cli/` — Command-line chat interface
- `tools/quantize/` — Model quantization
- `tools/perplexity/` — Perplexity evaluation
- `tools/imatrix/` — Importance matrix computation for quantization
- `tools/rpc/` — RPC server for distributed inference

**Python tooling**:
- `convert_hf_to_gguf.py` — Convert HuggingFace models to GGUF format
- `convert_lora_to_gguf.py` — Convert LoRA adapters
- `gguf-py/` — Python library for GGUF format manipulation

## Testing

Tests live in `tests/`. Key test binaries:
- `test-backend-ops` — Comprehensive ggml operator tests across backends (run this when modifying ggml operators)
- `test-llama-archs` — Model architecture tests
- `test-tokenizer-*` — Tokenization tests
- `test-chat` — Chat template tests
- `test-grammar-*` — Grammar constraint tests
- `test-backend-sampler` — Sampling tests

When modifying ggml operators, run `test-backend-ops` with at least two different backends to check consistency.

## Coding Guidelines

- Use `snake_case` for all names; enum values in `UPPER_CASE` prefixed with enum name
- Naming pattern: `<class>_<action>_<noun>` (e.g., `llama_sampler_chain_remove`)
- Optimize for longest common prefix in related names: `number_small`/`number_big`, not `small_number`/`big_number`
- 4-space indentation, brackets on same line, `void * ptr` / `int & a` style
- Avoid STL templates and fancy modern constructs; prefer simple `for` loops
- Use sized integer types (`int32_t`) in public APIs
- Vertical alignment for readability
- C/C++ filenames: all lowercase with dashes, `.h`/`.cpp`/`.c` extensions

## Key Concepts

- **Matrix multiplication convention**: `ggml_mul_mat(ctx, A, B)` computes `C = B * A^T` (not `A * B`)
- **Tensor dimensions**: dim 0 = columns, dim 1 = rows, dim 2 = matrices (row-major order)
- **Model file format**: GGUF (`.gguf`) — see `ggml/src/gguf.cpp` and `gguf-py/`
- **Backends can run simultaneously**: build with multiple backends (e.g., `-DGGML_CUDA=ON -DGGML_VULKAN=ON`), select at runtime with `--device`
- **Dynamic backend loading**: enabled via `GGML_BACKEND_DL` CMake option

## Server Development

See [tools/server/README-dev.md](tools/server/README-dev.md) for server-specific development documentation.
