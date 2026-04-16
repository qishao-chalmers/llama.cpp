#### for remote server
module load cmake
conda activate llama
CC=gcc CXX=g++ cmake -B build_release -DCMAKE_BUILD_TYPE=Release  -DBUILD_SHARED_LIBS=ON  -DGGML_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES="90" -DGGML_NATIVE=OFF

#### for local machine
# Avoid conda / env leaking a RISC-V or other cross CC/CXX; use host GCC 10 explicitly.
unset CC CXX CPP

cmake -B build_release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-10 \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="90" \
    -DGGML_NATIVE=OFF
cmake --build build_release --config Release -j $(nproc)


python3 research/scripts/run_sweep.py ../temporary/models/Qwen3-14B-Q8_0.gguf  --prefill-tokens "16384"  --score-windows 512 1024  --quants fp16 int8_ch int4_ch int4_tok int4_ch:int4_tok int3_ch int3_tok int3_ch:int3_tok int2_ch  int2_tok int2_ch:int2_tok --quant-group-size 128  research/data/wikitext2_test.txt  --n-chunks 1 --out results_test.json
