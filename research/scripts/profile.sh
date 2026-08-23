# ./build_nvtx/bin/llama-batched-bench -m ./models/Qwen3-14B-Q8_0.gguf -fa 1 -ngl 99 -c 2176 -npp 1024 -ntg 128 -npl 1 --cache-type-k q8_0 --cache-type-v q8_0   2>&1 | tee 1024_128_1.log
# ./build_nvtx/bin/llama-batched-bench -m ./models/Qwen3-14B-Q8_0.gguf -fa 1 -ngl 99 -c 3328 -npp 1024 -ntg 128 -npl 2 --cache-type-k q8_0 --cache-type-v q8_0   2>&1 | tee 1024_128_2.log
# ./build_nvtx/bin/llama-batched-bench -m ./models/Qwen3-14B-Q8_0.gguf -fa 1 -ngl 99 -c 5632 -npp 1024 -ntg 128 -npl 4 --cache-type-k q8_0 --cache-type-v q8_0   2>&1 | tee 1024_128_4.log
# ./build_nvtx/bin/llama-batched-bench -m ./models/Qwen3-14B-Q8_0.gguf -fa 1 -ngl 99 -c 10240 -npp 1024 -ntg 128 -npl 8 --cache-type-k q8_0 --cache-type-v q8_0  2>&1 | tee 1024_128_8.log
# ./build_nvtx/bin/llama-batched-bench -m ./models/Qwen3-14B-Q8_0.gguf -fa 1 -ngl 99 -c 19456 -npp 1024 -ntg 128 -npl 16 --cache-type-k q8_0 --cache-type-v q8_0 2>&1 | tee 1024_128_16.log

GGML_CUDA_DISABLE_GRAPHS=1  nsys profile --trace=cuda,nvtx,osrt -o qwen3_14b_prefill_batch_nograph /gpfs/projects/bsc93/bsc747505/llama.cpp/build_release/bin/llama-batched-bench -m /gpfs/projects/bsc93/bsc747505/llama.cpp/models/Qwen3-14B-Q8_0.gguf -fa 1 --no-warmup -r 1 -npp 1024 -ntg 64 -b 2048 -ngl 99 -npl 1
