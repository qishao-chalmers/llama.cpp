export GGML_CUDA_DISABLE_GRAPHS=1 

### profiling

nsys profile -o qwen3_prefill_nograph  --trace=cuda,nvtx --cuda-graph-trace=node /home/bsc/bsc747505/project/llama.cpp/build_release/bin/llama-bench -m /home/bsc/bsc747505/project/llama.cpp/models/Qwen3-8B-Q8_0.gguf -p 1024 -n 64 -b 2048 -ngl 99 --flash-attn 1 --no-warmup -r 1

ncu  --set full   --kernel-name mul_mat_q --launch-skip 249 --launch-count 1 "/home/bsc/bsc747505/project/llama.cpp/build_release/bin/llama-bench" -m /home/bsc/bsc747505/project/llama.cpp/models/Qwen3-8B-Q8_0.gguf -p 1024 -n 64 -b 2048 -ngl 99 --flash-attn 1 --no-warmup -r 1


ncu  --set full   --kernel-name mul_mat_q --launch-skip 249  --section    InstructionStats --section  ComputeWorkloadAnalysis --section   MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section  SourceCounters --section   WarpStateStats --section  SpeedOfLight_HierarchicalTensorRooflineChart --section  WarpStateStats  --launch-count 1 "/home/bsc/bsc747505/project/llama.cpp/build_release/bin/llama-bench" -m /home/bsc/bsc747505/project/llama.cpp/models/Qwen3-8B-Q8_0.gguf -p 1024 -n 64 -b 2048 -ngl 99 --flash-attn 1 --no-warmup -r 1
