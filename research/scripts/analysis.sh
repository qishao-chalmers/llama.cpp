  Step 1 — collect diagnostics (adds ~0 overhead vs --show-text, no display):                                         
  python3 research/scripts/run_sweep.py models/Qwen3-8B-Q5_K_M.gguf \
      research/data/wikitext2_test.txt \                                                                              
      --prefill-tokens 1024 --score-windows 2048 \                                                                    
      --n-chunks 1 --n-gpu-layers 35 \                                                                                
      --quants fp16 int8_ch int3_ch int2_ch \                                                                         
      --out /tmp/results.json \                                                                                       
      --save-diags /tmp/diags.json                                                                                    
                                                                                                                      
  Step 2 — plot entropy vs position:                                                                                  
  # Raw entropy per quant                                                                                             
  python3 research/scripts/plot_entropy.py /tmp/diags.json \                                                          
      --quants fp16 int8_ch int3_ch int2_ch \                                                                         
      --metric H --smooth 50 --out entropy_H.png                                                                      
                                                                                                                      
  # Delta (quant - fp16) to see where divergence concentrates                                                         
  python3 research/scripts/plot_entropy.py /tmp/diags.json \                                                          
      --quants int8_ch int3_ch int2_ch \                                                                              
      --metric H --diff --smooth 30 --out entropy_dH.png                                                              
                                                                                                                      
  # Multiple panels in one figure                                                                                     
  python3 research/scripts/plot_entropy.py /tmp/diags.json \                                                          
      --metrics H lp ppl_curve \                                                                                      
      --smooth 50 --out entropy_multi.png                                                                             
                                                                                                                      
  # Mark the 10 highest-divergence positions                                                                          
  python3 research/scripts/plot_entropy.py /tmp/diags.json \                                                          
      --metric H --diff --annotate --smooth 50 --out entropy_annotated.png                                            
                                                                                                                      
  What to look for:                                                                                                   
  - --metric H --diff: if ΔH grows monotonically with position → accumulated KV noise; if it spikes at a fixed        
  position → likely a group-size boundary or attention pattern issue                                                  
  - --metric ppl_curve: cumulative PPL — does int3_ch diverge from fp16 early (prompt KV) or late (long decode)?    
  - --metric lp --diff: which individual tokens drive the PPL gap (spikes downward = high-damage tokens)              
                                                                                                         


######## for niah test

    How to use it

  # Step 1: build (uses model for tokenization, CPU only, ~5s)
  python3 research/scripts/build_niah_dataset.py \
      models/Qwen3-14B-Q8_0.gguf \
      research/data/c4_val.txt \
      --out research/data/niah_4096.jsonl \
      --target-ctx 4096 \
      --n-positions 11 \
      --n-needles 10

  # Step 2: evaluate (GPU, generates short answers)
  python3 research/scripts/run_sweep.py \
      models/Qwen3-14B-Q8_0.gguf \
      research/data/niah_4096.jsonl \
      --corpus-mode structured \
      --eval-accuracy --eval-metric f1 \
      --n-ctx 4160 --n-chunks 0 \
      --n-gpu-layers 99 --flash-attn --n-threads 10 \
      --quants fp16 int8_ch int4_ch int3_ch int2_ch \
      --max-gen-tokens 20 \
      --stop-strings $'\n' \
      --save-per-example research/results/niah_per_ex.json \
      --out research/results/niah_results.json

  # Step 3: plot
  python3 research/scripts/plot_niah.py \
      research/results/niah_per_ex.json \
      --jsonl research/data/niah_4096.jsonl \
      --show-gen-len \
      --out research/results/niah_plot.png

