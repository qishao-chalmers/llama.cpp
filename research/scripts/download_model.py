from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-14B"

# This automatically downloads and caches all necessary files
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
