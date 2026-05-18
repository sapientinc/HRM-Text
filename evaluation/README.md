## HRM evaluation

Evaluation takes 1 GPU (assmuing 80 GiB VRAM). Lower batch_size when OOM issues occur.

```bash
python -m evaluation.main ckpt_path="<CHECKPOINT_PATH>"
```

## Baseline evaluation

```bash
# Llama3.2 3B
python -m evaluation.main ckpt_path="unsloth/Llama-3.2-3B" config="evaluation/config/vllm_benchmarking.yaml"
lm-eval run --model vllm --model_args pretrained=unsloth/Llama-3.2-3B,max_model_len=3072 --tasks minerva_math --gen_kwargs temperature=0.0 --batch_size auto

# Olmo-3 7B
python -m evaluation.main ckpt_path="allenai/Olmo-3-1025-7B" config="evaluation/config/vllm_benchmarking.yaml"

# Qwen-3.5 2B
python -m evaluation.main ckpt_path="Qwen/Qwen3.5-2B" config="evaluation/config/vllm_benchmarking.yaml"
lm-eval run --model vllm --model_args pretrained=Qwen/Qwen3.5-0.8B,max_model_len=3072 --tasks minerva_math --gen_kwargs temperature=0.0 --batch_size auto

# Ouro 1.4B
python -m evaluation.main ckpt_path="ByteDance/Ouro-1.4B" config="evaluation/config/vllm_benchmarking.yaml" trust_remote_code=True gpu_memory_utilization=0.8
lm-eval run --model vllm --model_args pretrained=ByteDance/Ouro-1.4B,max_model_len=3072,gpu_memory_utilization=0.8,trust_remote_code=True,add_bos_token=True --tasks minerva_math --gen_kwargs temperature=0.0 --batch_size auto
```