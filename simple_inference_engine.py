from typing import Any, Iterator, Generator, Optional
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import os
import yaml

import torch
from torch import Tensor, nn
import torch.utils._pytree as pytree
import numpy as np
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from transformers import AutoTokenizer, PreTrainedTokenizer

from pretrain import Carry, PretrainConfig, V1DatasetMeta, load_model_class, AdamATan2


@dataclass
class InferenceCheckpoint:
    model: nn.Module
    carry: Carry
    tokenizer: PreTrainedTokenizer
    tokenizer_info: dict[str, Any]

    def tokenize_prompt(self, condition: str, prompt: str) -> np.ndarray:
        condition_tokens = "".join(self.tokenizer_info["condition_mapping"][c] for c in condition.split(","))
        return self.tokenizer(f'{self.tokenizer_info["boq"]}{condition_tokens}{prompt}{self.tokenizer_info["eoq"]}',
                              return_tensors="np", return_attention_mask=False, add_special_tokens=False)["input_ids"][0]  # pyright: ignore[reportIndexIssue]
    
    def decode_generation(self, tokens: np.ndarray, eos_id: int) -> str:
        if tokens.size > 0 and tokens[-1] == eos_id:
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens)  # pyright: ignore[reportReturnType]


def inference_load_checkpoint(ckpt_path: str, ckpt_epoch: Optional[int], ckpt_use_ema: bool):
    # Load Checkpoint
    # Load config
    with open(os.path.join(ckpt_path, "all_config.yaml"), "r") as f:
        model_cfg = PretrainConfig(**yaml.safe_load(f))
    with open(os.path.join(ckpt_path, "train_metadata.yaml"), "r") as f:
        train_metadata = V1DatasetMeta(**yaml.safe_load(f))

    # Create model
    model_cls = load_model_class(model_cfg.arch.name)
    head_cls = load_model_class(model_cfg.arch.head)
    with torch.device("cuda"):
        combined_cfg = model_cfg.arch.model_dump() | train_metadata.model_dump() | model_cfg.data.model_dump()

        model: nn.Module = model_cls(combined_cfg)
        # Attach loss head
        model = head_cls(model, combined_cfg)
        # Optimizer (ONLY for loading states)
        optim = AdamATan2(model.parameters(),
                        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
                        betas=(model_cfg.beta1, model_cfg.beta2),
                        weight_decay=model_cfg.weight_decay,
                        ema=model_cfg.ema)
    
    # Detect checkpoint epoch if not specified
    if ckpt_epoch is None:
        ckpt_files = glob(os.path.join(ckpt_path, "fsdp2_epoch_*"))
        if len(ckpt_files) == 0:
            raise ValueError(f"No checkpoint files found in {ckpt_path}")

        ckpt_epoch = max(int(Path(f).stem.split("_")[-1]) for f in ckpt_files)
        print(f"Detected latest checkpoint epoch: {ckpt_epoch}")

    # Load checkpoint
    dcp.load({"model": model.state_dict(), "optim": get_optimizer_state_dict(model, optim)},  # pyright: ignore[reportPrivateImportUsage]
        checkpoint_id=os.path.join(ckpt_path, f"fsdp2_epoch_{ckpt_epoch}"),
        no_dist=True  # <--- Critical for single rank loading
    )
    carry = torch.load(os.path.join(ckpt_path, f"carry_epoch_{ckpt_epoch}.0.pt"), map_location="cuda")

    # Use EMA weights
    if ckpt_use_ema:
        optim.swap_ema()
    # Cast to fwd dtype & eval mode
    model = model.to(getattr(torch, model_cfg.fwd_bwd_dtype)).eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_metadata.tokenizer_info["tokenizer_path"], use_fast=True)
    return InferenceCheckpoint(
        model=model,
        carry=carry,
        tokenizer=tokenizer,
        tokenizer_info=train_metadata.tokenizer_info
    )


@torch.compile(fullgraph=True)
def _sample_gumbel(logits: Tensor, temp: Tensor):
    scaled_logits = logits.to(torch.float32) / temp
    return (scaled_logits - torch.log(-torch.log(torch.rand_like(scaled_logits).clamp_min(torch.finfo(scaled_logits.dtype).tiny)))).argmax(-1)


def _sample(logits: Tensor, temp: float) -> Tensor:
    if temp < 1e-5:
        return logits.argmax(-1)
    return _sample_gumbel(logits, torch.tensor(temp, dtype=torch.float32))


@torch.compile(fullgraph=True)
def _prefill(model: nn.Module, carry: Carry, inputs: Tensor, cache: Any) -> Tensor:
    return model(carry=carry, batch={"inputs": inputs.unsqueeze(0), "position_ids": torch.arange(inputs.shape[0]), "cache": cache, "cache_lengths": 0})[-1][..., -1, :]


@torch.compile(dynamic=False, fullgraph=True)
def _batched_decode(model: nn.Module, carry: Carry, inputs: Tensor, cache: Any, cache_lengths: Tensor) -> Tensor:
    return model(carry=carry, batch={"inputs": inputs.unsqueeze(-1), "position_ids": cache_lengths.unsqueeze(-1), "cache": cache, "cache_lengths": cache_lengths})[-1][..., -1, :]


@torch.inference_mode()
def inference_generate(ckpt: InferenceCheckpoint, iterator: Iterator[tuple[int, tuple[str, str]]], max_tokens: int, max_generation: int, batch_size: int, temp: float = 0.0) -> Generator[tuple[int, str], None]:
    def fetch_next():
        for pid, p_tuple in iterator:
            tok = ckpt.tokenize_prompt(*p_tuple)
            # Check length: if it exceeds or equals max_tokens, it will overflow buffers
            if tok.size >= max_tokens:
                yield pid, ""  # Instantly yield empty string to the caller
            else:
                return pid, tok
        return -1, None

    # Stop condition
    stop_token: int = ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoa"])  # pyright: ignore[reportAssignmentType]

    # Create GPU tensors: KV-cache
    gpu_cache = ckpt.model.create_cache(max_batch_size=batch_size, max_seq_len=max_tokens, dtype=torch.bfloat16, device="cuda")  # FIXME: hardcoded dtype # pyright: ignore[reportCallIssue]
    gpu_cache_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    gpu_last_tokens = torch.zeros((batch_size, ), dtype=torch.long, device="cuda")

    generated = np.zeros((batch_size, max_tokens), dtype=np.int64)
    generated_starts = np.zeros(batch_size, dtype=np.int64)
    generated_lengths = np.zeros(batch_size, dtype=np.int64)
    stopped = np.ones(batch_size, dtype=bool)

    # Output ID tracking
    generation_ids = [-1] * batch_size
    # Prefetch tokenized
    tokenized_prompt_id, tokenized_prompt = yield from fetch_next()

    while True:
        # PHASE 1: PREFILL & YIELD (Optimized for CPU-GPU Overlap)
        for i in stopped.nonzero()[0]:
            # Launch GPU prefill kernel
            launched_prefill = False
            if tokenized_prompt is not None:
                length = tokenized_prompt.size  # pyright: ignore[reportOptionalMemberAccess]
                inputs = torch.from_numpy(tokenized_prompt).cuda()  # <--- NOTE CPU to GPU (async)

                torch._dynamo.mark_dynamic(inputs, 0, min=1, max=max_tokens)
                gpu_last_tokens[i] = _sample(_prefill(ckpt.model, ckpt.carry, inputs, pytree.tree_map(lambda x: x[i: i+1], gpu_cache)), temp)[0]
                gpu_cache_lengths[i] = length
                launched_prefill = True

            # ---- De-tokenize & yield (Overlap with prefill)
            if generation_ids[i] != -1:
                yield generation_ids[i], ckpt.decode_generation(generated[i, generated_starts[i]: generated_lengths[i]], stop_token)
                generation_ids[i] = -1

            # ---- Prefetch tokenized (Overlap with prefill)
            if launched_prefill:
                generation_ids[i] = tokenized_prompt_id
                tokenized_prompt_id, tokenized_prompt = yield from fetch_next()

                generated_starts[i] = max(length, max_tokens - max_generation)  # pyright: ignore[reportPossiblyUnboundVariable]
                generated_lengths[i] = generated_starts[i] + 1

                last_tokens = gpu_last_tokens[i].item()  # <--- NOTE BLOCKING SYNC
                generated[i, generated_starts[i]] = last_tokens
                stopped[i] = (last_tokens == stop_token) or (generated_lengths[i] >= max_tokens)

        # PHASE 2: DECODE
        if not stopped.all():
            # Decode one token
            gpu_last_tokens = _sample(_batched_decode(ckpt.model, ckpt.carry, gpu_last_tokens, gpu_cache, gpu_cache_lengths), temp)
            gpu_cache_lengths.add_(1).clamp_max_(max_tokens - 1)  # Saturating add: prevent buffer overflow

            # Put to generated (Overlap with decode)
            active_mask = ~stopped
            generated_lengths[active_mask] += 1

            last_tokens = gpu_last_tokens.cpu().numpy()  # <--- NOTE BLOCKING SYNC
            generated[active_mask, generated_lengths[active_mask] - 1] = last_tokens[active_mask]
            stopped |= (last_tokens == stop_token) | (generated_lengths >= max_tokens)
        else:
            # Exit condition: if everything is stopped AND no more to prefill
            if tokenized_prompt is None:
                break
    
    # Flush: yield any remaining completed generations that were left in the pipeline
    for i in range(batch_size):
        if generation_ids[i] != -1:
            yield generation_ids[i], ckpt.decode_generation(generated[i, generated_starts[i]: generated_lengths[i]], stop_token)
