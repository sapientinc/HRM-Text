from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
import yaml
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch import Tensor, nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

from models.layers import Carry
from models.common import IGNORE_LABEL_ID, wrap_tensor
from models.accelerator import (
    AcceleratorType,
    empty_accelerator_cache,
    memory_stats_for_device,
    set_accelerator_type,
    synchronize_device,
    torch_device_for_accelerator,
)
from models.transformer import TransformerBlock
from models.adam_atan2 import AdamATan2
from utils.functions import load_model_class, get_model_source_path
from dataset_new import V1Dataset, V1DatasetConfig, V1DatasetMeta


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    head: str


class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    path: str
    target_only: bool = True  # Only supervise Answer.


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    data: DataConfig

    # Hyperparams
    global_batch_size: int
    epochs: int
    gradient_accumulation_steps: int = pydantic.Field(default=1, ge=1)

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None
    fwd_bwd_dtype: str = "bfloat16"
    accelerator_type: AcceleratorType = "sm100"
    distributed_strategy: Literal["fsdp", "ddp", "none"] = "fsdp"
    fsdp_params_precision: Literal["fp32", "bf16"] = "fp32"
    ddp_params_precision: Literal["fp32", "bf16"] = "fp32"
    ddp_find_unused_parameters: bool = True
    compile_train_batch: bool = True
    memory_log_interval: int = 0
    empty_cache_interval: int = 0
    resume_trace: bool = False

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_resume: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_format: Literal["sharded", "unsharded"] = "sharded"
    resume_checkpoint_path: Optional[str] = None
    resume_checkpoint_tag: Optional[str] = None
    resume_epoch: Optional[int] = None
    resume_step: Optional[int] = None
    resume_batch_in_epoch: Optional[int] = None
    upcast_optimizer_state_on_resume: bool = False
    reset_ema_on_resume: bool = False

    # Extras
    seed: int = 0
    checkpoint_interval: int = 1
    checkpoint_step_interval: Optional[int] = None
    ephemeral_checkpoint_step_interval: Optional[int] = None
    log_interval: int = 5

    @pydantic.model_validator(mode='after')
    def check_intervals(self):
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.checkpoint_step_interval is not None and self.checkpoint_step_interval < 1:
            raise ValueError("checkpoint_step_interval must be >= 1 when set")
        if self.ephemeral_checkpoint_step_interval is not None and self.ephemeral_checkpoint_step_interval < 1:
            raise ValueError("ephemeral_checkpoint_step_interval must be >= 1 when set")
        if self.log_interval < 1:
            raise ValueError("log_interval must be >= 1")
        return self


@dataclass
class TrainState:
    model: nn.Module
    carry: Optional[Carry]
    
    optim: AdamATan2

    step: int
    total_steps: int
    fwd_bwd_dtype: torch.dtype
    use_cuda_autocast: bool


@dataclass
class ResumeState:
    tag: str
    step: int
    start_epoch: int
    skip_batches: int


def trace_print(config: PretrainConfig, rank: int, message: str):
    if config.resume_trace:
        print(f"[resume_trace rank={rank}] {message}", flush=True)


def create_dataloader(config: PretrainConfig, local_batch_size: int, drop_last_batch: bool, rank: int, world_size: int):
    dataset = V1Dataset(V1DatasetConfig(
        seed=config.seed,

        dataset_path=config.data.path,
        drop_last_batch=drop_last_batch,

        target_only=config.data.target_only,

        batch_max_length=local_batch_size,
        rank=rank,
        num_replicas=world_size,
    ))
    num_workers = 0 if config.accelerator_type in ("mps", "cpu", "none") else 1
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": config.accelerator_type in ("sm90", "sm100", "rocm"),
    }
    if num_workers > 0:
        dataloader_kwargs |= {
            "prefetch_factor": 8,
            "persistent_workers": True,  # NOTE: Required for correct epoch handling
        }
    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader, dataset.metadata


def apply_fsdp(module: nn.Module, param_dtype: torch.dtype):
    fully_shard(module,
                mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype,
                                               reduce_dtype=torch.get_default_dtype()),  # Use master dtype for reduction
                reshard_after_forward=False)  # Trade off VRAM for less comms
    
    assert isinstance(module, FSDPModule)
    # Disable gradient division. Adams is scale invariant.
    # The FSDP2 API for this was renamed across torch versions; support both.
    if hasattr(module, "set_gradient_divide_factor"):
        module.set_gradient_divide_factor(1.0)
    else:
        module.set_reduce_scatter_divide_factor(1.0)
    if hasattr(module, "set_force_sum_reduction_for_comms"):
        module.set_force_sum_reduction_for_comms(True)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def compute_train_extra_args(model: nn.Module, train_state: TrainState) -> dict:
    return unwrap_model(model).compute_train_extra_args(train_state)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]


def fsdp_model_dtype(config: PretrainConfig) -> torch.dtype:
    if config.fsdp_params_precision == "fp32":
        return torch.float32
    if config.fsdp_params_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported fsdp_params_precision: {config.fsdp_params_precision}")


def ddp_model_dtype(config: PretrainConfig, fwd_bwd_dtype: torch.dtype) -> torch.dtype:
    if config.ddp_params_precision == "fp32":
        return torch.float32
    if config.ddp_params_precision == "bf16":
        return fwd_bwd_dtype
    raise ValueError(f"Unsupported ddp_params_precision: {config.ddp_params_precision}")


def optimizer_ema_dtype(config: PretrainConfig) -> Optional[torch.dtype]:
    if config.distributed_strategy == "ddp" and config.ddp_params_precision == "bf16":
        return torch.float32
    if config.distributed_strategy == "fsdp" and config.fsdp_params_precision == "bf16":
        return torch.float32
    return None


def create_model_and_carry(config: PretrainConfig, train_metadata: V1DatasetMeta, local_batch_size: int, device: torch.device):
    model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump() | {"fwd_bwd_dtype": config.fwd_bwd_dtype}
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)

    # Instantiate model with head
    model_cls = load_model_class(config.arch.name)
    head_cls = load_model_class(config.arch.head)

    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        carry = model.initial_carry(local_batch_size, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue]
        # Attach loss head
        model = head_cls(model, model_cfg)

    if dist.is_available() and dist.is_initialized():
        if config.distributed_strategy == "fsdp":
            model_dtype = fsdp_model_dtype(config)
            if model_dtype != torch.float32:
                model = model.to(dtype=model_dtype)

            # Broadcast buffers
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

            # Detect TransformerBlock recursively and apply FSDP
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    apply_fsdp(module, fwd_bwd_dtype)

            apply_fsdp(model, fwd_bwd_dtype)
        elif config.distributed_strategy == "ddp":
            if device.type != "cuda":
                raise RuntimeError("distributed_strategy=ddp is currently only supported for CUDA torchrun jobs")
            model_dtype = ddp_model_dtype(config, fwd_bwd_dtype)
            if model_dtype != torch.float32:
                model = model.to(dtype=model_dtype)
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )
        elif config.distributed_strategy == "none":
            pass
        else:
            raise ValueError(f"Unsupported distributed_strategy: {config.distributed_strategy}")
    elif config.distributed_strategy not in ("fsdp", "none"):
        raise RuntimeError(f"distributed_strategy={config.distributed_strategy} requires torchrun/distributed training")

    # ----Create optimizer----
    optim = AdamATan2(model.parameters(),
                      lr=0.0,
                      betas=(config.beta1, config.beta2),
                      weight_decay=config.weight_decay,
                      ema=config.ema,
                      ema_dtype=optimizer_ema_dtype(config))

    return model, carry, optim


def init_train(config: PretrainConfig, rank: int, world_size: int, device: Optional[torch.device] = None):
    set_accelerator_type(config.accelerator_type)
    if device is None:
        device = torch_device_for_accelerator(config.accelerator_type, local_rank=rank)
    physical_batch_divisor = world_size * config.gradient_accumulation_steps
    assert config.global_batch_size % physical_batch_divisor == 0, (
        f"Global batch size {config.global_batch_size} must be divisible by "
        f"world_size * gradient_accumulation_steps ({world_size} * {config.gradient_accumulation_steps})."
    )
    local_batch_size = config.global_batch_size // physical_batch_divisor

    # Dataset
    train_loader, train_metadata = create_dataloader(config, local_batch_size, drop_last_batch=True,  rank=rank, world_size=world_size)

    # Model
    model, carry, optim = create_model_and_carry(config, train_metadata, local_batch_size, device)
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)

    # Train state
    # Estimated optimizer steps. Each epoch is iterated separately and drops its
    # own incomplete final effective batch.
    total_steps = config.epochs * int(train_metadata.total_length // config.global_batch_size)
    train_state = TrainState(
        model=model,
        carry=carry,
        optim=optim,
        
        step=0,
        total_steps=total_steps,
        fwd_bwd_dtype=fwd_bwd_dtype,
        use_cuda_autocast=(
            config.distributed_strategy == "ddp"
            and device.type == "cuda"
            and fwd_bwd_dtype != torch.float32
        ),
    )
    return train_state, train_loader, train_metadata


def update_lr(config: PretrainConfig, train_state: TrainState) -> float:
    # Linear warmup cosine schedule
    if train_state.step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, train_state.step / config.lr_warmup_steps)
    else:
        progress = (train_state.step - config.lr_warmup_steps) / (train_state.total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))

    for param_group in train_state.optim.param_groups:
        param_group["lr"] = lr

    return lr


def normalize_checkpoint_tag(tag: str) -> str:
    if tag.startswith("fsdp2_"):
        tag = tag.removeprefix("fsdp2_")
    elif tag.startswith("unsharded_"):
        tag = tag.removeprefix("unsharded_")
        if tag.endswith(".pt"):
            tag = tag.removesuffix(".pt")
    elif tag.startswith("carry_"):
        tag = tag.removeprefix("carry_")
    if os.path.basename(tag) != tag:
        raise ValueError(f"Checkpoint tag must be a name, not a path: {tag}")
    if not (tag.startswith("epoch_") or tag.startswith("step_") or tag.startswith("ephemeral_step_")):
        raise ValueError(f"Checkpoint tag must start with 'epoch_', 'step_', or 'ephemeral_step_': {tag}")
    return tag


def load_checkpoint_metadata(checkpoint_path: str, tag: str) -> dict:
    metadata_path = os.path.join(checkpoint_path, f"checkpoint_state_{tag}.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "rt") as f:
        return json.load(f)


def resolve_resume_state(config: PretrainConfig) -> Optional[ResumeState]:
    if config.resume_checkpoint_path is None and config.resume_checkpoint_tag is None:
        return None
    if config.resume_checkpoint_path is None or config.resume_checkpoint_tag is None:
        raise ValueError("Both resume_checkpoint_path and resume_checkpoint_tag must be set to resume training")

    tag = normalize_checkpoint_tag(config.resume_checkpoint_tag)
    metadata = load_checkpoint_metadata(config.resume_checkpoint_path, tag)

    step = metadata.get("step", config.resume_step)
    epoch = metadata.get("epoch", config.resume_epoch)
    batch_in_epoch = metadata.get("batch_in_epoch", config.resume_batch_in_epoch)

    if tag.startswith("epoch_"):
        tag_epoch = int(tag.removeprefix("epoch_"))
        epoch = tag_epoch if epoch is None else int(epoch)
        step = -1 if step is None else int(step)
        return ResumeState(tag=tag, step=step, start_epoch=tag_epoch + 1, skip_batches=0)

    step_match = re.fullmatch(r"(?:ephemeral_)?step_(\d+)", tag)
    if step_match is None:
        raise ValueError(f"Unsupported checkpoint tag: {tag}")
    step = int(step_match.group(1)) if step is None else int(step)
    if epoch is None or batch_in_epoch is None:
        raise ValueError(
            f"Step checkpoint {tag} needs checkpoint_state_{tag}.json or explicit "
            "resume_epoch and resume_batch_in_epoch overrides"
        )
    return ResumeState(tag=tag, step=step, start_epoch=int(epoch), skip_batches=int(batch_in_epoch))


def load_train_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int) -> Optional[ResumeState]:
    resume_state = resolve_resume_state(config)
    if resume_state is None:
        return None

    assert config.resume_checkpoint_path is not None
    carry_path = os.path.join(config.resume_checkpoint_path, f"carry_{resume_state.tag}.{rank}.pt")
    if not os.path.isfile(carry_path):
        raise ValueError(f"Carry file not found: {carry_path}")

    trace_print(config, rank, f"load_checkpoint_begin tag={resume_state.tag} format={config.checkpoint_format}")
    if config.checkpoint_format == "sharded":
        load_sharded_train_state(config, train_state, resume_state.tag)
    elif config.checkpoint_format == "unsharded":
        load_unsharded_train_state(config, train_state, resume_state.tag, rank)
    else:
        raise ValueError(f"Unsupported checkpoint_format: {config.checkpoint_format}")
    trace_print(config, rank, f"load_checkpoint_end tag={resume_state.tag}")

    if config.upcast_optimizer_state_on_resume:
        trace_print(config, rank, "upcast_optimizer_state_begin")
        upcast_optimizer_state(train_state.optim)
        trace_print(config, rank, "upcast_optimizer_state_end")
    if config.reset_ema_on_resume:
        trace_print(config, rank, "reset_ema_to_params_begin")
        reset_optimizer_ema_to_params(train_state.optim)
        trace_print(config, rank, "reset_ema_to_params_end")

    trace_print(config, rank, f"load_carry_begin path={carry_path}")
    train_state.carry = torch.load(carry_path, map_location="cuda")
    trace_print(config, rank, "load_carry_end")
    if resume_state.step >= 0:
        train_state.step = resume_state.step
    elif resume_state.tag.startswith("epoch_"):
        completed_epoch = int(resume_state.tag.removeprefix("epoch_"))
        train_state.step = int(completed_epoch * train_state.total_steps // config.epochs)
        resume_state.step = train_state.step
    else:
        raise ValueError(f"Cannot infer resume step for checkpoint tag {resume_state.tag}")
    return resume_state


@torch.no_grad()
def upcast_optimizer_state(optim: AdamATan2):
    for state in optim.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                state[key] = value.to(torch.float32)


@torch.no_grad()
def reset_optimizer_ema_to_params(optim: AdamATan2):
    for group in optim.param_groups:
        if group.get("ema") is None:
            continue
        for param in group["params"]:
            state = optim.state[param]
            if "param_ema" in state:
                state["param_ema"] = param.detach().clone().to(torch.float32)


def sharded_checkpoint_id(checkpoint_path: str, tag: str) -> str:
    return os.path.join(checkpoint_path, f"fsdp2_{tag}")


def unsharded_checkpoint_path(checkpoint_path: str, tag: str) -> str:
    return os.path.join(checkpoint_path, f"unsharded_{tag}.pt")


def load_sharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str):
    assert config.resume_checkpoint_path is not None
    checkpoint_id = sharded_checkpoint_id(config.resume_checkpoint_path, tag)
    if not os.path.isdir(checkpoint_id):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_id}")

    model_state = train_state.model.state_dict()
    optim_state = get_optimizer_state_dict(train_state.model, train_state.optim)  # pyright: ignore[reportPrivateImportUsage]
    dcp.load({"model": model_state, "optim": optim_state}, checkpoint_id=checkpoint_id)
    set_state_dict(
        train_state.model,
        train_state.optim,
        model_state_dict=model_state,
        optim_state_dict=optim_state,
    )


def load_unsharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str, rank: int):
    assert config.resume_checkpoint_path is not None
    checkpoint_path = unsharded_checkpoint_path(config.resume_checkpoint_path, tag)
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    if dist.is_available() and dist.is_initialized():
        trace_print(config, rank, f"unsharded_load_begin path={checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if (config.distributed_strategy == "ddp" or rank == 0) else {}
        trace_print(config, rank, f"unsharded_load_end keys={list(checkpoint.keys()) if checkpoint else []}")
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=config.distributed_strategy != "ddp",
        )
        trace_print(config, rank, "set_state_dict_begin")
        set_state_dict(
            train_state.model,
            train_state.optim,
            model_state_dict=checkpoint.get("model", {}),
            optim_state_dict=checkpoint.get("optim", {}),
            options=options,
        )
        trace_print(config, rank, "set_state_dict_end")
    else:
        trace_print(config, rank, f"unsharded_load_begin path={checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trace_print(config, rank, f"unsharded_load_end keys={list(checkpoint.keys())}")
        trace_print(config, rank, "model_load_state_dict_begin")
        train_state.model.load_state_dict(checkpoint["model"])
        trace_print(config, rank, "model_load_state_dict_end")
        trace_print(config, rank, "optim_load_state_dict_begin")
        train_state.optim.load_state_dict(checkpoint["optim"])
        trace_print(config, rank, "optim_load_state_dict_end")


def save_sharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str):
    assert config.checkpoint_path is not None
    if dist.is_available() and dist.is_initialized():
        dcp.save({"model": train_state.model.state_dict(), "optim": get_optimizer_state_dict(train_state.model, train_state.optim)},  # pyright: ignore[reportPrivateImportUsage]
                 checkpoint_id=sharded_checkpoint_id(config.checkpoint_path, tag))
    else:
        torch.save({"model": train_state.model.state_dict(), "optim": train_state.optim.state_dict()}, os.path.join(config.checkpoint_path, f"{tag}.pt"))


def save_unsharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str, rank: int):
    assert config.checkpoint_path is not None
    if dist.is_available() and dist.is_initialized():
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state = get_model_state_dict(train_state.model, options=options)
        optim_state = get_optimizer_state_dict(train_state.model, train_state.optim, options=options)  # pyright: ignore[reportPrivateImportUsage]
        if rank == 0:
            torch.save(
                {"model": model_state, "optim": optim_state},
                unsharded_checkpoint_path(config.checkpoint_path, tag),
            )
    else:
        torch.save(
            {"model": train_state.model.state_dict(), "optim": train_state.optim.state_dict()},
            unsharded_checkpoint_path(config.checkpoint_path, tag),
        )


@torch.compile(dynamic=False)
def forward_backward_batch(train_state: TrainState, batch: dict[str, Tensor], loss_scale: Tensor, **kwargs):
    device_type = batch["inputs"].device.type
    use_autocast = (
        (device_type in ("mps", "cpu") and train_state.fwd_bwd_dtype != torch.float32)
        or (device_type == "cuda" and train_state.use_cuda_autocast)
    )
    with torch.autocast(device_type=device_type, dtype=train_state.fwd_bwd_dtype, enabled=use_autocast, cache_enabled=False):
        train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    (loss * loss_scale).backward()
    return metrics


def forward_backward_batch_uncompiled(train_state: TrainState, batch: dict[str, Tensor], loss_scale: Tensor, **kwargs):
    device_type = batch["inputs"].device.type
    use_autocast = (
        (device_type in ("mps", "cpu") and train_state.fwd_bwd_dtype != torch.float32)
        or (device_type == "cuda" and train_state.use_cuda_autocast)
    )
    with torch.autocast(device_type=device_type, dtype=train_state.fwd_bwd_dtype, enabled=use_autocast, cache_enabled=False):
        train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    (loss * loss_scale).backward()
    return metrics


def train_batch(train_state: TrainState, batch: dict[str, Tensor], **kwargs):
    metrics = forward_backward_batch(train_state, batch, torch.tensor(1.0, device=batch["inputs"].device), **kwargs)
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics


def train_batch_uncompiled(train_state: TrainState, batch: dict[str, Tensor], **kwargs):
    metrics = forward_backward_batch_uncompiled(train_state, batch, torch.tensor(1.0, device=batch["inputs"].device), **kwargs)
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics


def _add_metrics(total_metrics: Optional[dict[str, tuple[Tensor, Tensor]]], metrics: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor]]:
    if total_metrics is None:
        return {name: (value_sum.detach(), divisor.detach()) for name, (value_sum, divisor) in metrics.items()}

    for name, (value_sum, divisor) in metrics.items():
        total_sum, total_divisor = total_metrics[name]
        total_metrics[name] = (total_sum + value_sum.detach(), total_divisor + divisor.detach())
    return total_metrics


def _supervised_token_count(config: PretrainConfig, rank: int, batch: dict[str, Tensor]) -> Tensor:
    count = (batch["labels"] != IGNORE_LABEL_ID).sum().to(torch.float32)
    if dist.is_available() and dist.is_initialized():
        trace_print(config, rank, f"supervised_count_all_reduce_begin local={count.item()}")
        dist.all_reduce(count, op=dist.ReduceOp.AVG)
        trace_print(config, rank, f"supervised_count_all_reduce_end avg={count.item()}")
    return count


def train_accumulated_batches(
    config: PretrainConfig,
    rank: int,
    train_state: TrainState,
    batches: list[dict[str, Tensor]],
    use_compiled: bool,
    zero_grad_after_step: bool = True,
    **kwargs,
) -> dict[str, tuple[Tensor, Tensor]]:
    trace_print(config, rank, f"train_accumulated_begin step={train_state.step} microbatches={len(batches)} compiled={use_compiled}")
    supervised_counts = [_supervised_token_count(config, rank, batch) for batch in batches]
    total_supervised = torch.stack(supervised_counts).sum().clamp_min(1.0)
    backward_step = forward_backward_batch if use_compiled else forward_backward_batch_uncompiled

    trace_print(config, rank, f"zero_grad_begin step={train_state.step}")
    train_state.optim.zero_grad()
    trace_print(config, rank, f"zero_grad_end step={train_state.step}")
    metrics = None
    for microbatch_idx, (batch, supervised_count) in enumerate(zip(batches, supervised_counts), start=1):
        loss_scale = supervised_count / total_supervised
        trace_print(config, rank, f"forward_backward_begin step={train_state.step} microbatch={microbatch_idx} loss_scale={loss_scale.item()}")
        metrics = _add_metrics(metrics, backward_step(train_state, batch, loss_scale, **kwargs))
        trace_print(config, rank, f"forward_backward_end step={train_state.step} microbatch={microbatch_idx}")

    trace_print(config, rank, f"optim_step_begin step={train_state.step}")
    train_state.optim.step()
    trace_print(config, rank, f"optim_step_end step={train_state.step}")
    if zero_grad_after_step:
        trace_print(config, rank, f"zero_grad_after_step_begin step={train_state.step}")
        train_state.optim.zero_grad()
        trace_print(config, rank, f"zero_grad_after_step_end step={train_state.step}")
    assert metrics is not None
    trace_print(config, rank, f"train_accumulated_end step={train_state.step}")
    return metrics


@torch.inference_mode()
def reduce_metrics(local_metrics: dict[str, Tensor], prefix: str):
    metric_keys = list(sorted(local_metrics.keys()))  # Sort keys to guarantee all processes use the same order.
    # Reduce and reconstruct
    metric_values = torch.stack([local_metrics[k][0] for k in metric_keys] + [local_metrics[k][1] for k in metric_keys])
    if dist.is_available() and dist.is_initialized():
        dist.reduce(metric_values, dst=0)
    # Split and normalize
    metrics, metrics_div = metric_values.chunk(2, dim=-1)
    metrics = (metrics / metrics_div).cpu().numpy().tolist()
    return {prefix + name: metrics[idx] for idx, name in enumerate(metric_keys)}


def save_code_and_config(config: PretrainConfig, train_metadata: V1DatasetMeta):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    with open(os.path.join(config.checkpoint_path, "train_metadata.yaml"), "wt") as f:
        yaml.dump(train_metadata.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def save_checkpoint_metadata(
    config: PretrainConfig,
    train_state: TrainState,
    tag: str,
    epoch: int,
    batch_in_epoch: int,
    rank: int,
    checkpoint_kind: str,
):
    if config.checkpoint_path is None or rank != 0:
        return

    metadata = {
        "tag": tag,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_format": config.checkpoint_format,
        "step": train_state.step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_batch_size": config.global_batch_size,
        "data_path": config.data.path,
        "seed": config.seed,
    }
    with open(os.path.join(config.checkpoint_path, f"checkpoint_state_{tag}.json"), "wt") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def save_train_checkpoint(
    config: PretrainConfig,
    train_state: TrainState,
    tag: str,
    epoch: int,
    batch_in_epoch: int,
    rank: int,
    checkpoint_kind: str = "regular",
):
    if config.checkpoint_path is None:
        return

    if config.checkpoint_format == "sharded":
        save_sharded_train_state(config, train_state, tag)
    elif config.checkpoint_format == "unsharded":
        save_unsharded_train_state(config, train_state, tag, rank)
    else:
        raise ValueError(f"Unsupported checkpoint_format: {config.checkpoint_format}")

    # Save carry on all ranks
    torch.save(train_state.carry, os.path.join(config.checkpoint_path, f"carry_{tag}.{rank}.pt"))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    save_checkpoint_metadata(config, train_state, tag, epoch, batch_in_epoch, rank, checkpoint_kind)


def _ephemeral_step_from_name(name: str) -> Optional[int]:
    patterns = (
        r"^fsdp2_ephemeral_step_(\d+)$",
        r"^unsharded_ephemeral_step_(\d+)\.pt$",
        r"^carry_ephemeral_step_(\d+)\.\d+\.pt$",
        r"^checkpoint_state_ephemeral_step_(\d+)\.json$",
        r"^ephemeral_step_(\d+)\.pt$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match is not None:
            return int(match.group(1))
    return None


def remove_stale_ephemeral_checkpoints(config: PretrainConfig, keep_tag: str, rank: int):
    if config.checkpoint_path is None:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        keep_step: Optional[int] = None
        keep_match = re.fullmatch(r"ephemeral_step_(\d+)", keep_tag)
        if keep_match is not None:
            keep_step = int(keep_match.group(1))

        removed_steps: set[int] = set()
        for entry in os.scandir(config.checkpoint_path):
            step = _ephemeral_step_from_name(entry.name)
            if step is None or step == keep_step:
                continue
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
            removed_steps.add(step)

        if removed_steps:
            print(
                f"Removed stale ephemeral checkpoints: {', '.join(f'ephemeral_step_{step}' for step in sorted(removed_steps))}",
                flush=True,
            )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_synced_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{Path(config.data.path).stem.capitalize()} HLM-torch"
        if config.run_name is None:
            config.run_name = os.environ.get("MLP_TASK_NAME", f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}")  # pyright: ignore[reportPrivateImportUsage]
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore


def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in batch.items()}


def format_mib(value: int) -> str:
    return f"{value / 1024 / 1024:.3f} MiB"


def format_memory_stats(stats: dict[str, int]) -> str:
    return " ".join(f"{name}={format_mib(value)}" for name, value in stats.items())


def maybe_log_memory(step: int, label: str, device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    stats = memory_stats_for_device(device)
    if not stats:
        return
    print(f"[{device.type} memory] step={step} {label}: {format_memory_stats(stats)}", flush=True)


def maybe_empty_cache(step: int, device: torch.device, interval: int) -> None:
    if interval <= 0 or step % interval != 0:
        return
    before = memory_stats_for_device(device)
    empty_accelerator_cache(device)
    synchronize_device(device)
    after = memory_stats_for_device(device)
    if not before and not after:
        return
    changes = []
    for name in sorted(before.keys() | after.keys()):
        changes.append(f"{name} {format_mib(before.get(name, 0))}->{format_mib(after.get(name, 0))}")
    print(f"[{device.type} empty_cache] step={step}: {' '.join(changes)}", flush=True)


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0
    requested_accelerator = OmegaConf.select(hydra_config, "accelerator_type", default="sm100")

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        if requested_accelerator in ("mps", "cpu", "none"):
            raise RuntimeError(f"accelerator_type={requested_accelerator} supports single-process training only.")
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(DEVICE_ID)

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK)
    set_accelerator_type(config.accelerator_type)
    device = torch_device_for_accelerator(config.accelerator_type, local_rank=DEVICE_ID)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # --- Training
    trace_print(config, RANK, "init_train_begin")
    train_state, train_loader, train_metadata = init_train(config, rank=RANK, world_size=WORLD_SIZE, device=device)
    trace_print(config, RANK, "init_train_end")
    resume_state = load_train_checkpoint(config, train_state, rank=RANK)
    start_epoch = 1
    skip_batches = 0
    if resume_state is not None:
        start_epoch = resume_state.start_epoch
        skip_batches = resume_state.skip_batches
        trace_print(config, RANK, f"set_epoch_begin epoch_index={start_epoch - 1}")
        train_loader.dataset.set_epoch(start_epoch - 1)
        trace_print(config, RANK, f"set_epoch_end epoch_index={start_epoch - 1}")
        trace_print(config, RANK, f"set_start_batch_begin skip_batches={skip_batches}")
        train_loader.dataset.set_start_batch(skip_batches)
        trace_print(config, RANK, f"set_start_batch_end skip_batches={skip_batches}")
        if RANK == 0:
            print(
                f"Resumed from {config.resume_checkpoint_path} ({config.checkpoint_format}:{resume_state.tag}): "
                f"step={train_state.step}, start_epoch={start_epoch}, skip_batches={skip_batches}",
                flush=True,
            )

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)

        wandb.init(
            project=config.project_name,
            name=config.run_name,
            id=config.wandb_run_id,
            resume=config.wandb_resume,
            config=config.model_dump() | {"train_metadata": train_metadata.model_dump()},
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        num_params = sum(x.numel() for x in train_state.model.parameters())
        if resume_state is None:
            wandb.log({"num_params": num_params}, step=0)
        else:
            wandb.run.summary["num_params"] = num_params  # type: ignore[union-attr]
        save_code_and_config(config, train_metadata)

    # Training Loop
    for epoch in range(start_epoch, config.epochs + 1):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")
        trace_print(config, RANK, f"epoch_begin epoch={epoch}")

        # ############ Train Iter
        train_state.model.train()
        accumulation_batches: list[dict[str, Tensor]] = []
        batch_start = skip_batches + 1 if skip_batches > 0 else 1
        trace_print(config, RANK, f"dataloader_iter_begin epoch={epoch} batch_start={batch_start}")
        for batch_in_epoch, (batch, batch_info) in enumerate(train_loader, start=batch_start):
            if config.resume_trace and batch_in_epoch == batch_start:
                trace_print(config, RANK, f"first_batch_yielded batch_in_epoch={batch_in_epoch}")
            batch = move_batch_to_device(batch, device)
            if config.resume_trace and batch_in_epoch == batch_start:
                trace_print(config, RANK, f"first_batch_moved batch_in_epoch={batch_in_epoch}")
            accumulation_batches.append(batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in batch_info.items()})
            if len(accumulation_batches) < config.gradient_accumulation_steps:
                continue

            train_state.step += 1            
            lr = update_lr(config, train_state)
            trace_print(config, RANK, f"optimizer_step_start step={train_state.step} batch_in_epoch={batch_in_epoch} lr={lr}")
            # Extra train arguments (such as BP warmup etc.)
            train_extra_args = compute_train_extra_args(train_state.model, train_state)
            trace_print(config, RANK, f"train_extra_args step={train_state.step} {train_extra_args}")
            maybe_log_memory(
                train_state.step,
                "before_train",
                device,
                config.memory_log_interval > 0 and train_state.step % config.memory_log_interval == 0,
            )
            metrics = train_accumulated_batches(config, RANK, train_state, accumulation_batches, config.compile_train_batch, **train_extra_args)
            accumulation_batches = []
            maybe_log_memory(
                train_state.step,
                "after_train",
                device,
                config.memory_log_interval > 0 and train_state.step % config.memory_log_interval == 0,
            )
            maybe_empty_cache(train_state.step, device, config.empty_cache_interval)

            if train_state.step % config.log_interval == 0:
                trace_print(config, RANK, f"reduce_metrics_begin step={train_state.step}")
                metrics = reduce_metrics(metrics, prefix="train/")
                trace_print(config, RANK, f"reduce_metrics_end step={train_state.step}")
                if RANK == 0:
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    trace_print(config, RANK, f"wandb_log_begin step={train_state.step}")
                    wandb.log(metrics | train_extra_args | {"train/lr": lr}, step=train_state.step)
                    trace_print(config, RANK, f"wandb_log_end step={train_state.step}")

            del metrics

            saved_regular_step_checkpoint = False
            if config.checkpoint_step_interval is not None and train_state.step % config.checkpoint_step_interval == 0:
                save_train_checkpoint(config, train_state, f"step_{train_state.step}", epoch, batch_in_epoch, RANK)
                saved_regular_step_checkpoint = True

            if config.ephemeral_checkpoint_step_interval is not None and train_state.step % config.ephemeral_checkpoint_step_interval == 0:
                if saved_regular_step_checkpoint:
                    remove_stale_ephemeral_checkpoints(config, f"step_{train_state.step}", RANK)
                else:
                    ephemeral_tag = f"ephemeral_step_{train_state.step}"
                    save_train_checkpoint(config, train_state, ephemeral_tag, epoch, batch_in_epoch, RANK, checkpoint_kind="ephemeral")
                    remove_stale_ephemeral_checkpoints(config, ephemeral_tag, RANK)

        skip_batches = 0

        ############ EVAL STACK: TBD TODO

        ############ Checkpointing
        if (epoch % config.checkpoint_interval == 0) or (epoch == config.epochs):
            save_train_checkpoint(config, train_state, f"epoch_{epoch}", epoch, 0, RANK)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
