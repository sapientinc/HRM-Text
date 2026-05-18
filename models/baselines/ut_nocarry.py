from typing import Optional
from contextlib import nullcontext

import torch
from torch import nn, Tensor

from models.common import trunc_normal_init_
from models.transformer import TransformerConfig, Cache
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModelRecurrentBlock


class UniversalTransformerConfig(TransformerConfig):
    cycles: int
    bp_cycles: int


class UniversalTransformer(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = UniversalTransformerConfig(**config_dict)

        # Reasoning Layers
        self.L_level = HierarchicalReasoningModelRecurrentBlock(TransformerConfig(**config.model_dump()))

        # Config
        self.cycles = config.cycles
        self.bp_cycles = config.bp_cycles

        self.head_hint = self.L_level.core.head_hint  # Hint for LMHead init (inherit from H)

        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0), persistent=True)  # NOTE: hardcoded dtype.
        
        # Create cache function
        self.create_cache = lambda **kwargs: [self.L_level.create_cache(**kwargs) for _i in range(self.cycles)]

    def forward(self, carry: None, x: Tensor, cache: Optional[list[list[Cache]]] = None, **seq_info) -> tuple[None, Tensor]:
        z = self.zL_init

        for i in range(self.cycles):
            with (nullcontext if i >= self.cycles - self.bp_cycles else torch.no_grad)():
                z = self.L_level(z, x, cache=cache[i] if cache is not None else None, **seq_info)

        return None, z

    def compute_train_extra_args(self, train_state):
        return {}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
