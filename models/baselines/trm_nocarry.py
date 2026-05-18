from typing import Tuple, Optional

import torch
from torch import nn
from torch import Tensor

from models.common import trunc_normal_init_
from models.transformer import Cache, TransformerConfig
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModelRecurrentBlock


class TinyRecursiveModelConfig(TransformerConfig):
    half_layers: bool = False

    H_cycles: int
    L_cycles: int

    H_bp_steps: int
    L_bp_steps: int


class TinyRecursiveModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = TinyRecursiveModelConfig(**config_dict)
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2."
            config.n_layers //= 2

        # Reasoning Layers
        self.L_level = HierarchicalReasoningModelRecurrentBlock(config)

        # Config
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.H_bp_steps = config.H_bp_steps
        self.L_bp_steps = config.L_bp_steps

        self.hidden_size = config.hidden_size
        self.head_hint = self.L_level.core.head_hint  # Hint for LMHead init (inherit from H)
        
        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0), persistent=True)  # NOTE: hardcoded dtype.
        
        # Create cache function
        self.create_cache = lambda **kwargs: dict(H=[self.L_level.create_cache(**kwargs) for _i in range(self.H_cycles)],
                                                  L=[self.L_level.create_cache(**kwargs) for _i in range(self.H_cycles * self.L_cycles)])

    def forward(self, carry: None, x: torch.Tensor, cache: Optional[dict[str, list[list[Cache]]]] = None, **seq_info) -> Tuple[None, torch.Tensor]:
        z_H, z_L = x, self.zL_init

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - self.L_bp_steps)):
                    z_L = self.L_level(z_L, z_H, **seq_info, cache=cache["L"][k] if cache is not None else None)
            
            with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= self.H_cycles - self.H_bp_steps)):
                z_H = self.L_level(z_H, z_L, **seq_info, cache=cache["H"][i] if cache is not None else None)

        return None, z_H

    def compute_train_extra_args(self, train_state):
        return {}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
