from typing import Any, Optional
from contextlib import nullcontext

import torch
from torch import nn, Tensor

from models.common import trunc_normal_init_
from models.transformer import Transformer, Cache, TransformerConfig


class RINSConfig(TransformerConfig):
    half_layers: bool = False

    L_cycles: int
    L_bp_cycles: int

    # Change some Transformer config of H-level
    H_override: dict[str, Any] = {}


class RINSRecurrentBlock(Transformer):
    """Backward compatible to earlier checkpoints."""
    def forward(self, hidden_states: Tensor, input_injection: Optional[Tensor], **kwargs) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().forward(hidden_states + input_injection if input_injection is not None else hidden_states, **kwargs)


class RINS(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = RINSConfig(**config_dict)
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2."
            config.n_layers //= 2

        # Reasoning Layers
        self.H_level = RINSRecurrentBlock(TransformerConfig(**(config.model_dump() | config.H_override)))
        self.L_level = RINSRecurrentBlock(config)

        # Config
        self.L_cycles = config.L_cycles
        self.L_bp_cycles = config.L_bp_cycles

        self.hidden_size = config.hidden_size
        self.head_hint = self.H_level.head_hint  # Hint for LMHead init (inherit from H)

        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0), persistent=True)  # NOTE: hardcoded dtype.
        
        # Create cache function
        self.create_cache = lambda **kwargs: dict(H=self.H_level.create_cache(**kwargs),
                                                  L=[self.L_level.create_cache(**kwargs) for _i in range(self.L_cycles)])

    def forward(self, carry: None, x: Tensor, cache: Optional[dict[str, list[list[Cache]]]] = None, **seq_info) -> tuple[None, torch.Tensor]:
        # Forward iterations (L)
        z_L = self.zL_init
        for i in range(self.L_cycles):
            with (nullcontext if i >= self.L_cycles - self.L_bp_cycles else torch.no_grad)():
                z_L = self.L_level(z_L, x, **seq_info, cache=cache["L"][i] if cache is not None else None)

        # Forward iterations (H)
        return None, self.H_level(z_L, None, **seq_info, cache=cache["H"] if cache is not None else None)

    def compute_train_extra_args(self, train_state):
        return {}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
