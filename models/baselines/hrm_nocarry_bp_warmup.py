from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn
from torch import Tensor

from models.common import trunc_normal_init_
from models.transformer import Transformer, Cache, TransformerConfig


class HierarchicalReasoningModelConfig(TransformerConfig):
    half_layers: bool = False

    H_cycles: int
    L_cycles: int

    bp_warmup_ratio: float = 0.0
    bp_min_steps: int = 2
    bp_max_steps: int = 5

    # Change some Transformer config of H-level
    # TODO: Try asymmetric H and L module, such as different size, hidden dims, architecture, attention type, etc.
    H_override: Dict[str, Any] = {}


class HierarchicalReasoningModelRecurrentBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.core = Transformer(config)

        # Create cache function
        self.create_cache = self.core.create_cache

    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        # Input injection (add)
        # TODO: Try better alternatives, such as GRU / gating in the following papers
        # Alternatively, "fixed" gating that does not depend on hidden state is also worth trying
        # E.g. only depends on position and index of hidden_states dimension
        # https://arxiv.org/pdf/1910.06764
        # https://arxiv.org/pdf/2202.10447
        
        # TODO: Asymmetric fusion is also worth trying. assign different number of tokens to H and L.
        return self.core(hidden_states + input_injection, **kwargs)


class HierarchicalReasoningModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = HierarchicalReasoningModelConfig(**config_dict)
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2."
            config.n_layers //= 2

        # Reasoning Layers
        # TODO: Asymmetric.
        self.H_level = HierarchicalReasoningModelRecurrentBlock(TransformerConfig(**(config.model_dump() | config.H_override)))
        self.L_level = HierarchicalReasoningModelRecurrentBlock(config)

        # Config
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bp_warmup_ratio = config.bp_warmup_ratio
        self.bp_min_steps = config.bp_min_steps
        self.bp_max_steps = config.bp_max_steps

        self.hidden_size = config.hidden_size
        self.head_hint = self.H_level.core.head_hint  # Hint for LMHead init (inherit from H)
        
        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0), persistent=True)  # NOTE: hardcoded dtype.
        
        # Create cache function
        self.create_cache = lambda **kwargs: dict(H=[self.H_level.create_cache(**kwargs) for _i in range(self.H_cycles)],
                                                  L=[self.L_level.create_cache(**kwargs) for _i in range(self.H_cycles * self.L_cycles)])

    def forward(self, carry: None, x: torch.Tensor, cache: Optional[dict[str, list[list[Cache]]]] = None, bp_steps: int = 2, **seq_info) -> Tuple[None, torch.Tensor]:
        z_H, z_L = x, self.zL_init

        # Calculate H and L bp_steps
        # Priortize H, and at least 1 is allocated to L.
        H_bp_steps = min(self.H_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - L_bp_steps)):
                    z_L = self.L_level(z_L, z_H, **seq_info, cache=cache["L"][k] if cache is not None else None)
            
            with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)):
                z_H = self.H_level(z_H, z_L, **seq_info, cache=cache["H"][i] if cache is not None else None)

        return None, z_H

    def compute_train_extra_args(self, train_state: Any) -> dict[str, Any]:
        return dict(bp_steps=self.bp_min_steps + int(min(1, train_state.step / (train_state.total_steps * self.bp_warmup_ratio)) * (self.bp_max_steps - self.bp_min_steps)))

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
