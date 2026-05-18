import torch
from torch import Tensor

from models.transformer import Transformer, TransformerConfig

class TransformerWrapper(Transformer):
    def __init__(self, config_dict: dict) -> None:
        super().__init__(TransformerConfig(**config_dict))

    def forward(self, carry: None, x: Tensor, **kwargs) -> tuple[None, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return None, super().forward(x, **kwargs)

    def compute_train_extra_args(self, train_state):
        return {}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
