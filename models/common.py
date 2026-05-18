from dataclasses import dataclass

from torch import Tensor
import torch.nn.functional as F


IGNORE_LABEL_ID = -100


def trunc_normal_init_(tensor: Tensor, std: float = 1.0):
    """Fast approximate truncated normal initialization. Fairly accurate."""

    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)


def packing_sequence_sum(x: Tensor, cu_seqlens: Tensor):
    c = F.pad(x.cumsum(0), (1, 0))
    return c[cu_seqlens[1:]] - c[cu_seqlens[:-1]]


@dataclass
class WrappedTensor:
    value: Tensor


def wrap_tensor(value: Tensor) -> WrappedTensor:
    """Wrap a Tensor, so that FSDP2 won't see this Tensor, and do preprocessing such as moving to device and casting."""
    return WrappedTensor(value)


def unwrap_tensor(wrapped: Tensor | WrappedTensor) -> Tensor:
    return wrapped.value if isinstance(wrapped, WrappedTensor) else wrapped
