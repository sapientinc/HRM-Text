from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


class AdamATan2(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        # Optimizer parameters
        lr: float | Tensor = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        # Extra features
        ema: Optional[float] = None,
    ):
        # Initialize the Adam-atan2 optimizer
        if isinstance(lr, Tensor):
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "ema": ema
        }
        super().__init__(params, defaults)
        # Initialize state
        self._init_state()

    @torch.no_grad()
    def _init_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                # Step counter
                state["step"] = torch.tensor(0.0, dtype=torch.get_default_dtype())

                # Momentum
                if group["betas"][0] > 0:
                    state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

                # Extra features
                if group["ema"] is not None:
                    state["param_ema"] = torch.empty_like(p).copy_(p)

    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Perform a single optimization step."""
        assert closure is None, "Closure is not supported"

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                state = self.state[param]
                grad = param.grad

                # Weight decay update
                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                # Momentums
                if "exp_avg" in state:
                    state["exp_avg"].lerp_(grad, 1 - group["betas"][0])
                state["exp_avg_sq"].mul_(group["betas"][1]).addcmul_(grad, grad, value=1 - group["betas"][1])
                
                state["step"] += 1
                bias_correction1 = 1 - group["betas"][0] ** state["step"]
                bias_correction2 = 1 - group["betas"][1] ** state["step"]
                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = bias_correction2.sqrt()

                denom = state["exp_avg_sq"].sqrt() / bias_correction2_sqrt
                # AdamW-atan2
                if "exp_avg" in state:
                    param.add_(torch.atan2(state["exp_avg"], denom), alpha=-step_size)  # pyright: ignore[reportArgumentType]
                else:
                    param.add_(torch.atan2(grad, denom), alpha=-group["lr"])  # pyright: ignore[reportArgumentType]

                # [Extra features] EMA
                if "param_ema" in state:
                    state["param_ema"].lerp_(param, 1 - group["ema"])

    @torch.no_grad()
    def swap_ema(self):
        """Swap param buffer and EMA buffer for evaluation. Remember to swap back after evaluation."""
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if "param_ema" in state:
                    temp = torch.empty_like(param).copy_(param)
                    param.copy_(state["param_ema"])
                    state["param_ema"].copy_(temp)
