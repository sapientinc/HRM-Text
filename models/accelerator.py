from typing import Literal

import torch


AcceleratorType = Literal["sm90", "sm100", "rocm", "mps", "cpu", "none"]

_accelerator_type: AcceleratorType = "sm100"

__all__ = [
    "AcceleratorType",
    "empty_accelerator_cache",
    "get_accelerator_type",
    "is_accelerator_available",
    "memory_stats_for_device",
    "set_accelerator_type",
    "synchronize_device",
    "torch_device_for_accelerator",
    "validate_accelerator_available",
]


def set_accelerator_type(accelerator_type: AcceleratorType) -> None:
    global _accelerator_type
    _accelerator_type = accelerator_type


def get_accelerator_type() -> AcceleratorType:
    return _accelerator_type


def _expected_cuda_major(accelerator_type: AcceleratorType) -> int:
    if accelerator_type == "sm90":
        return 9
    if accelerator_type == "sm100":
        return 10
    raise ValueError(f"Unsupported CUDA accelerator_type: {accelerator_type}")


def is_accelerator_available(accelerator_type: AcceleratorType, local_rank: int = 0) -> bool:
    if accelerator_type in ("sm90", "sm100"):
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            return False
        major, _minor = torch.cuda.get_device_capability(local_rank)
        return major == _expected_cuda_major(accelerator_type)
    if accelerator_type == "rocm":
        return torch.version.hip is not None and torch.cuda.is_available() and local_rank < torch.cuda.device_count()
    if accelerator_type == "mps":
        return torch.backends.mps.is_available()
    if accelerator_type in ("cpu", "none"):
        return True
    raise ValueError(f"Unsupported accelerator_type: {accelerator_type}")


def validate_accelerator_available(accelerator_type: AcceleratorType, local_rank: int = 0) -> None:
    if is_accelerator_available(accelerator_type, local_rank=local_rank):
        return

    if accelerator_type in ("sm90", "sm100"):
        expected_major = _expected_cuda_major(accelerator_type)
        cuda_state = "available" if torch.cuda.is_available() else "unavailable"
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        capability = None
        if torch.cuda.is_available() and local_rank < device_count:
            capability = torch.cuda.get_device_capability(local_rank)
        raise RuntimeError(
            f"accelerator_type={accelerator_type} requires CUDA device capability {expected_major}.x "
            f"at local_rank={local_rank}; CUDA is {cuda_state}, device_count={device_count}, "
            f"detected_capability={capability}."
        )

    if accelerator_type == "rocm":
        hip_version = torch.version.hip
        cuda_state = "available" if torch.cuda.is_available() else "unavailable"
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        raise RuntimeError(
            f"accelerator_type=rocm requires a HIP/ROCm torch build with a device at "
            f"local_rank={local_rank}; torch.version.hip={hip_version}, torch.cuda is {cuda_state}, "
            f"device_count={device_count}."
        )

    if accelerator_type == "mps":
        raise RuntimeError(
            "accelerator_type=mps was requested, but torch.backends.mps.is_available() is false. "
            f"torch.backends.mps.is_built()={torch.backends.mps.is_built()}."
        )

    raise ValueError(f"Unsupported accelerator_type: {accelerator_type}")


def torch_device_for_accelerator(
    accelerator_type: AcceleratorType,
    local_rank: int = 0,
    validate: bool = True,
) -> torch.device:
    if validate:
        validate_accelerator_available(accelerator_type, local_rank=local_rank)

    if accelerator_type in ("sm90", "sm100", "rocm"):
        return torch.device("cuda", local_rank)
    if accelerator_type == "mps":
        return torch.device("mps")
    if accelerator_type in ("cpu", "none"):
        return torch.device("cpu")
    raise ValueError(f"Unsupported accelerator_type: {accelerator_type}")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _cpu_memory_stats() -> dict[str, int]:
    try:
        import psutil  # type: ignore[import-not-found]

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        import platform
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Linux":
            # Linux reports KiB; macOS reports bytes.
            rss *= 1024
    return {"rss": rss}


def memory_stats_for_device(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        synchronize_device(device)
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device),
        }
    if device.type == "mps":
        synchronize_device(device)
        return {
            "allocated": torch.mps.current_allocated_memory(),
            "reserved": torch.mps.driver_allocated_memory(),
        }
    if device.type == "cpu":
        return _cpu_memory_stats()
    return {}


def empty_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
