"""Compute policy: put GPU work on the GPU, and make it fast.

What runs where in this project:

* foundation-model inference (Chronos-2, TimesFM 3.0, TabPFN v3) - GPU
* LightGBM, imputation, extraction, harvesting - CPU (the AIFS/TIGGE jobs
  are downloads plus GRIB decoding; there is no inference on our side)

:func:`select_device` picks the CUDA device with the most free memory, so a
card occupied by someone else's job is avoided automatically;
:func:`configure_torch` turns on TF32 and cuDNN autotuning; :func:`autocast`
runs inference in bfloat16 on Ampere and newer.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    name: str
    free_gb: float
    total_gb: float
    bf16: bool


def gpu_inventory() -> list[DeviceInfo]:
    import torch

    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        major, _ = torch.cuda.get_device_capability(i)
        out.append(
            DeviceInfo(
                f"cuda:{i}", torch.cuda.get_device_name(i), free / 1e9, total / 1e9, major >= 8
            )
        )
    return out


def select_device(min_free_gb: float = 4.0) -> str:
    """The CUDA device with the most free memory, else CPU. Honours CUDA_VISIBLE_DEVICES."""
    devices = gpu_inventory()
    if not devices:
        logger.warning("No CUDA device visible; falling back to CPU")
        return "cpu"
    best = max(devices, key=lambda d: d.free_gb)
    if best.free_gb < min_free_gb:
        logger.warning("Best GPU %s has only %.1f GB free", best.name, best.free_gb)
    logger.info(
        "Using %s (%s, %.1f/%.1f GB free)", best.device, best.name, best.free_gb, best.total_gb
    )
    return best.device


def configure_torch(threads: int | None = None) -> None:
    """Fast defaults for inference: TF32 matmuls, cuDNN autotune, bounded CPU threads."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    n = threads or min(16, os.cpu_count() or 1)
    torch.set_num_threads(n)


def inference_dtype(device: str):
    import torch

    if (
        device.startswith("cuda")
        and torch.cuda.get_device_capability(int(device.split(":")[1]) if ":" in device else 0)[0]
        >= 8
    ):
        return torch.bfloat16
    return torch.float32


@contextlib.contextmanager
def autocast(device: str):
    """bfloat16 autocast on Ampere+ GPUs, no-op elsewhere."""
    import torch

    if device.startswith("cuda"):
        with torch.inference_mode(), torch.autocast("cuda", dtype=inference_dtype(device)):
            yield
    else:
        with torch.inference_mode():
            yield
