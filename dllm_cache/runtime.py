"""Small, optional-backend runtime helpers.

The project supports CPU and CUDA without requiring ``torch_npu`` to be
installed.  Importing this module therefore never makes the NPU package a
hard dependency; requesting an NPU explicitly still produces an actionable
error when the package or device is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

try:  # torch_npu registers the ``npu`` torch device at import time.
    import torch_npu as _torch_npu  # noqa: F401
except Exception as exc:  # Keep CPU/CUDA usable with a missing/broken plugin.
    _torch_npu = None
    _TORCH_NPU_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


DeviceSpec = Union[str, torch.device, "RuntimeContext"]
DTypeSpec = Union[str, torch.dtype, None]


@dataclass(frozen=True)
class MemoryStats:
    """Memory usage for one accelerator device, in bytes.

    CPU runtimes report zero for allocator counters and ``None`` for total
    device memory because PyTorch does not expose equivalent process-local
    allocator statistics for CPU.
    """

    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    total_bytes: Optional[int]


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved execution backend and concrete torch device."""

    device: torch.device
    backend: str

    @property
    def index(self) -> Optional[int]:
        return self.device.index

    @property
    def is_accelerator(self) -> bool:
        return self.backend in {"npu", "cuda"}

    def synchronize(self) -> None:
        """Wait for queued work on this runtime's device."""

        if self.backend == "npu":
            torch.npu.synchronize(self.device)
        elif self.backend == "cuda":
            torch.cuda.synchronize(self.device)

    def empty_cache(self) -> None:
        """Release unused cached accelerator memory; CPU is a no-op."""

        if self.backend == "npu":
            torch.npu.empty_cache()
        elif self.backend == "cuda":
            torch.cuda.empty_cache()

    def reset_peak_memory_stats(self) -> None:
        """Reset peak allocator counters for the selected accelerator."""

        if self.backend == "npu":
            torch.npu.reset_peak_memory_stats(self.device)
        elif self.backend == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats(self) -> MemoryStats:
        """Return current and peak allocator counters for this device."""

        if self.backend == "cpu":
            return MemoryStats(0, 0, 0, 0, None)

        module = torch.npu if self.backend == "npu" else torch.cuda
        properties = module.get_device_properties(self.device)
        return MemoryStats(
            allocated_bytes=int(module.memory_allocated(self.device)),
            reserved_bytes=int(module.memory_reserved(self.device)),
            max_allocated_bytes=int(module.max_memory_allocated(self.device)),
            max_reserved_bytes=int(module.max_memory_reserved(self.device)),
            total_bytes=int(properties.total_memory),
        )


def _npu_available() -> bool:
    npu = getattr(torch, "npu", None)
    if _torch_npu is None or npu is None:
        return False
    try:
        return bool(npu.is_available()) and int(npu.device_count()) > 0
    except Exception:
        return False


def _validate_index(backend: str, index: Optional[int], count: int) -> int:
    if index is None:
        if backend == "npu":
            return int(torch.npu.current_device())
        return int(torch.cuda.current_device())
    if index < 0 or index >= count:
        raise ValueError(
            f"{backend} device index {index} is out of range; "
            f"available indices are 0..{count - 1}"
        )
    return index


def _parse_device(device: Union[str, torch.device]) -> tuple[str, Optional[int]]:
    if isinstance(device, torch.device):
        raw = str(device)
    elif isinstance(device, str):
        raw = device.strip().lower()
    else:
        raise TypeError(
            "device must be 'auto', a cpu/cuda/npu device string, "
            f"or torch.device; got {type(device).__name__}"
        )

    if raw == "auto":
        return "auto", None
    if raw == "cpu":
        return "cpu", None

    backend, separator, raw_index = raw.partition(":")
    if backend not in {"npu", "cuda"}:
        raise ValueError(
            f"unsupported device {raw!r}; expected auto, cpu, npu[:N], or cuda[:N]"
        )
    if not separator:
        return backend, None
    if not raw_index or not raw_index.isdigit():
        raise ValueError(f"invalid {backend} device index in {raw!r}")
    return backend, int(raw_index)


def resolve_runtime(device: DeviceSpec = "auto") -> RuntimeContext:
    """Resolve ``auto|npu[:N]|cuda[:N]|cpu`` to a concrete runtime.

    ``auto`` has a fixed NPU -> CUDA -> CPU priority.  Explicit accelerator
    requests never silently fall back to another backend.
    """

    if isinstance(device, RuntimeContext):
        return device

    backend, index = _parse_device(device)
    if backend == "auto":
        if _npu_available():
            backend = "npu"
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            backend = "cuda"
        else:
            return RuntimeContext(torch.device("cpu"), "cpu")

    if backend == "cpu":
        return RuntimeContext(torch.device("cpu"), "cpu")

    if backend == "npu":
        if _torch_npu is None:
            detail = (
                f": {_TORCH_NPU_IMPORT_ERROR}"
                if _TORCH_NPU_IMPORT_ERROR is not None
                else ""
            )
            raise RuntimeError(
                "NPU was requested, but torch_npu could not be imported" + detail
            )
        if not _npu_available():
            raise RuntimeError(
                "NPU was requested, but torch.npu reports no available devices; "
                "check the CANN environment and ASCEND_RT_VISIBLE_DEVICES"
            )
        concrete_index = _validate_index("npu", index, int(torch.npu.device_count()))
        return RuntimeContext(torch.device(f"npu:{concrete_index}"), "npu")

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError(
            "CUDA was requested, but torch.cuda reports no available devices; "
            "check the CUDA runtime and CUDA_VISIBLE_DEVICES"
        )
    concrete_index = _validate_index("cuda", index, int(torch.cuda.device_count()))
    return RuntimeContext(torch.device(f"cuda:{concrete_index}"), "cuda")


def resolve_dtype(
    dtype: DTypeSpec = "auto", runtime: Optional[DeviceSpec] = None
) -> torch.dtype:
    """Resolve a public dtype option to a ``torch.dtype``.

    Automatic dtype is BF16 on NPU/CUDA and FP32 on CPU.  ``runtime`` may be a
    pre-resolved context or any device value accepted by :func:`resolve_runtime`.
    """

    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        dtype = "auto"
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string or torch.dtype; got {type(dtype).__name__}")

    normalized = dtype.strip().lower().replace("torch.", "")
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if normalized == "auto":
        context = resolve_runtime("auto" if runtime is None else runtime)
        return torch.float32 if context.backend == "cpu" else torch.bfloat16
    if normalized not in aliases:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected auto, bfloat16, float16, or float32"
        )
    return aliases[normalized]


__all__ = [
    "MemoryStats",
    "RuntimeContext",
    "resolve_dtype",
    "resolve_runtime",
]
