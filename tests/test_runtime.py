import pytest
import torch

from dllm_cache.runtime import MemoryStats, resolve_dtype, resolve_runtime


def test_cpu_runtime_is_a_noop_backend():
    runtime = resolve_runtime("cpu")
    assert runtime.backend == "cpu"
    assert runtime.device == torch.device("cpu")
    runtime.synchronize()
    runtime.empty_cache()
    runtime.reset_peak_memory_stats()
    assert runtime.memory_stats() == MemoryStats(0, 0, 0, 0, None)


def test_cpu_auto_dtype_and_explicit_aliases():
    runtime = resolve_runtime("cpu")
    assert resolve_dtype("auto", runtime) is torch.float32
    assert resolve_dtype("bf16", runtime) is torch.bfloat16
    assert resolve_dtype("float16", runtime) is torch.float16
    assert resolve_dtype(torch.float32, runtime) is torch.float32


@pytest.mark.parametrize("device", ["xpu", "npu:-1", "cuda:abc", "cpu:0"])
def test_invalid_device_spec_is_rejected(device):
    with pytest.raises((ValueError, RuntimeError)):
        resolve_runtime(device)


def test_explicit_unavailable_accelerator_does_not_fall_back():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        with pytest.raises(RuntimeError):
            resolve_runtime("npu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_runtime("cuda")


def test_auto_resolves_to_an_available_backend():
    runtime = resolve_runtime("auto")
    if hasattr(torch, "npu") and torch.npu.is_available():
        assert runtime.backend == "npu"
    elif torch.cuda.is_available():
        assert runtime.backend == "cuda"
    else:
        assert runtime.backend == "cpu"
