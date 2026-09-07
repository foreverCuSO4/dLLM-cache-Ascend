#!/usr/bin/env python3
"""Microbenchmark the dynamic operators used by dLLM-Cache on Ascend NPU.

The benchmark deliberately does not load a model.  It allocates tensors with
the hidden sizes/head layouts used by Dream-v0-7B and LLaDA-8B, then measures
the individual operations which are added by the cache path (selection,
indexing, concatenation and attention).  This makes it useful when a full
model profiler trace is too large or too noisy.

Example::

    python scripts/microbench_npu.py --model dream \
        --warmup 100 --repeats 200 \
        --output-json reports/microbench_dream_npu.json

Only ``npu:<index>`` devices are accepted.  Synchronization is performed
around every sample, so reported values are device execution latency rather
than host enqueue time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

try:  # Importing torch_npu registers the Ascend device backend.
    import torch_npu  # noqa: F401
    _TORCH_NPU_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised on non-Ascend hosts
    # Defer the error until ``main`` so ``--help`` and static tooling work on a
    # development machine that does not have CANN installed.
    _TORCH_NPU_IMPORT_ERROR = exc


MODEL_SHAPES = {
    # Values match the model configurations used by benchmark_text.py.  The
    # Dream model uses grouped-query attention (28 Q heads / 4 KV heads).
    "dream": {
        "hidden_size": 3584,
        "q_heads": 28,
        "kv_heads": 4,
        "head_dim": 128,
    },
    # LLaDA-8B follows the LLaMA-style 32-head layout.
    "llada": {
        "hidden_size": 4096,
        "q_heads": 32,
        "kv_heads": 32,
        "head_dim": 128,
    },
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SHAPES), default="dream")
    parser.add_argument("--device", default="npu:0", help="Ascend device, e.g. npu:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument(
        "--transfer-ratio",
        type=float,
        default=0.25,
        help="fraction of generation tokens selected by top-k",
    )
    # The requested protocol is warmup >= 100 and repeats >= 200.  Keeping
    # those as parser defaults also makes an accidental short run impossible.
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path for the JSON report",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.device.startswith("npu"):
        raise ValueError("--device must be npu:<index>; this benchmark is NPU-only")
    if args.batch_size < 1 or args.prompt_length < 1 or args.gen_length < 1:
        raise ValueError("batch size and sequence lengths must be positive")
    if args.warmup < 100:
        raise ValueError("--warmup must be at least 100")
    if args.repeats < 200:
        raise ValueError("--repeats must be at least 200")
    if not 0.0 <= args.transfer_ratio <= 1.0:
        raise ValueError("--transfer-ratio must be in [0, 1]")

    try:
        index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    except ValueError as exc:
        raise ValueError("--device must look like npu:0") from exc
    if index < 0:
        raise ValueError("NPU index must be non-negative")


def device_index(device_name: str) -> int:
    return int(device_name.split(":", 1)[1]) if ":" in device_name else 0


def sync() -> None:
    torch.npu.synchronize()


def memory_stats() -> dict[str, int]:
    """Return allocator counters as plain JSON-compatible integers."""

    # Different torch-npu releases expose the CUDA-compatible counters under
    # slightly different names.  The first branch is available in the tested
    # 2.4.x stack; the fallbacks keep the script useful on older CANN releases.
    def read(name: str) -> int:
        value = getattr(torch.npu, name, None)
        if value is None:
            return 0
        try:
            return int(value())
        except TypeError:
            return 0

    stats = {
        "current_allocated_bytes": read("memory_allocated"),
        "current_reserved_bytes": read("memory_reserved"),
        "peak_allocated_bytes": read("max_memory_allocated"),
        "peak_reserved_bytes": read("max_memory_reserved"),
    }
    try:
        stats["total_device_bytes"] = int(
            torch.npu.get_device_properties(torch.npu.current_device()).total_memory
        )
    except (AttributeError, RuntimeError):
        stats["total_device_bytes"] = 0
    return stats


def reset_peak_memory() -> None:
    reset = getattr(torch.npu, "reset_peak_memory_stats", None)
    if reset is not None:
        reset()


def digest_tensor(value: torch.Tensor) -> str:
    """Hash a small CPU copy of the final result for correctness checks."""

    # Hash at most 4 KiB; this avoids making the report itself a benchmark
    # bottleneck while still detecting accidental shape/value changes.
    flat = value.detach().reshape(-1)
    if flat.numel() > 1024:
        flat = flat[:1024]
    payload = flat.float().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def make_operations(args: argparse.Namespace) -> tuple[dict[str, Callable[[], torch.Tensor]], dict]:
    shape = MODEL_SHAPES[args.model]
    device = torch.device(f"npu:{device_index(args.device)}")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    batch = args.batch_size
    prompt = args.prompt_length
    generation = args.gen_length
    sequence = prompt + generation
    hidden = shape["hidden_size"]
    q_heads = shape["q_heads"]
    kv_heads = shape["kv_heads"]
    head_dim = shape["head_dim"]
    kv_hidden = kv_heads * head_dim
    transfer_tokens = max(1, min(generation, math.ceil(generation * args.transfer_ratio)))

    # Inputs are allocated once and reused.  In particular, the benchmark
    # should measure gather/scatter/attention and not random-number generation.
    features_a = torch.randn(
        (batch, generation, kv_hidden), device=device, dtype=dtype
    )
    features_b = torch.randn(
        (batch, generation, kv_hidden), device=device, dtype=dtype
    )
    scores = torch.randn((batch, generation), device=device, dtype=torch.float32)

    hidden_generation = torch.randn(
        (batch, generation, hidden), device=device, dtype=dtype
    )
    kv_generation_cache = torch.randn(
        (batch, generation, kv_hidden), device=device, dtype=dtype
    )
    hidden_generation_cache = torch.randn(
        (batch, generation, hidden), device=device, dtype=dtype
    )
    kv_src = torch.randn(
        (batch, transfer_tokens, kv_hidden), device=device, dtype=dtype
    )
    hidden_src = torch.randn(
        (batch, transfer_tokens, hidden), device=device, dtype=dtype
    )
    # One token index per selected generation position.  Expand over hidden or
    # projected KV dimensions as required by torch.gather/scatter.
    index_1d = torch.randperm(generation, device=device)[:transfer_tokens]
    hidden_index = index_1d.view(1, transfer_tokens, 1).expand(
        batch, transfer_tokens, hidden
    ).contiguous()
    kv_index = index_1d.view(1, transfer_tokens, 1).expand(
        batch, transfer_tokens, kv_hidden
    ).contiguous()
    topk = min(transfer_tokens, generation)

    prompt_kv_cache = torch.randn(
        (batch, prompt, kv_hidden), device=device, dtype=dtype
    )
    generation_kv_cache = torch.randn(
        (batch, generation, kv_hidden), device=device, dtype=dtype
    )
    prompt_hidden_cache = torch.randn(
        (batch, prompt, hidden), device=device, dtype=dtype
    )
    generation_hidden_cache = torch.randn(
        (batch, generation, hidden), device=device, dtype=dtype
    )
    # ``contiguous`` is measured on an intentionally non-contiguous transpose.
    contiguous_input = torch.randn(
        (batch, q_heads, sequence, head_dim), device=device, dtype=dtype
    ).transpose(2, 3)

    # SDPA uses a Q-head-expanded K/V tensor.  This models the GQA layout while
    # avoiding any backend-specific enable_gqa implementation differences.
    q_full = torch.randn(
        (batch, q_heads, sequence, head_dim), device=device, dtype=dtype
    )
    q_partial = torch.randn(
        (batch, q_heads, transfer_tokens, head_dim), device=device, dtype=dtype
    )
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads for GQA")
    group_size = q_heads // kv_heads
    k_base = torch.randn(
        (batch, kv_heads, sequence, head_dim), device=device, dtype=dtype
    )
    v_base = torch.randn(
        (batch, kv_heads, sequence, head_dim), device=device, dtype=dtype
    )
    k_attn = k_base.repeat_interleave(group_size, dim=1)
    v_attn = v_base.repeat_interleave(group_size, dim=1)

    # The projections in the cache hook operate on only selected tokens.  A
    # square matrix is representative of a q/k/v projection at this shape.
    linear_input = torch.randn(
        (batch, transfer_tokens, hidden), device=device, dtype=dtype
    )
    linear_q_weight = torch.randn((hidden, hidden), device=device, dtype=dtype)
    linear_k_weight = torch.randn((kv_hidden, hidden), device=device, dtype=dtype)
    linear_v_input = torch.randn(
        (batch, generation, hidden), device=device, dtype=dtype
    )
    linear_v_weight = torch.randn((kv_hidden, hidden), device=device, dtype=dtype)

    def do_cosine() -> torch.Tensor:
        return F.cosine_similarity(features_a, features_b, dim=-1)

    def do_topk() -> torch.Tensor:
        # The cache refresh policy selects the least-similar tokens.
        return torch.topk(scores, k=topk, dim=-1, largest=False, sorted=False).indices

    def do_gather() -> torch.Tensor:
        return torch.gather(hidden_generation, dim=1, index=hidden_index)

    def do_scatter_kv() -> torch.Tensor:
        return kv_generation_cache.scatter_(1, kv_index, kv_src)

    def do_scatter_hidden() -> torch.Tensor:
        return hidden_generation_cache.scatter_(1, hidden_index, hidden_src)

    def do_cat_kv() -> torch.Tensor:
        return torch.cat((prompt_kv_cache, generation_kv_cache), dim=1)

    def do_cat_hidden() -> torch.Tensor:
        return torch.cat((prompt_hidden_cache, generation_hidden_cache), dim=1)

    def do_contiguous() -> torch.Tensor:
        return contiguous_input.contiguous()

    def do_sdpa_full() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_full, k_attn, v_attn, dropout_p=0.0, is_causal=False
        )

    def do_sdpa_partial() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_partial, k_attn, v_attn, dropout_p=0.0, is_causal=False
        )

    def do_linear_q_small() -> torch.Tensor:
        return F.linear(linear_input, linear_q_weight)

    def do_linear_k_small() -> torch.Tensor:
        return F.linear(linear_input, linear_k_weight)

    def do_linear_v_full() -> torch.Tensor:
        return F.linear(linear_v_input, linear_v_weight)

    operations = {
        "cosine_similarity": do_cosine,
        "topk": do_topk,
        "gather": do_gather,
        "scatter_kv": do_scatter_kv,
        "scatter_hidden": do_scatter_hidden,
        "cat_kv": do_cat_kv,
        "cat_hidden": do_cat_hidden,
        "contiguous": do_contiguous,
        "sdpa_full": do_sdpa_full,
        "sdpa_partial": do_sdpa_partial,
        "linear_q_small": do_linear_q_small,
        "linear_k_small": do_linear_k_small,
        "linear_v_full": do_linear_v_full,
    }
    config = {
        "batch_size": batch,
        "prompt_length": prompt,
        "gen_length": generation,
        "sequence_length": sequence,
        "transfer_tokens": transfer_tokens,
        "hidden_size": hidden,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_hidden_size": kv_hidden,
        "gqa_group_size": group_size,
        "dtype": args.dtype,
        "operations": {
            "cosine_similarity": [batch, generation, kv_hidden],
            "topk": {"scores": [batch, generation], "k": topk},
            "gather": [batch, transfer_tokens, hidden],
            "scatter_kv": [batch, generation, kv_hidden],
            "scatter_hidden": [batch, generation, hidden],
            "cat_kv": {
                "prompt": [batch, prompt, kv_hidden],
                "generation": [batch, generation, kv_hidden],
            },
            "cat_hidden": {
                "prompt": [batch, prompt, hidden],
                "generation": [batch, generation, hidden],
            },
            "contiguous": [batch, q_heads, head_dim, sequence],
            "sdpa_full": {
                "q": [batch, q_heads, sequence, head_dim],
                "k_v": [batch, q_heads, sequence, head_dim],
            },
            "sdpa_partial": {
                "q": [batch, q_heads, transfer_tokens, head_dim],
                "k_v": [batch, q_heads, sequence, head_dim],
            },
            "linear_q_small": {
                "input": [batch, transfer_tokens, hidden],
                "weight": [hidden, hidden],
            },
            "linear_k_small": {
                "input": [batch, transfer_tokens, hidden],
                "weight": [kv_hidden, hidden],
            },
            "linear_v_full": {
                "input": [batch, generation, hidden],
                "weight": [kv_hidden, hidden],
            },
        },
    }
    return operations, config


def benchmark_operation(
    name: str,
    operation: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
) -> dict:
    for _ in range(warmup):
        result = operation()
        del result
    sync()
    baseline_memory = memory_stats()
    reset_peak_memory()

    elapsed_us: list[float] = []
    result = None
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = operation()
        sync()
        elapsed_us.append((time.perf_counter() - start) * 1e6)
        del result

    # Re-run once outside the timed region to produce a stable correctness
    # digest.  This copy is intentionally excluded from latency statistics.
    sync()
    final = operation()
    sync()
    output_hash = digest_tensor(final)
    del final
    memory = memory_stats()
    # Inputs are intentionally persistent across samples.  Reporting the
    # excess over that baseline makes temporary workspace cost visible.
    memory["peak_extra_allocated_bytes"] = max(
        0,
        memory["peak_allocated_bytes"] - baseline_memory["current_allocated_bytes"],
    )
    memory["peak_extra_reserved_bytes"] = max(
        0,
        memory["peak_reserved_bytes"] - baseline_memory["current_reserved_bytes"],
    )
    stats = {
        "name": name,
        "samples": repeats,
        "latency_us": {
            "mean": statistics.fmean(elapsed_us),
            "p50": percentile(elapsed_us, 0.50),
            "p90": percentile(elapsed_us, 0.90),
            "p99": percentile(elapsed_us, 0.99),
            "min": min(elapsed_us),
            "max": max(elapsed_us),
        },
        "output_sha256": output_hash,
        "memory": memory,
    }
    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)
    if _TORCH_NPU_IMPORT_ERROR is not None:
        raise RuntimeError(
            "microbench_npu.py requires torch-npu; run it on an Ascend host"
        ) from _TORCH_NPU_IMPORT_ERROR
    index = device_index(args.device)
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false; an Ascend NPU is required")
    torch.npu.set_device(index)
    torch.manual_seed(1234)
    try:
        torch.npu.manual_seed(1234)
    except AttributeError:
        pass

    operations, shape_config = make_operations(args)
    results = []
    for name, operation in operations.items():
        print(f"[microbench] {name}", flush=True)
        try:
            results.append(
                benchmark_operation(name, operation, args.warmup, args.repeats)
            )
        except Exception as exc:
            # Keep a machine-readable record for unsupported kernels (for
            # example an older CANN release without NPU SDPA) and continue with
            # the remaining operators.
            results.append(
                {
                    "name": name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[microbench] {name} failed: {exc}", flush=True)

    npu_name = None
    try:
        npu_name = torch.npu.get_device_name(index)
    except (AttributeError, RuntimeError):
        pass
    report = {
        "benchmark": "dllm_cache_npu_operator_microbench",
        "model": args.model,
        "device": f"npu:{index}",
        "backend": "npu",
        "npu_name": npu_name,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "transfer_ratio": args.transfer_ratio,
        "shape_config": shape_config,
        "results": results,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
