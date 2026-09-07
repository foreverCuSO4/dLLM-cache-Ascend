#!/usr/bin/env python3
"""Validate the pinned software stack, visible NPUs, and required NPU ops."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


EXPECTED_VERSIONS = {
    "torch": "2.4.0",
    "torch-npu": "2.4.0.post2",
    "transformers": "4.51.3",
    "accelerate": "1.6.0",
    "datasets": "3.6.0",
    "lm-eval": "0.4.12",
    "jieba": "0.42.1",
    "fuzzywuzzy": "0.18.0",
    "rouge": "1.0.1",
    "numpy": "1.26.4",
    "peft": "0.15.2",
    "huggingface-hub": "0.36.2",
    "hf-xet": "1.5.1",
    "decorator": "5.2.1",
    "cloudpickle": "3.1.2",
    "ml-dtypes": "0.5.3",
    "tornado": "6.5.4",
    "einops": "0.8.2",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name, True, fn())
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


def package_versions(allow_mismatch: bool) -> str:
    mismatches: list[str] = []
    installed: list[str] = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        installed.append(f"{package}={actual}")
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, got {actual}")
    if mismatches and not allow_mismatch:
        raise RuntimeError("; ".join(mismatches))
    suffix = " (version mismatch allowed)" if mismatches else ""
    return ", ".join(installed) + suffix


def python_version(allow_mismatch: bool) -> str:
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != "3.10" and not allow_mismatch:
        raise RuntimeError(f"expected Python 3.10, got {actual}")
    return f"{platform.python_version()} ({platform.platform()})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-devices",
        type=int,
        default=8,
        help="minimum number of visible NPUs required (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=int,
        action="append",
        help="device index to test; repeat to test several (default: every visible NPU)",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="report but do not fail on Python/package version drift",
    )
    parser.add_argument(
        "--run-hccl",
        action="store_true",
        help="launch one torchrun rank per visible NPU and test HCCL all_reduce",
    )
    args = parser.parse_args()

    results = [
        check("python", lambda: python_version(args.allow_version_mismatch)),
        check(
            "package versions",
            lambda: package_versions(args.allow_version_mismatch),
        ),
    ]

    try:
        import torch
        import torch.nn.functional as functional
        import torch_npu  # noqa: F401
    except Exception as exc:
        results.append(CheckResult("torch_npu import", False, repr(exc)))
        for result in results:
            print(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
        return 1

    def visible_devices() -> str:
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu was not registered by torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is false")
        count = int(torch.npu.device_count())
        if count < args.min_devices:
            raise RuntimeError(f"expected >= {args.min_devices} visible NPUs, got {count}")
        names = [str(torch.npu.get_device_name(index)) for index in range(count)]
        return f"count={count}, names={names}"

    results.append(check("visible NPUs", visible_devices))
    count = int(torch.npu.device_count()) if hasattr(torch, "npu") else 0
    device_indices = args.device if args.device is not None else list(range(count))

    def test_ops(index: int) -> str:
        if index < 0 or index >= count:
            raise ValueError(f"device npu:{index} is out of range for {count} visible NPUs")
        device = torch.device(f"npu:{index}")
        torch.npu.set_device(device)
        torch.manual_seed(910 + index)

        q_cpu = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        k_cpu = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        v_cpu = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        expected_sdpa = functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        actual_sdpa = functional.scaled_dot_product_attention(
            q_cpu.to(device=device, dtype=torch.bfloat16),
            k_cpu.to(device=device, dtype=torch.bfloat16),
            v_cpu.to(device=device, dtype=torch.bfloat16),
        )
        torch.npu.synchronize(device)
        if not torch.isfinite(actual_sdpa).all().item():
            raise AssertionError("BF16 SDPA produced NaN or Inf")
        torch.testing.assert_close(
            actual_sdpa.float().cpu(), expected_sdpa, rtol=2e-2, atol=2e-2
        )

        matmul_lhs_cpu = torch.randn(64, 64, dtype=torch.float32)
        matmul_rhs_cpu = torch.randn(64, 64, dtype=torch.float32)
        expected_matmul = torch.matmul(matmul_lhs_cpu, matmul_rhs_cpu)
        actual_matmul = torch.matmul(
            matmul_lhs_cpu.to(device=device, dtype=torch.bfloat16),
            matmul_rhs_cpu.to(device=device, dtype=torch.bfloat16),
        )
        if not torch.isfinite(actual_matmul).all().item():
            raise AssertionError("BF16 matmul produced NaN or Inf")
        torch.testing.assert_close(
            actual_matmul.float().cpu(), expected_matmul, rtol=2e-2, atol=8e-2
        )

        lhs_cpu = torch.randn(4, 32, dtype=torch.float32)
        rhs_cpu = torch.randn(4, 32, dtype=torch.float32)
        expected_cosine = functional.cosine_similarity(lhs_cpu, rhs_cpu, dim=-1)
        actual_cosine = functional.cosine_similarity(
            lhs_cpu.to(device=device, dtype=torch.bfloat16),
            rhs_cpu.to(device=device, dtype=torch.bfloat16),
            dim=-1,
        )
        if not torch.isfinite(actual_cosine).all().item():
            raise AssertionError("BF16 cosine_similarity produced NaN or Inf")
        torch.testing.assert_close(
            actual_cosine.float().cpu(), expected_cosine, rtol=2e-2, atol=2e-2
        )

        source = torch.arange(24, device=device, dtype=torch.float32).reshape(4, 6)
        values, indices = torch.topk(source, k=3, dim=-1)
        gathered = torch.gather(source, 1, indices)
        torch.testing.assert_close(gathered, values)
        scattered = torch.zeros_like(source)
        scattered.scatter_(1, indices, values)
        if not all(
            torch.isfinite(output).all().item()
            for output in (values, gathered, scattered)
        ):
            raise AssertionError("index operators produced NaN or Inf")
        if not torch.equal(torch.gather(scattered, 1, indices), values):
            raise AssertionError("scatter_/gather round-trip mismatch")

        torch.npu.synchronize(device)
        total = int(torch.npu.get_device_properties(device).total_memory)
        allocated = int(torch.npu.memory_allocated(device))
        return (
            "BF16 SDPA/matmul/cosine/topk/gather/scatter passed; "
            f"HBM={total}, allocated={allocated}"
        )

    for index in device_indices:
        results.append(check(f"npu:{index} operators", lambda index=index: test_ops(index)))

    if args.run_hccl:
        def test_hccl() -> str:
            if count < args.min_devices:
                raise RuntimeError(
                    f"need at least {args.min_devices} visible NPUs, got {count}"
                )
            script = Path(__file__).with_name("hccl_smoke.py")
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.min_devices}",
                str(script),
            ]
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = [line for line in completed.stdout.splitlines() if line.strip()]
            return output[-1] if output else f"passed with {args.min_devices} ranks"

        results.append(check(f"HCCL {args.min_devices}-rank all_reduce", test_hccl))

    for result in results:
        print(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
    failures = sum(not result.ok for result in results)
    if failures:
        print(f"Preflight failed: {failures} check(s) did not pass.", file=sys.stderr)
        return 1
    print("Ascend preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
