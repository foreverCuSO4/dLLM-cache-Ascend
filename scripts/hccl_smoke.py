#!/usr/bin/env python3
"""Eight-rank HCCL all-reduce smoke test, intended to run under torchrun."""

from __future__ import annotations

import os

import torch
import torch.distributed as distributed
import torch_npu  # noqa: F401  # Registers torch.npu and the HCCL backend.


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if int(torch.npu.device_count()) < world_size:
        raise RuntimeError(
            f"HCCL smoke needs {world_size} visible NPUs, "
            f"but torch.npu sees {torch.npu.device_count()}"
        )

    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    distributed.init_process_group(backend="hccl", init_method="env://")
    try:
        value = torch.tensor(float(rank + 1), device=device, dtype=torch.float32)
        distributed.all_reduce(value, op=distributed.ReduceOp.SUM)
        torch.npu.synchronize(device)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise AssertionError(
                f"rank {rank}: expected all_reduce result {expected}, got {value.item()}"
            )
        distributed.barrier()
        if rank == 0:
            print(
                f"HCCL all_reduce passed on {world_size} ranks; "
                f"sum(1..{world_size})={value.item()}"
            )
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
