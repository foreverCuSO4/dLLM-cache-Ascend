#!/usr/bin/env python3
"""Profile three diffusion steps of a full Ascend-NPU generation.

The model still runs the requested ``steps`` (128 by default); the profiler
records only a small window inside that run.  This keeps the trace manageable
while preserving the real cache state at the selected diffusion steps.  Dream
exposes a generation-token hook, while LLaDA uses the optional ``step_callback``
argument added to :func:`utils.generate`.

Examples::

    # Profile the NPU baseline, steps 10--12 of a 128-step run.
    python scripts/profile_npu.py --model dream --mode baseline \
        --device npu:0 --output-dir reports/profile_dream_baseline_npu

    # Profile adaptive with transfer_ratio=0 (the cache-overhead anchor).
    python scripts/profile_npu.py --model dream --mode adaptive \
        --transfer-ratio 0 --device npu:1 \
        --output-dir reports/profile_dream_adaptive_ratio0_npu

The trace directory can be opened with TensorBoard's profiler plugin.  A
``metadata.json`` file records the exact model revision, shape, mode and
profile window.  No CUDA APIs are used or required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

# Running ``python scripts/profile_npu.py`` puts ``scripts/`` (rather than the
# repository root) on sys.path.  Keep imports consistent with benchmark_text.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The Ascend environment is self-contained.  A stale user-site torchvision
# can otherwise be imported by Transformers before the pinned environment
# packages (for example, ``torchvision::nms`` mismatches torch 2.4).  Remove
# only the user-site entry; callers can still opt into it explicitly with
# ``PYTHONNOUSERSITE=0`` and a custom launcher.
try:
    _user_site = site.getusersitepackages()
except (AttributeError, TypeError):  # pragma: no cover - unusual Python build
    _user_site = None
if _user_site and _user_site in sys.path:
    sys.path.remove(_user_site)

import torch
from torch.profiler import record_function
from transformers import AutoModel, AutoTokenizer

from benchmark_text import MODEL_SPECS, build_inputs, configure_mode, validate_args
from dllm_cache.cache import dLLMCache
from dllm_cache.runtime import resolve_dtype, resolve_runtime
from utils import generate


def _load_npu_profiler():
    """Import torch_npu's profiler lazily with an actionable error."""

    try:
        import torch_npu  # noqa: F401  # registers torch.npu
        from torch_npu import profiler as npu_profiler
    except Exception as exc:  # pragma: no cover - host without CANN
        raise RuntimeError(
            "scripts/profile_npu.py requires torch_npu and a CANN runtime"
        ) from exc
    return npu_profiler


class _DiffusionStepTrace:
    """Delimit diffusion steps and advance the NPU profiler schedule.

    A single ``diffusion_generate`` call contains many model forwards.  Calling
    ``prof.step`` from the model's generation hook lets the profiler schedule
    capture exactly three diffusion steps rather than the whole 128-step run.
    ``record_function`` spans the forward, sampling and cache update of each
    step, making the selected window easy to find in a TensorBoard trace.
    """

    def __init__(self, profiler: Any, total_steps: int):
        self.profiler = profiler
        self.total_steps = total_steps
        self._scope = None
        self._active_step: Optional[int] = None

    def begin(self, step: int) -> None:
        if self._scope is not None:
            raise RuntimeError(f"diffusion step {self._active_step} was not closed")
        self._active_step = int(step)
        self._scope = record_function(f"dllm.diffusion_step.{step:04d}")
        self._scope.__enter__()

    def end(self, step: int) -> None:
        if self._scope is None:
            raise RuntimeError(f"diffusion step {step} ended before it began")
        if int(step) != self._active_step:
            raise RuntimeError(
                f"diffusion step mismatch: active={self._active_step}, ended={step}"
            )
        self._scope.__exit__(None, None, None)
        self._scope = None
        self._active_step = None
        # torch_npu's schedule is driven explicitly by step().
        self.profiler.step()

    def close(self) -> None:
        """Close a scope if generation raised before its end callback."""

        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None
            self._active_step = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="dream")
    parser.add_argument(
        "--mode",
        choices=("baseline", "full-refresh", "adaptive"),
        default="adaptive",
        help="cache mode; use adaptive + --transfer-ratio 0 for ratio=0 anchor",
    )
    parser.add_argument("--device", default="npu:0", help="must be npu[:index]")
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32")
    )
    parser.add_argument("--revision", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="Explain why caching can accelerate diffusion language models.",
    )
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--prompt-interval-steps", type=int, default=100)
    parser.add_argument("--gen-interval-steps", type=int, default=8)
    parser.add_argument("--cfg-interval-steps", type=int, default=1)
    parser.add_argument("--transfer-ratio", type=float, default=0.25)
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=10,
        help="first diffusion step in the trace (the full run is still executed)",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=3,
        help="number of consecutive diffusion steps to record",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="trace/metadata directory (default: reports/profile_<model>_<mode>_npu)",
    )
    return parser.parse_args()


def _validate_profile_args(args: argparse.Namespace) -> None:
    # Reuse benchmark validation for model shape/cache interval constraints.
    # A profiler invocation is one generation, so provide the two benchmark-
    # only fields expected by validate_args without exposing misleading CLI
    # controls here.
    args.warmup = 0
    args.repeats = 1
    validate_args(args)
    if not str(args.device).strip().lower().startswith("npu"):
        raise ValueError("profile_npu.py is NPU-only; pass --device npu[:index]")
    if args.profile_start_step < 0:
        raise ValueError("--profile-start-step must be non-negative")
    if args.profile_steps < 1:
        raise ValueError("--profile-steps must be at least 1")
    if args.profile_start_step + args.profile_steps > args.steps:
        raise ValueError(
            "profile window exceeds generation steps: start + profile-steps "
            f"= {args.profile_start_step + args.profile_steps} > {args.steps}"
        )


def _make_dream_step_hook(trace: _DiffusionStepTrace, total_steps: int):
    def hook(step, x, logits):
        if step is None:
            trace.begin(0)
        else:
            trace.end(int(step))
            next_step = int(step) + 1
            if next_step < total_steps:
                trace.begin(next_step)
        return x

    return hook


def _make_llada_step_callback(trace: _DiffusionStepTrace):
    def callback(step: int, phase: str) -> None:
        if phase == "begin":
            trace.begin(int(step))
        elif phase == "end":
            trace.end(int(step))
        else:
            raise ValueError(f"unknown LLaDA step callback phase {phase!r}")

    return callback


def _run_generation(
    args: argparse.Namespace,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    trace: _DiffusionStepTrace,
) -> torch.Tensor:
    if args.model == "llada":
        return generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            model=model,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length or args.gen_length,
            step_callback=_make_llada_step_callback(trace),
        )

    dLLMCache().reset_cache(input_ids.shape[1])
    return model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.gen_length,
        output_history=False,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=0.0,
        generation_tokens_hook_func=_make_dream_step_hook(trace, args.steps),
    ).sequences[:, input_ids.shape[1] :]


def main() -> None:
    args = parse_args()
    _validate_profile_args(args)
    npu_profiler = _load_npu_profiler()
    runtime = resolve_runtime(args.device)
    if runtime.backend != "npu":
        raise RuntimeError(f"expected NPU runtime, got {runtime.backend!r}")
    dtype = resolve_dtype(args.dtype, runtime=runtime)
    spec = MODEL_SPECS[args.model]
    revision = spec.revision if args.revision == "auto" else args.revision
    output_dir = args.output_dir or (
        REPO_ROOT / "reports" / f"profile_{args.model}_{args.mode}_npu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    model = AutoModel.from_pretrained(
        spec.model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(runtime.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, revision=revision, trust_remote_code=True
    )
    input_ids, attention_mask = build_inputs(
        tokenizer, args.prompt, args.prompt_length, args.batch_size, runtime.device
    )
    cache_config = configure_mode(model, spec, args.mode, args)

    schedule = npu_profiler.schedule(
        skip_first=args.profile_start_step,
        wait=0,
        warmup=0,
        active=args.profile_steps,
        repeat=1,
    )
    activities = [npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU]
    trace_handler = npu_profiler.tensorboard_trace_handler(
        str(trace_dir), analyse_flag=True
    )
    profiler = npu_profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    trace = _DiffusionStepTrace(profiler, args.steps)
    runtime.reset_peak_memory_stats()
    runtime.synchronize()
    output = None
    try:
        with profiler:
            with record_function(f"dllm.generation.{args.model}.{args.mode}"):
                output = _run_generation(
                    args, model, input_ids, attention_mask, trace
                )
            runtime.synchronize()
    finally:
        trace.close()
        spec.logout_cache(model, spec.module_path)

    if output is None:
        raise RuntimeError("generation returned no output")
    output_token_ids = output.detach().cpu().tolist()
    output_payload = json.dumps(output_token_ids, separators=(",", ":")).encode()
    metadata = {
        "model": args.model,
        "model_id": spec.model_id,
        "revision": revision,
        "mode": args.mode,
        "device": str(runtime.device),
        "backend": runtime.backend,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "block_length": args.block_length if args.model == "llada" else None,
        "cache_config": asdict(cache_config),
        "profile_start_step": args.profile_start_step,
        "profile_steps": args.profile_steps,
        "profile_activities": ["CPU", "NPU"],
        "trace_dir": str(trace_dir),
        "output_sha256": hashlib.sha256(output_payload).hexdigest(),
        "output_token_ids": output_token_ids,
        "memory": asdict(runtime.memory_stats()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
