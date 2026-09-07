"""Benchmark LLaDA or Dream text generation on CPU, CUDA, or Ascend NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import (
    logout_cache_Dream,
    logout_cache_LLaDA,
    register_cache_Dream,
    register_cache_LLaDA,
)
from dllm_cache.runtime import resolve_dtype, resolve_runtime
from utils import generate


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    module_path: str
    register_cache: Callable
    logout_cache: Callable


MODEL_SPECS = {
    "llada": ModelSpec(
        model_id="GSAI-ML/LLaDA-8B-Instruct",
        revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",
        module_path="model.transformer.blocks",
        register_cache=register_cache_LLaDA,
        logout_cache=logout_cache_LLaDA,
    ),
    "dream": ModelSpec(
        model_id="Dream-org/Dream-v0-Instruct-7B",
        revision="05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        module_path="model.layers",
        register_cache=register_cache_Dream,
        logout_cache=logout_cache_Dream,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--mode",
        default="both",
        choices=("baseline", "full-refresh", "adaptive", "both"),
        help="'both' runs baseline followed by adaptive",
    )
    parser.add_argument("--device", default="auto", help="auto, npu[:N], cuda[:N], or cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--revision",
        default="auto",
        help="Hugging Face revision; 'auto' uses the tested pinned commit",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default="Explain why caching can accelerate diffusion language models.")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="LLaDA block length; defaults to gen-length",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--prompt-interval-steps", type=int, default=100)
    parser.add_argument("--gen-interval-steps", type=int, default=7)
    parser.add_argument("--cfg-interval-steps", type=int, default=1)
    parser.add_argument("--transfer-ratio", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "prompt_length", "gen_length", "steps", "repeats"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if (
        args.prompt_interval_steps < 1
        or args.gen_interval_steps < 1
        or args.cfg_interval_steps < 1
    ):
        raise ValueError("cache interval steps must be at least 1")
    if not 0.0 <= args.transfer_ratio <= 1.0:
        raise ValueError("--transfer-ratio must be between 0 and 1")
    if args.model == "llada":
        block_length = args.block_length or args.gen_length
        if args.gen_length % block_length:
            raise ValueError("--gen-length must be divisible by --block-length")
        blocks = args.gen_length // block_length
        if args.steps % blocks:
            raise ValueError("--steps must be divisible by the number of LLaDA blocks")


def build_inputs(tokenizer, prompt: str, prompt_length: int, batch_size: int, device):
    token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if not token_ids:
        raise ValueError("the benchmark prompt tokenized to an empty sequence")
    copies = (prompt_length + len(token_ids) - 1) // len(token_ids)
    token_ids = (token_ids * copies)[-prompt_length:]
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def configure_mode(
    model, spec: ModelSpec, mode: str, args: argparse.Namespace
) -> dLLMCacheConfig:
    spec.logout_cache(model, spec.module_path)
    if mode == "baseline":
        config = dLLMCacheConfig(
            prompt_interval_steps=1,
            gen_interval_steps=1,
            cfg_interval_steps=args.cfg_interval_steps,
            transfer_ratio=0.0,
        )
    elif mode == "full-refresh":
        config = dLLMCacheConfig(
            prompt_interval_steps=1,
            gen_interval_steps=1,
            cfg_interval_steps=args.cfg_interval_steps,
            transfer_ratio=0.0,
        )
        spec.register_cache(model, spec.module_path)
    else:
        config = dLLMCacheConfig(
            prompt_interval_steps=args.prompt_interval_steps,
            gen_interval_steps=args.gen_interval_steps,
            cfg_interval_steps=args.cfg_interval_steps,
            transfer_ratio=args.transfer_ratio,
        )
        spec.register_cache(model, spec.module_path)
    dLLMCache.new_instance(**asdict(config))
    return config


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = resolve_runtime(args.device)
    dtype = resolve_dtype(args.dtype, runtime=runtime)
    spec = MODEL_SPECS[args.model]
    revision = spec.revision if args.revision == "auto" else args.revision
    modes = ["baseline", "adaptive"] if args.mode == "both" else [args.mode]

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
        tokenizer,
        args.prompt,
        args.prompt_length,
        args.batch_size,
        runtime.device,
    )

    @torch.inference_mode()
    def run_once():
        if args.model == "llada":
            return generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model=model,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length or args.gen_length,
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
        ).sequences[:, input_ids.shape[1] :]

    results = []
    for mode in modes:
        effective_cache_config = configure_mode(model, spec, mode, args)
        for _ in range(args.warmup):
            run_once()
        runtime.synchronize()
        runtime.reset_peak_memory_stats()

        elapsed_times = []
        for _ in range(args.repeats):
            runtime.synchronize()
            start = time.perf_counter()
            output = run_once()
            runtime.synchronize()
            elapsed_times.append(time.perf_counter() - start)
        if output.shape[0] != args.batch_size:
            raise RuntimeError(
                f"model returned batch size {output.shape[0]}, expected {args.batch_size}"
            )
        output_token_ids = output.detach().cpu().tolist()
        output_payload = json.dumps(
            output_token_ids, separators=(",", ":")
        ).encode("utf-8")

        total_time = sum(elapsed_times)
        result = {
            "mode": mode,
            "latency_seconds": {
                "mean": statistics.fmean(elapsed_times),
                "p50": percentile(elapsed_times, 0.50),
                "p90": percentile(elapsed_times, 0.90),
                "min": min(elapsed_times),
                "max": max(elapsed_times),
            },
            "samples_per_second": args.batch_size * args.repeats / total_time,
            "generated_tokens_per_second": (
                args.batch_size * args.gen_length * args.repeats / total_time
            ),
            "memory": asdict(runtime.memory_stats()),
            "cache_config": asdict(effective_cache_config),
            "output_sha256": hashlib.sha256(output_payload).hexdigest(),
            "output_token_ids": output_token_ids,
            "speedup_vs_baseline": None,
        }
        results.append(result)
        spec.logout_cache(model, spec.module_path)

    baseline = next((r for r in results if r["mode"] == "baseline"), None)
    if baseline is not None:
        baseline_mean = baseline["latency_seconds"]["mean"]
        for result in results:
            result["speedup_vs_baseline"] = (
                baseline_mean / result["latency_seconds"]["mean"]
            )

    report = {
        "model": args.model,
        "model_id": spec.model_id,
        "revision": revision,
        "device": str(runtime.device),
        "backend": runtime.backend,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "block_length": (
            (args.block_length or args.gen_length)
            if args.model == "llada"
            else None
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cache_config": {
            "prompt_interval_steps": args.prompt_interval_steps,
            "gen_interval_steps": args.gen_interval_steps,
            "cfg_interval_steps": args.cfg_interval_steps,
            "transfer_ratio": args.transfer_ratio,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
