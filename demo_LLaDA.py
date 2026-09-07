"""Interactive LLaDA text-generation demo."""

import argparse
import time
from dataclasses import asdict
from datetime import datetime

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import logout_cache_LLaDA, register_cache_LLaDA
from dllm_cache.runtime import resolve_dtype, resolve_runtime
from utils import generate


MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--max-history-tokens", type=int, default=2048)
    parser.add_argument("--prompt-interval-steps", type=int, default=100)
    parser.add_argument("--gen-interval-steps", type=int, default=7)
    parser.add_argument("--transfer-ratio", type=float, default=0.25)
    parser.add_argument("--no-cache", action="store_true", help="start with cache disabled")
    return parser.parse_args()


def format_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_conversation(tokenizer, history, max_tokens):
    total_tokens = 0
    truncated_history = []
    for message in reversed(history):
        tokens = len(tokenizer(message["content"])["input_ids"])
        if total_tokens + tokens > max_tokens:
            break
        truncated_history.insert(0, message)
        total_tokens += tokens
    return truncated_history


def configure_cache(model, args, enabled: bool) -> None:
    if enabled:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=args.prompt_interval_steps,
                    gen_interval_steps=args.gen_interval_steps,
                    transfer_ratio=args.transfer_ratio,
                )
            )
        )
        register_cache_LLaDA(model, "model.transformer.blocks")
    else:
        logout_cache_LLaDA(model, "model.transformer.blocks")
        dLLMCache.new_instance(prompt_interval_steps=1, gen_interval_steps=1)


def print_help() -> None:
    print("\nAvailable commands:")
    print("  <help>       : Show this help message")
    print("  <use_cache>  : Enable cache")
    print("  <no_cache>   : Disable cache")
    print("  <clear>      : Clear conversation history")
    print("  <exit>       : Exit the program\n")


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime(args.device)
    dtype = resolve_dtype(args.dtype, runtime=runtime)
    revision = MODEL_REVISION if args.revision == "auto" else args.revision

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(runtime.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=True
    )

    use_cache = not args.no_cache
    configure_cache(model, args, use_cache)
    conversation_history = []

    print("*" * 66)
    print(
        f"** Device: {runtime.device} | Dtype: {dtype} | Answer Length: "
        f"{args.gen_length} | Sampling Steps: {args.steps} | Cache Enabled: {use_cache}"
    )
    print("*" * 66)
    print("Type '<help>' for available commands.")

    while True:
        print("\n" + "=" * 70)
        user_input = input(
            f"Enter your question (Cache is {'enable' if use_cache else 'disable'}, "
            "Type '<help>' for available commands): "
        )
        command = user_input.lower()
        if command == "<exit>":
            print("Conversation ended.")
            break
        if command == "<help>":
            print_help()
            continue
        if command == "<no_cache>":
            configure_cache(model, args, False)
            use_cache = False
            print("Cache disabled. Please continue with your question.")
            continue
        if command == "<use_cache>":
            configure_cache(model, args, True)
            use_cache = True
            print("Cache enabled. Please continue with your question.")
            continue
        if command == "<clear>":
            conversation_history = []
            print("Conversation history cleared. Please continue with your question.")
            continue

        conversation_history.append(
            {"role": "user", "content": user_input, "time": format_time()}
        )
        conversation_history = truncate_conversation(
            tokenizer, conversation_history, args.max_history_tokens
        )
        formatted_input = tokenizer.apply_chat_template(
            conversation_history, add_generation_prompt=True, tokenize=False
        )
        encoded = tokenizer(formatted_input, return_tensors="pt")
        input_ids = encoded["input_ids"].to(runtime.device)
        attention_mask = encoded["attention_mask"].to(runtime.device)

        runtime.reset_peak_memory_stats()
        runtime.synchronize()
        start_time = time.perf_counter()
        generation_ids = generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            model=model,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
        )
        runtime.synchronize()
        elapsed = time.perf_counter() - start_time

        answer = tokenizer.batch_decode(
            generation_ids, skip_special_tokens=True
        )[0]
        reply_time = format_time()
        conversation_history.append(
            {"role": "assistant", "content": answer, "time": reply_time}
        )
        print(f"LLaDA ({reply_time}): {answer}")
        print(f"Generation Time: {elapsed:.2f} seconds")
        print(f"Memory: {runtime.memory_stats()}")


if __name__ == "__main__":
    main()
