import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union,Type,TypeVar

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,

)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
eval_logger = logging.getLogger(__name__)
from dllm_cache.cache import  dLLMCacheConfig,dLLMCache
from dllm_cache.hooks import  register_cache_Dream
from dllm_cache.runtime import resolve_dtype, resolve_runtime
from dataclasses import asdict
T = TypeVar("T", bound="LM")
from lm_eval.api.model import LM


PINNED_REVISIONS = {
    "Dream-org/Dream-v0-Instruct-7B": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    "Dream-org/Dream-v0-Base-7B": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
}


def _resolve_revision(pretrained: str, revision: Optional[str]) -> str:
    if revision is not None and str(revision).lower() != "auto":
        return str(revision)
    if pretrained in PINNED_REVISIONS:
        return PINNED_REVISIONS[pretrained]
    raise ValueError(
        f"No tested revision is registered for {pretrained!r}; pass revision=<commit SHA> explicitly"
    )


@register_model("dream")
class Dream(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        revision: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        is_feature_cache: bool = False,
        is_cfg_cache: bool = False,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        cfg_interval_steps: int = 1,
        transfer_ratio:float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.prompt_interval_steps = prompt_interval_steps
        self.gen_interval_steps = gen_interval_steps
        self.cfg_interval_steps = cfg_interval_steps
        self.transfer_ratio = transfer_ratio
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        self.revision = _resolve_revision(pretrained, revision)
        self._rank = 0
        self._world_size = 1

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        runtime_device = (
            str(accelerator.device) if accelerator.num_processes > 1 else device
        )
        self._runtime = resolve_runtime(runtime_device)
        self._device = self._runtime.device
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        if parallelize and self._device.type == "npu":
            raise ValueError(
                "parallelize=True is not supported on NPU; launch one process per "
                "NPU with Accelerate data parallelism instead"
            )
        if parallelize and accelerator.num_processes > 1:
            raise ValueError(
                "parallelize=True cannot be combined with an Accelerate multi-process launch"
            )
        eval_logger.info(
            "Using device '%s' (rank %s/%s)",
            self._device,
            self._rank,
            self._world_size,
        )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        self._create_model_and_tokenizer(
            pretrained, self.revision, dtype, trust_remote_code
        )


        if is_feature_cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))
            register_cache_Dream(self.model,"model.layers")
        else:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=1,
                    gen_interval_steps=1,
                    transfer_ratio=0,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))

        if self.rank == 0:
                print(f"Feature Cache is {is_feature_cache}.CFG Cache is {is_cfg_cache},prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, cfg_interval_steps={cfg_interval_steps}")

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        # generation params
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # lm-eval's distributed evaluator uses these primitives to balance
    # requests and gather per-rank samples/metrics.  The original adapter
    # only exposed rank/world_size, so the base no-op implementations made an
    # eight-rank run treat a scalar local count as the global count.  Delegate
    # to Accelerate/torch.distributed exactly as the upstream HF adapter does.
    def all_gather(self, tensor):
        if self.world_size <= 1:
            return tensor
        return self.accelerator.gather(tensor)

    def gather_object(self, obj, dst=0):
        if self.world_size <= 1:
            return [obj]
        result = [None] * self.world_size if self.rank == dst else None
        torch.distributed.gather_object(obj=obj, object_gather_list=result, dst=dst)
        return result

    def barrier(self):
        if self.world_size > 1:
            self.accelerator.wait_for_everyone()

    def _create_model_and_tokenizer(
        self, pretrained, revision, dtype, trust_remote_code
    ):
        self.model = (
            transformers.AutoModel.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=(
                    dtype
                    if isinstance(dtype, torch.dtype)
                    else resolve_dtype(dtype, runtime=self._runtime)
                ),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, revision=revision, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        # tokenize
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        prompt_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids
        self.tokenizer.padding_side = old_padding_side
        prompt_limit = self.max_length - self.max_new_tokens
        if prompt_ids.shape[1] > prompt_limit:
            eval_logger.warning(
                "Prompt length %s is larger than %s, cutoff on the left side",
                prompt_ids.shape[1],
                prompt_limit,
            )
            prompt_ids = prompt_ids[:, -prompt_limit:]

        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)
        feature_cache = dLLMCache()
        feature_cache.reset_cache(prompt_ids.shape[1])
        generation_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
        )

        # decode
        responses = [
            self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_ids.sequences)
        ]

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)
            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0]['until']:
                        r = r.split(s)[0]
                    responses[i] = r

            # if self.rank == 0:
            #     print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")

            res.extend(responses)
            pbar.update(len(contexts))

        return res
