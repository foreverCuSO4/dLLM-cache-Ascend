import torch
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from dllm_cache.cache import dLLMCache
import torch.nn as nn
import inspect
import types


_HOOK_MARKER = "_dllm_cache_dream_hooked"


def _find_target_module(
    model: nn.Module, tf_block_module_key_name: str
) -> Optional[nn.Module]:
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            return module
    return None


def _has_parameters(function, required_names, description: str) -> None:
    if not callable(function):
        raise TypeError(f"{description} must be callable")
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cannot inspect {description} signature") from exc
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    missing = [
        name for name in required_names if name not in parameters and not accepts_kwargs
    ]
    if missing:
        raise TypeError(
            f"{description} is missing required parameters: {', '.join(missing)}"
        )


def _is_hooked_method(method, hook) -> bool:
    return getattr(method, "__func__", None) is hook


def _validate_block(tf_block: nn.Module, block_index: int) -> None:
    required_block_attributes = (
        "self_attn",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
    )
    missing = [
        name for name in required_block_attributes if not hasattr(tf_block, name)
    ]
    if missing:
        raise TypeError(
            f"Decoder block {block_index} is missing required attributes: "
            f"{', '.join(missing)}"
        )
    required_attention_attributes = (
        "layer_idx",
        "q_proj",
        "k_proj",
        "v_proj",
        "num_heads",
        "num_key_value_heads",
        "num_key_value_groups",
        "head_dim",
        "hidden_size",
        "attention_dropout",
        "o_proj",
    )
    missing = [
        name
        for name in required_attention_attributes
        if not hasattr(tf_block.self_attn, name)
    ]
    if missing:
        raise TypeError(
            f"Decoder block {block_index} self_attn is missing required attributes: "
            f"{', '.join(missing)}"
        )

    if getattr(tf_block, _HOOK_MARKER, False):
        correctly_hooked = _is_hooked_method(
            tf_block.forward, decoder_hook
        ) and _is_hooked_method(tf_block.self_attn.forward, attention)
        if not correctly_hooked:
            raise RuntimeError(
                f"Decoder block {block_index} has an inconsistent Dream hook state"
            )
        return

    conflicts = [
        name
        for owner, name in (
            (tf_block, "_old_forward"),
            (tf_block.self_attn, "_old_forward"),
        )
        if hasattr(owner, name)
    ]
    if conflicts:
        raise RuntimeError(
            f"Decoder block {block_index} already contains hook backup attributes: "
            f"{', '.join(conflicts)}"
        )

    _has_parameters(
        tf_block.forward,
        ("hidden_states", "attention_mask", "position_ids"),
        f"decoder block {block_index} forward",
    )
    _has_parameters(
        tf_block.self_attn.forward,
        ("hidden_states", "attention_mask"),
        f"decoder block {block_index} attention forward",
    )


def _restore_block(tf_block: nn.Module) -> None:
    if hasattr(tf_block, "_old_forward"):
        tf_block.forward = tf_block._old_forward
        delattr(tf_block, "_old_forward")
    self_attn = getattr(tf_block, "self_attn", None)
    if self_attn is not None and hasattr(self_attn, "_old_forward"):
        self_attn.forward = self_attn._old_forward
        delattr(self_attn, "_old_forward")
    if hasattr(tf_block, _HOOK_MARKER):
        delattr(tf_block, _HOOK_MARKER)


def logout_cache_Dream(model: nn.Module, tf_block_module_key_name: str) -> None:
    """Restore original functions for transformer blocks and attention modules."""
    target_module = _find_target_module(model, tf_block_module_key_name)
    if target_module is None:
        return
    try:
        blocks = list(target_module)
    except TypeError as exc:
        raise TypeError(
            f"Target module '{tf_block_module_key_name}' is not an iterable block container"
        ) from exc
    for tf_block in blocks:
        _restore_block(tf_block)

def register_cache_Dream(model: nn.Module, tf_block_module_key_name: str) -> None:
    target_module = _find_target_module(model, tf_block_module_key_name)
    if target_module is None:
        raise ValueError(
            f"Could not find transformer block module '{tf_block_module_key_name}'"
        )
    try:
        blocks = list(target_module)
    except TypeError as exc:
        raise TypeError(
            f"Target module '{tf_block_module_key_name}' is not an iterable block container"
        ) from exc
    if not blocks:
        raise ValueError(
            f"Transformer block module '{tf_block_module_key_name}' is empty"
        )

    for block_index, tf_block in enumerate(blocks):
        _validate_block(tf_block, block_index)

    newly_hooked = []
    try:
        for tf_block in blocks:
            if getattr(tf_block, _HOOK_MARKER, False):
                continue
            newly_hooked.append(tf_block)
            setattr(tf_block, "_old_forward", tf_block.forward)
            setattr(tf_block.self_attn, "_old_forward", tf_block.self_attn.forward)
            tf_block.forward = types.MethodType(decoder_hook, tf_block)
            tf_block.self_attn.forward = types.MethodType(
                attention, tf_block.self_attn
            )
            setattr(tf_block, _HOOK_MARKER, True)
    except Exception:
        for tf_block in reversed(newly_hooked):
            _restore_block(tf_block)
        raise


def _normalize_q_index(
    q_index: Optional[torch.Tensor], batch_size: int, device: torch.device
) -> Optional[torch.Tensor]:
    if q_index is None:
        return None
    if q_index.ndim == 1:
        q_index = q_index.unsqueeze(0)
    if q_index.ndim != 2:
        raise ValueError(f"q_index must be 1D or 2D, got shape {tuple(q_index.shape)}")
    if q_index.shape[0] == 1 and batch_size != 1:
        q_index = q_index.expand(batch_size, -1)
    elif q_index.shape[0] != batch_size:
        raise ValueError(
            f"q_index batch size {q_index.shape[0]} does not match query batch "
            f"size {batch_size}"
        )
    return q_index.to(device=device, dtype=torch.long)


def _select_attention_mask(
    attention_mask: torch.Tensor,
    q_index: Optional[torch.Tensor],
    query_len: int,
    key_len: int,
) -> torch.Tensor:
    if attention_mask.ndim == 2:
        if attention_mask.shape[-1] < key_len:
            raise ValueError(
                f"attention_mask key length {attention_mask.shape[-1]} is smaller "
                f"than the required key length {key_len}"
            )
        return attention_mask[:, None, None, :key_len]
    if attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(1)
    if attention_mask.ndim != 4:
        raise ValueError(
            "attention_mask must have 2, 3, or 4 dimensions, "
            f"got shape {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[-1] < key_len:
        raise ValueError(
            f"attention_mask key length {attention_mask.shape[-1]} is smaller than "
            f"the required key length {key_len}"
        )

    attention_mask = attention_mask[..., :key_len]
    if attention_mask.shape[-2] == 1:
        return attention_mask
    if q_index is None:
        if attention_mask.shape[-2] < query_len:
            raise ValueError(
                f"attention_mask query length {attention_mask.shape[-2]} is smaller "
                f"than the required query length {query_len}"
            )
        return attention_mask[..., -query_len:, :]

    if attention_mask.shape[0] == 1 and q_index.shape[0] != 1:
        attention_mask = attention_mask.expand(q_index.shape[0], -1, -1, -1)
    elif attention_mask.shape[0] != q_index.shape[0]:
        raise ValueError(
            f"attention_mask batch size {attention_mask.shape[0]} does not match "
            f"q_index batch size {q_index.shape[0]}"
        )
    mask_q_index = q_index.to(device=attention_mask.device)
    gather_index = mask_q_index[:, None, :, None].expand(
        -1, attention_mask.shape[1], -1, key_len
    )
    return torch.gather(attention_mask, dim=2, index=gather_index)


def refresh_index(
    new_features: torch.Tensor,
    cached_features: torch.Tensor = None,
    transfer_ratio: float = 0.5,
    layer_id: int = 0,
) -> torch.Tensor:
    batch_size, gen_len, d_model = new_features.shape
    num_replace = int(gen_len * transfer_ratio)
    if num_replace == 0:
        return torch.empty(
            (batch_size, 0), dtype=torch.long, device=new_features.device
        )
    if cached_features is None:
        raise ValueError("cached_features is required when selecting refresh indices")
    if new_features.shape != cached_features.shape:
        raise ValueError(
            "new_features and cached_features must have the same shape, got "
            f"{tuple(new_features.shape)} and {tuple(cached_features.shape)}"
        )
    cos_sim = torch.nn.functional.cosine_similarity(
        new_features, cached_features, dim=-1
    )
    transfer_index = torch.topk(cos_sim, largest=False, k=num_replace).indices
    return transfer_index


def decoder_hook(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    x = hidden_states
    feature_cache = dLLMCache()
    feature_cache.update_step(self.self_attn.layer_idx)
    prompt_length = feature_cache.prompt_length
    refresh_gen = feature_cache.refresh_gen(layer_id=self.self_attn.layer_idx) or self.self_attn.layer_idx == 0
    refresh_prompt = feature_cache.refresh_prompt(layer_id=self.self_attn.layer_idx) or self.self_attn.layer_idx == 0
    x_prompt = x[:, :prompt_length]
    x_gen = x[:, prompt_length:]
    transfer_ratio = feature_cache.transfer_ratio
    bs, seq_len, dim = x.shape
    transfer = (
        0 < transfer_ratio <= 1
        and int(x_gen.shape[1] * transfer_ratio) > 0
    )

    def project(x):
        x_normed = self.input_layernorm(x)
        q = self.self_attn.q_proj(x_normed)
        k = self.self_attn.k_proj(x_normed)
        v = self.self_attn.v_proj(x_normed)
        return q, k, v

    def compute_mlp(x):
        x_norm = self.post_attention_layernorm(x)
        x = self.mlp(x_norm)
        return x

    if refresh_gen and refresh_prompt:
        q, k, v = project(x)
        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="kv_cache",
            features={"k": k[:, :prompt_length, :], "v": v[:, :prompt_length, :]},
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="kv_cache",
            features={"k": k[:, prompt_length:, :], "v": v[:, prompt_length:, :]},
            cache_type="gen",
        )
        attn = self.self_attn(q, k, v, attention_mask, position_embeddings)
        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="attn",
            features=attn[:, :prompt_length, :],
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="attn",
            features=attn[:, prompt_length:, :],
            cache_type="gen",
        )

    elif refresh_gen and not refresh_prompt:
        q_gen, k_gen, v_gen = project(x_gen)

        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="kv_cache",
            features={"k": k_gen, "v": v_gen},
            cache_type="gen",
        )
        kv_cache_prompt = feature_cache.get_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="kv_cache",
            cache_type="prompt",
        )
        k = torch.cat([kv_cache_prompt["k"], k_gen], dim=1)
        v = torch.cat([kv_cache_prompt["v"], v_gen], dim=1)
        gen_index = (
            (
                torch.arange(seq_len - prompt_length, device=q_gen.device)
                + prompt_length
            )
            .unsqueeze(0)
            .expand(bs, -1)
        )
        attn_gen = self.self_attn(
            q_gen, k, v, attention_mask, position_embeddings, gen_index
        )

        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="attn",
            features=attn_gen,
            cache_type="gen",
        )
        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.self_attn.layer_idx, feature_name="attn", cache_type="prompt"
        )
        attn = torch.cat([att_prompt_cache, attn_gen], dim=1)
    elif not refresh_gen and refresh_prompt:
        q_prompt, k_prompt, v_prompt = project(x_prompt)
        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="kv_cache",
            features={"k": k_prompt, "v": v_prompt},
            cache_type="prompt",
        )
        kv_cache_gen = feature_cache.get_cache(
            layer_id=self.self_attn.layer_idx, feature_name="kv_cache", cache_type="gen"
        )
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.self_attn.layer_idx, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_norm = self.input_layernorm(x_gen)
            v_gen = self.self_attn.v_proj(x_gen_norm)
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.self_attn.layer_idx
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_norm, dim=1, index=index_expanded)
            q_gen_index = self.self_attn.q_proj(x_gen_selected)
            k_gen_index = self.self_attn.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            k_index_expanded = index.unsqueeze(-1).expand_as(k_gen_index)
            kv_cache_gen["k"].scatter_(
                dim=1,
                index=k_index_expanded,
                src=k_gen_index,
            )
            feature_cache.set_cache(
                layer_id=self.self_attn.layer_idx,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
        k = torch.cat([k_prompt, kv_cache_gen["k"]], dim=1)
        v = torch.cat([v_prompt, kv_cache_gen["v"]], dim=1)
        if transfer:
            q_prompt_gen_index = torch.cat([q_prompt, q_gen_index], dim=1)
            prompt_index = (
                torch.arange(prompt_length, device=q_prompt_gen_index.device)
                .unsqueeze(0)
                .expand(bs, -1)
            )
            gen_index = index + prompt_length
            att_prompt_gen_index = self.self_attn(
                q_prompt_gen_index,
                k,
                v,
                attention_mask,
                position_embeddings,
                torch.cat([prompt_index, gen_index], dim=1),
            )
            att_prompt = att_prompt_gen_index[:, :prompt_length, :]
            att_gen_index = att_prompt_gen_index[:, prompt_length:, :]
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.self_attn.layer_idx,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )
        else:
            att_prompt = self.self_attn(
                q_prompt,
                k,
                v,
                attention_mask,
                position_embeddings,
                torch.arange(prompt_length, device=q_prompt.device)
                .unsqueeze(0)
                .expand(bs, -1),
            )
        feature_cache.set_cache(
            layer_id=self.self_attn.layer_idx,
            feature_name="attn",
            features=att_prompt,
            cache_type="prompt",
        )
        attn = torch.cat([att_prompt, att_gen_cache], dim=1)
    else:
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.self_attn.layer_idx, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_norm = self.input_layernorm(x_gen)
            v_gen = self.self_attn.v_proj(x_gen_norm)
            kv_cache_gen = feature_cache.get_cache(
                layer_id=self.self_attn.layer_idx,
                feature_name="kv_cache",
                cache_type="gen",
            )
            kv_cache_prompt = feature_cache.get_cache(
                layer_id=self.self_attn.layer_idx,
                feature_name="kv_cache",
                cache_type="prompt",
            )
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.self_attn.layer_idx
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_norm, dim=1, index=index_expanded)
            q_gen_index = self.self_attn.q_proj(x_gen_selected)
            k_gen_index = self.self_attn.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            k_index_expanded = index.unsqueeze(-1).expand_as(k_gen_index)
            kv_cache_gen["k"].scatter_(
                dim=1,
                index=k_index_expanded,
                src=k_gen_index,
            )
            feature_cache.set_cache(
                layer_id=self.self_attn.layer_idx,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
            k = torch.cat([kv_cache_prompt["k"], kv_cache_gen["k"]], dim=1)
            v = torch.cat([kv_cache_prompt["v"], kv_cache_gen["v"]], dim=1)
            gen_index = index + prompt_length
            att_gen_index = self.self_attn(
                q_gen_index, k, v, attention_mask, position_embeddings, gen_index
            )
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.self_attn.layer_idx,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )
        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.self_attn.layer_idx, feature_name="attn", cache_type="prompt"
        )
        attn = torch.cat([att_prompt_cache, att_gen_cache], dim=1)

    x = residual + attn
    residual = x
    x_prompt = x[:, :prompt_length]
    x_gen = x[:, prompt_length:]

    if refresh_gen and refresh_prompt:
        x = compute_mlp(x)
        feature_cache.set_cache(
            self.self_attn.layer_idx, "mlp", x[:, prompt_length:, :], cache_type="gen"
        )
        feature_cache.set_cache(
            self.self_attn.layer_idx,
            "mlp",
            x[:, :prompt_length, :],
            cache_type="prompt",
        )
    elif refresh_gen and not refresh_prompt:
        x_gen = compute_mlp(x_gen)
        feature_cache.set_cache(
            self.self_attn.layer_idx, "mlp", x_gen, cache_type="gen"
        )
        x_prompt_cache = feature_cache.get_cache(
            self.self_attn.layer_idx, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen], dim=1)
    elif not refresh_gen and refresh_prompt:
        x_gen_cache = feature_cache.get_cache(
            self.self_attn.layer_idx, "mlp", cache_type="gen"
        )
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_prompt_gen_index = torch.cat([x_prompt, x_gen_selected], dim=1)
            x_prompt_gen_index = compute_mlp(x_prompt_gen_index)
            x_prompt = x_prompt_gen_index[:, :prompt_length, :]
            x_gen_index = x_prompt_gen_index[:, prompt_length:, :]
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(
                self.self_attn.layer_idx, "mlp", x_gen_cache, cache_type="gen"
            )
        else:
            x_prompt = compute_mlp(x_prompt)
        feature_cache.set_cache(
            self.self_attn.layer_idx, "mlp", x_prompt, cache_type="prompt"
        )
        x = torch.cat([x_prompt, x_gen_cache], dim=1)
    else:
        x_gen_cache = feature_cache.get_cache(
            self.self_attn.layer_idx, "mlp", cache_type="gen"
        )
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_gen_index = compute_mlp(x_gen_selected)
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(
                self.self_attn.layer_idx, "mlp", x_gen_cache, cache_type="gen"
            )
        x_prompt_cache = feature_cache.get_cache(
            self.self_attn.layer_idx, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen_cache], dim=1)

    x = residual + x
    outputs = (x,)
    return outputs


def attention(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    q_index: torch.Tensor = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = query_states.size()
    bsz, k_len, _ = key_states.size()
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, k_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, k_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    q_index = _normalize_q_index(q_index, bsz, query_states.device)
    if q_index is not None and q_index.shape[1] != q_len:
        raise ValueError(
            f"q_index length {q_index.shape[1]} does not match query length {q_len}"
        )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, q_index
    )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # Dream's generation code uses the string sentinel ``"full"`` for an
    # unmasked sequence.  The original Dream SDPA attention only forwards a
    # mask when it is a tensor, so preserve that remote-code contract here.
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = _select_attention_mask(
            attention_mask, q_index, q_len, k_len
        )
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
    else:
        attention_mask = None
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output


def apply_rotary_pos_emb(q, k, cos, sin, q_index=None, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q_index is not None:
        q_index = _normalize_q_index(q_index, q.shape[0], q.device)
        if q_index.shape[1] != q.shape[-2]:
            raise ValueError(
                f"q_index length {q_index.shape[1]} does not match query length "
                f"{q.shape[-2]}"
            )
        bs, _ = q_index.shape
        q_embed = []
        if cos.shape[0] != q_index.shape[0] or sin.shape[0] != q_index.shape[0]:
            cos = cos.repeat(q_index.shape[0] // cos.shape[0], 1, 1, 1)
            sin = sin.repeat(q_index.shape[0] // sin.shape[0], 1, 1, 1)
        for i in range(bs):
            q_i = (q[i].squeeze(0) * cos[i, :, q_index[i]]) + (
                rotate_half(q[i].squeeze(0)) * sin[i, :, q_index[i]]
            )
            q_embed.append(q_i.unsqueeze(0))
        q_embed = torch.cat(q_embed, dim=0)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
