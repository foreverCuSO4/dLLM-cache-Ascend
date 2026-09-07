import torch
from typing import Optional, Tuple
import torch.nn as nn
import inspect
import types
from dllm_cache.cache import dLLMCache


_HOOK_MARKER = "_dllm_cache_llada_hooked"


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
    required_attributes = (
        "layer_id",
        "attention",
        "rotary_emb",
        "attn_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "dropout",
        "ff_norm",
        "ff_proj",
        "up_proj",
        "act",
        "ff_out",
        "_activation_checkpoint_fn",
        "q_norm",
        "k_norm",
        "config",
        "_cast_attn_bias",
        "_scaled_dot_product_attention",
        "attn_out",
    )
    missing = [name for name in required_attributes if not hasattr(tf_block, name)]
    if missing:
        raise TypeError(
            f"Transformer block {block_index} is missing required attributes: "
            f"{', '.join(missing)}"
        )
    if not hasattr(tf_block.rotary_emb, "forward"):
        raise TypeError(f"Transformer block {block_index} rotary_emb has no forward")

    if getattr(tf_block, _HOOK_MARKER, False):
        correctly_hooked = (
            _is_hooked_method(tf_block.forward, cache_hook_feature)
            and _is_hooked_method(tf_block.attention, _attention)
            and _is_hooked_method(tf_block.rotary_emb.forward, RoPe_forward)
        )
        if not correctly_hooked:
            raise RuntimeError(
                f"Transformer block {block_index} has an inconsistent LLaDA hook state"
            )
        return

    conflicts = [
        name
        for owner, name in (
            (tf_block, "_old_forward"),
            (tf_block, "_old_attention"),
            (tf_block.rotary_emb, "_old_forward"),
        )
        if hasattr(owner, name)
    ]
    if conflicts:
        raise RuntimeError(
            f"Transformer block {block_index} already contains hook backup "
            f"attributes: {', '.join(conflicts)}"
        )

    _has_parameters(
        tf_block.forward,
        ("x", "attention_bias", "layer_past", "use_cache"),
        f"transformer block {block_index} forward",
    )
    _has_parameters(
        tf_block.attention,
        ("q", "k", "v", "attention_bias", "layer_past", "use_cache"),
        f"transformer block {block_index} attention",
    )
    _has_parameters(
        tf_block.rotary_emb.forward,
        ("q", "k"),
        f"transformer block {block_index} rotary forward",
    )


def _restore_block(tf_block: nn.Module) -> None:
    if hasattr(tf_block, "_old_forward"):
        tf_block.forward = tf_block._old_forward
        delattr(tf_block, "_old_forward")
    if hasattr(tf_block, "_old_attention"):
        tf_block.attention = tf_block._old_attention
        delattr(tf_block, "_old_attention")
    rotary_emb = getattr(tf_block, "rotary_emb", None)
    if rotary_emb is not None and hasattr(rotary_emb, "_old_forward"):
        rotary_emb.forward = rotary_emb._old_forward
        delattr(rotary_emb, "_old_forward")
    if hasattr(tf_block, _HOOK_MARKER):
        delattr(tf_block, _HOOK_MARKER)



def logout_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
    """Restore original functions for transformer blocks, attention, and rotary embeddings."""
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
            

def register_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
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
            setattr(tf_block, "_old_attention", tf_block.attention)
            setattr(tf_block.rotary_emb, "_old_forward", tf_block.rotary_emb.forward)
            tf_block.forward = types.MethodType(cache_hook_feature, tf_block)
            tf_block.attention = types.MethodType(_attention, tf_block)
            tf_block.rotary_emb.forward = types.MethodType(
                RoPe_forward, tf_block.rotary_emb
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


def _select_attention_bias(
    attention_bias: torch.Tensor,
    q_index: Optional[torch.Tensor],
    query_len: int,
    key_len: int,
) -> torch.Tensor:
    if attention_bias.ndim == 2:
        attention_bias = attention_bias[:, None, None, :]
    elif attention_bias.ndim == 3:
        attention_bias = attention_bias.unsqueeze(1)
    if attention_bias.ndim != 4:
        raise ValueError(
            "attention_bias must have 2, 3, or 4 dimensions, "
            f"got shape {tuple(attention_bias.shape)}"
        )
    if attention_bias.shape[-1] < key_len:
        raise ValueError(
            f"attention_bias key length {attention_bias.shape[-1]} is smaller than "
            f"the required key length {key_len}"
        )

    attention_bias = attention_bias[..., :key_len]
    if attention_bias.shape[-2] == 1:
        return attention_bias
    if q_index is None:
        if attention_bias.shape[-2] < query_len:
            raise ValueError(
                f"attention_bias query length {attention_bias.shape[-2]} is smaller "
                f"than the required query length {query_len}"
            )
        return attention_bias[..., -query_len:, :]

    if attention_bias.shape[0] == 1 and q_index.shape[0] != 1:
        attention_bias = attention_bias.expand(
            q_index.shape[0], -1, -1, -1
        )
    elif attention_bias.shape[0] != q_index.shape[0]:
        raise ValueError(
            f"attention_bias batch size {attention_bias.shape[0]} does not match "
            f"q_index batch size {q_index.shape[0]}"
        )
    mask_q_index = q_index.to(device=attention_bias.device)
    gather_index = mask_q_index[:, None, :, None].expand(
        -1, attention_bias.shape[1], -1, key_len
    )
    return torch.gather(attention_bias, dim=2, index=gather_index)


def _attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    q_index: torch.Tensor = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

    batch_size, q_len, q_width = q.size()
    key_batch_size, k_len, k_width = k.size()
    value_batch_size, v_len, v_width = v.size()
    if key_batch_size != batch_size or value_batch_size != batch_size:
        raise ValueError("q, k, and v must have the same batch size")
    if q_width % self.config.n_heads:
        raise ValueError(
            f"query width {q_width} is not divisible by {self.config.n_heads} heads"
        )
    head_dim = q_width // self.config.n_heads
    expected_kv_width = self.config.effective_n_kv_heads * head_dim
    if k_width != expected_kv_width or v_width != expected_kv_width:
        raise ValueError(
            "key and value widths must equal effective_n_kv_heads * head_dim "
            f"({expected_kv_width}), got {k_width} and {v_width}"
        )
    dtype = k.dtype
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)
    q = q.view(
        batch_size, q_len, self.config.n_heads, head_dim
    ).transpose(1, 2)
    k = k.view(
        batch_size, k_len, self.config.effective_n_kv_heads, head_dim
    ).transpose(1, 2)
    v = v.view(
        batch_size, v_len, self.config.effective_n_kv_heads, head_dim
    ).transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)
    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]
    q_index = _normalize_q_index(q_index, batch_size, q.device)
    if q_index is not None and q_index.shape[1] != query_len:
        raise ValueError(
            f"q_index length {q_index.shape[1]} does not match query length {query_len}"
        )
    if self.config.rope:
        q, k = self.rotary_emb(q, k, q_index=q_index)
    if attention_bias is not None:
        attention_bias = _select_attention_bias(
            attention_bias, q_index, query_len, key_len
        )
        attention_bias = self._cast_attn_bias(
            attention_bias, dtype
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    att = self._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_bias,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    att = att.transpose(1, 2).contiguous().view(batch_size, q_len, q_width)
    return self.attn_out(att), present


def RoPe_forward(
    self, q: torch.Tensor, k: torch.Tensor, q_index: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.config.rope_full_precision:
        q_, k_ = q.float(), k.float()
    else:
        q_, k_ = q, k
    with torch.autocast(q.device.type, enabled=False):
        query_len, key_len = q_.shape[-2], k_.shape[-2]
        pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
        pos_sin = pos_sin.type_as(q_)
        pos_cos = pos_cos.type_as(q_)
        if q_index is not None:
            q_index = _normalize_q_index(q_index, q_.shape[0], q_.device)
            if q_index.shape[1] != query_len:
                raise ValueError(
                    f"q_index length {q_index.shape[1]} does not match query "
                    f"length {query_len}"
                )
            bs, _ = q_index.shape
            q_list = []
            for i in range(bs):
                q_i = self.apply_rotary_pos_emb(
                    pos_sin[:, :, q_index[i], :],
                    pos_cos[:, :, q_index[i], :],
                    q_[i].unsqueeze(0),
                )
                q_list.append(q_i)
            q_ = torch.cat(q_list, dim=0)
        else:
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
        k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
    return q_.type_as(q), k_.type_as(k)


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


def cache_hook_feature(
    self,
    x: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    feature_cache = dLLMCache()
    feature_cache.update_step(self.layer_id)
    prompt_length = feature_cache.prompt_length
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]
    refresh_gen = feature_cache.refresh_gen(layer_id=self.layer_id) or self.layer_id == 0
    refresh_prompt = feature_cache.refresh_prompt(layer_id=self.layer_id) or self.layer_id == 0
    transfer_ratio = feature_cache.transfer_ratio
    bs, seq_len, dim = x.shape
    transfer = (
        0 < transfer_ratio <= 1
        and int(x_gen.shape[1] * transfer_ratio) > 0
    )

    def attention(q, k, v, q_index: torch.Tensor = None):
        if self._activation_checkpoint_fn is not None:
            att, _ = self._activation_checkpoint_fn(
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                q_index=q_index,
            )
        else:
            att, _ = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                q_index=q_index,
            )
        return att

    def compute_mlp(input_x):

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, input_x)
        else:
            x = self.ff_norm(input_x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = x * x_up
        return self.ff_out(x)

    def project(x):
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        return q, k, v

    if refresh_gen and refresh_prompt:
        q, k, v = project(x)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k[:, :prompt_length, :], "v": v[:, :prompt_length, :]},
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k[:, prompt_length:, :], "v": v[:, prompt_length:, :]},
            cache_type="gen",
        )
        att = attention(q, k, v)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att[:, :prompt_length, :],
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att[:, prompt_length:, :],
            cache_type="gen",
        )

    elif refresh_gen and not refresh_prompt:
        q, k_gen, v_gen = project(x_gen)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k_gen, "v": v_gen},
            cache_type="gen",
        )
        kv_cache_prompt = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="kv_cache", cache_type="prompt"
        )
        k = torch.cat([kv_cache_prompt["k"], k_gen], dim=1)
        v = torch.cat([kv_cache_prompt["v"], v_gen], dim=1)
        att_gen = attention(q, k, v)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att_gen,
            cache_type="gen",
        )
        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="prompt"
        )
        att = torch.cat([att_prompt_cache, att_gen], dim=1)

    elif not refresh_gen and refresh_prompt:
        q_prompt, k_prompt, v_prompt = project(x_prompt)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k_prompt, "v": v_prompt},
            cache_type="prompt",
        )
        kv_cache_gen = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="kv_cache", cache_type="gen"
        )
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_normed = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_normed)
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.layer_id
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            k_index_expanded = index.unsqueeze(-1).expand_as(k_gen_index)
            kv_cache_gen["k"].scatter_(
                dim=1, index=k_index_expanded, src=k_gen_index
            )
            feature_cache.set_cache(
                layer_id=self.layer_id,
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
            att_prompt_gen_index = attention(
                q_prompt_gen_index,
                k,
                v,
                q_index=torch.cat([prompt_index, gen_index], dim=1),
            )
            att_prompt = att_prompt_gen_index[:, :prompt_length, :]
            att_gen_index = att_prompt_gen_index[:, prompt_length:, :]
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )
        else:
            att_prompt = attention(
                q_prompt,
                k,
                v,
                q_index=torch.arange(prompt_length, device=q_prompt.device)
                .unsqueeze(0)
                .expand(bs, -1),
            )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att_prompt,
            cache_type="prompt",
        )
        att = torch.cat([att_prompt, att_gen_cache], dim=1)
    else:
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_normed = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_normed)
            kv_cache_gen = feature_cache.get_cache(
                layer_id=self.layer_id, feature_name="kv_cache", cache_type="gen"
            )
            kv_cache_prompt = feature_cache.get_cache(
                layer_id=self.layer_id, feature_name="kv_cache", cache_type="prompt"
            )
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.layer_id
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            k_index_expanded = index.unsqueeze(-1).expand_as(k_gen_index)
            kv_cache_gen["k"].scatter_(
                dim=1, index=k_index_expanded, src=k_gen_index
            )
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
            k = torch.cat([kv_cache_prompt["k"], kv_cache_gen["k"]], dim=1)
            v = torch.cat([kv_cache_prompt["v"], kv_cache_gen["v"]], dim=1)
            att_gen_index = attention(q_gen_index, k, v, q_index=index + prompt_length)
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )

        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="prompt"
        )
        att = torch.cat([att_prompt_cache, att_gen_cache], dim=1)

    x = x + self.dropout(att)

    og_x = x
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]

    if refresh_gen and refresh_prompt:
        x = compute_mlp(x)
        feature_cache.set_cache(
            self.layer_id, "mlp", x[:, prompt_length:, :], cache_type="gen"
        )
        feature_cache.set_cache(
            self.layer_id, "mlp", x[:, :prompt_length, :], cache_type="prompt"
        )

    elif refresh_gen and not refresh_prompt:
        x_gen = compute_mlp(x_gen)
        feature_cache.set_cache(self.layer_id, "mlp", x_gen, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(
            self.layer_id, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen], dim=1)

    elif refresh_prompt and not refresh_gen:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_prompt_gen_index = torch.cat([x_prompt, x_gen_selected], dim=1)
            x_prompt_gen_index = compute_mlp(x_prompt_gen_index)
            x_prompt = x_prompt_gen_index[:, :prompt_length, :]
            x_gen_index = x_prompt_gen_index[:, prompt_length:, :]
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(self.layer_id, "mlp", x_gen_cache, cache_type="gen")
        else:
            x_prompt = compute_mlp(x_prompt)
        feature_cache.set_cache(self.layer_id, "mlp", x_prompt, cache_type="prompt")
        x = torch.cat([x_prompt, x_gen_cache], dim=1)

    else:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_gen_index = compute_mlp(x_gen_selected)
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(self.layer_id, "mlp", x_gen_cache, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(
            self.layer_id, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen_cache], dim=1)

    x = self.dropout(x)
    x = og_x + x

    return x, None
