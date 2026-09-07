from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dllm_cache.hooks.cache_hook_Dream import (
    _select_attention_mask,
    attention as dream_attention_hook,
    decoder_hook,
    logout_cache_Dream,
    register_cache_Dream,
)
from dllm_cache.hooks.cache_hook_LLaDA import (
    _attention as llada_attention_hook,
    _select_attention_bias,
    cache_hook_feature,
    logout_cache_LLaDA,
    register_cache_LLaDA,
)


class DummyRotary(nn.Module):
    def forward(self, q, k):
        return q, k


class DummyLLaDABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_id = 0
        self.rotary_emb = DummyRotary()
        self.attn_norm = nn.Identity()
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.dropout = nn.Identity()
        self.ff_norm = nn.Identity()
        self.ff_proj = nn.Identity()
        self.up_proj = nn.Identity()
        self.act = nn.Identity()
        self.ff_out = nn.Identity()
        self.q_norm = None
        self.k_norm = None
        self._activation_checkpoint_fn = None
        self.config = SimpleNamespace()
        self.attn_out = nn.Identity()

    def _cast_attn_bias(self, bias, dtype):
        return bias.to(dtype)

    def _scaled_dot_product_attention(
        self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    def attention(
        self,
        q,
        k,
        v,
        attention_bias=None,
        layer_past=None,
        use_cache=False,
    ):
        return q, None

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False):
        return x


class DummyDreamAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = 4
        self.hidden_size = 4
        self.attention_dropout = 0.0
        self.o_proj = nn.Identity()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return hidden_states


class DummyDreamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyDreamAttention()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        return hidden_states


def llada_model():
    root = nn.Module()
    root.model = nn.Module()
    root.model.transformer = nn.Module()
    root.model.transformer.blocks = nn.ModuleList([DummyLLaDABlock()])
    return root


def dream_model():
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([DummyDreamBlock()])
    return root


def test_llada_hook_registration_is_idempotent_and_reversible():
    model = llada_model()
    block = model.model.transformer.blocks[0]
    original_forward = block.forward.__func__
    register_cache_LLaDA(model, "model.transformer.blocks")
    register_cache_LLaDA(model, "model.transformer.blocks")
    assert block.forward.__func__ is cache_hook_feature
    assert block.attention.__func__ is llada_attention_hook
    logout_cache_LLaDA(model, "model.transformer.blocks")
    logout_cache_LLaDA(model, "model.transformer.blocks")
    assert block.forward.__func__ is original_forward


def test_dream_hook_registration_is_idempotent_and_reversible():
    model = dream_model()
    block = model.model.layers[0]
    original_forward = block.forward.__func__
    register_cache_Dream(model, "model.layers")
    register_cache_Dream(model, "model.layers")
    assert block.forward.__func__ is decoder_hook
    assert block.self_attn.forward.__func__ is dream_attention_hook
    logout_cache_Dream(model, "model.layers")
    logout_cache_Dream(model, "model.layers")
    assert block.forward.__func__ is original_forward


@pytest.mark.parametrize(
    ("register", "model"),
    [(register_cache_LLaDA, llada_model), (register_cache_Dream, dream_model)],
)
def test_register_rejects_missing_target(register, model):
    with pytest.raises(ValueError, match="Could not find"):
        register(model(), "does.not.exist")


def test_attention_mask_helpers_preserve_batch_and_select_query_rows():
    two_dimensional = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    assert _select_attention_bias(two_dimensional, None, 2, 3).shape == (2, 1, 1, 3)
    assert _select_attention_mask(two_dimensional, None, 2, 3).shape == (2, 1, 1, 3)

    three_dimensional = torch.ones(2, 4, 4, dtype=torch.bool)
    assert _select_attention_bias(three_dimensional, None, 4, 4).shape == (
        2,
        1,
        4,
        4,
    )
    assert _select_attention_mask(three_dimensional, None, 4, 4).shape == (
        2,
        1,
        4,
        4,
    )

    full = torch.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4)
    q_index = torch.tensor([[0, 2], [1, 3]])
    llada_selected = _select_attention_bias(full, q_index, 2, 4)
    dream_selected = _select_attention_mask(full, q_index, 2, 4)
    expected = torch.stack(
        [full[0, :, [0, 2], :], full[1, :, [1, 3], :]], dim=0
    )
    assert torch.equal(llada_selected, expected)
    assert torch.equal(dream_selected, expected)


def test_dream_attention_accepts_full_mask_sentinel():
    attention = DummyDreamAttention()
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 3, 4)
    value = torch.randn(2, 3, 4)
    position_embeddings = (torch.ones(2, 3, 4), torch.zeros(2, 3, 4))

    output = dream_attention_hook(
        attention, query, key, value, "full", position_embeddings
    )

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_llada_attention_supports_grouped_query_attention_widths():
    block = DummyLLaDABlock()
    block.config = SimpleNamespace(
        n_heads=4,
        effective_n_kv_heads=2,
        rope=False,
        attention_dropout=0.0,
    )
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 4)

    output, present = llada_attention_hook(block, query, key, value)

    assert output.shape == query.shape
    assert present is None
