import pytest
import torch

from dllm_cache.hooks.cache_hook_Dream import refresh_index as dream_refresh_index
from dllm_cache.hooks.cache_hook_LLaDA import refresh_index as llada_refresh_index


@pytest.mark.parametrize("selector", [llada_refresh_index, dream_refresh_index])
def test_refresh_index_handles_zero_transfer(selector):
    current = torch.randn(2, 5, 8)
    cached = current.clone()
    index = selector(current, cached, transfer_ratio=0.0)
    assert index.shape == (2, 0)
    assert index.device == current.device
    assert index.dtype == torch.int64


@pytest.mark.parametrize("selector", [llada_refresh_index, dream_refresh_index])
def test_refresh_index_selects_lowest_cosine_similarity(selector):
    current = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    cached = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    index = selector(current, cached, transfer_ratio=0.5)
    assert set(index[0].tolist()) == {1, 2}
