import pytest
import torch

from dllm_cache.cache import dLLMCache, dLLMCacheConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_interval_steps", 0),
        ("gen_interval_steps", 0),
        ("cfg_interval_steps", 0),
        ("transfer_ratio", -0.01),
        ("transfer_ratio", 1.01),
    ],
)
def test_config_rejects_invalid_cache_parameters(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        dLLMCacheConfig(**kwargs)
    with pytest.raises(ValueError):
        dLLMCache.new_instance(**kwargs)


def test_refresh_intervals_follow_step_counter():
    cache = dLLMCache.new_instance(
        prompt_interval_steps=2,
        gen_interval_steps=3,
        cfg_interval_steps=4,
        transfer_ratio=0.25,
    )
    cache.reset_cache(prompt_length=7)

    prompt_refreshes = []
    generation_refreshes = []
    for _ in range(7):
        cache.update_step(layer_id=0)
        prompt_refreshes.append(cache.refresh_prompt())
        generation_refreshes.append(cache.refresh_gen())

    assert prompt_refreshes == [True, False, True, False, True, False, True]
    assert generation_refreshes == [True, False, False, True, False, False, True]
    assert cache.prompt_length == 7


def test_reset_releases_entries_without_calling_cuda(monkeypatch):
    cache = dLLMCache.new_instance()
    cache.reset_cache(prompt_length=2)
    tensor = torch.ones(1, 2, 3)
    cache.set_cache(0, "feature", tensor, "prompt")
    assert cache.get_cache(0, "feature", "prompt") is tensor

    def fail_if_called():
        raise AssertionError("reset_cache must not manipulate a backend allocator")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail_if_called)
    cache.reset_cache(prompt_length=4)
    assert cache.prompt_length == 4
    with pytest.raises((KeyError, IndexError)):
        cache.get_cache(0, "feature", "prompt")
