import importlib
from unittest.mock import Mock, patch

import pytest
import torch

from eval_model.Dream import Dream
from eval_model.LLaDA import LLaDA


def test_llada_config_loader_omits_none_subfolder():
    adapter = object.__new__(LLaDA)
    sentinel = object()
    llada_module = importlib.import_module("eval_model.LLaDA")

    with patch.object(
        llada_module.transformers.AutoConfig,
        "from_pretrained",
        return_value=sentinel,
    ) as from_pretrained:
        adapter._get_config(
            "example/model",
            revision="fixed-revision",
            subfolder=None,
            trust_remote_code=True,
        )

    assert adapter._config is sentinel
    _, kwargs = from_pretrained.call_args
    assert kwargs["revision"] == "fixed-revision"
    assert "subfolder" not in kwargs


@pytest.mark.parametrize("adapter_type", [LLaDA, Dream])
def test_eval_adapter_distributed_primitives(adapter_type):
    adapter = object.__new__(adapter_type)
    adapter._rank = 1
    adapter._world_size = 2
    adapter.accelerator = Mock()
    tensor = torch.tensor(3)
    gathered = torch.tensor([2, 3])
    adapter.accelerator.gather.return_value = gathered

    assert adapter.all_gather(tensor) is gathered
    adapter.accelerator.gather.assert_called_once_with(tensor)

    with patch.object(torch.distributed, "gather_object") as gather_object:
        assert adapter.gather_object({"rank": 1}, dst=0) is None
        gather_object.assert_called_once_with(
            obj={"rank": 1}, object_gather_list=None, dst=0
        )

    adapter.barrier()
    adapter.accelerator.wait_for_everyone.assert_called_once_with()


@pytest.mark.parametrize("adapter_type", [LLaDA, Dream])
def test_eval_adapter_single_process_primitives_are_noops(adapter_type):
    adapter = object.__new__(adapter_type)
    adapter._rank = 0
    adapter._world_size = 1
    tensor = torch.tensor(7)

    assert adapter.all_gather(tensor) is tensor
    assert adapter.gather_object("sample") == ["sample"]
    assert adapter.barrier() is None
