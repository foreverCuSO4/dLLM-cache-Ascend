import torch
from types import SimpleNamespace

from utils.generate_function import generate, get_num_transfer_tokens


def test_transfer_tokens_are_distributed_across_steps():
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, False, False, False]]
    )
    result = get_num_transfer_tokens(mask, steps=3)
    assert result.dtype == torch.int64
    assert result.tolist() == [[2, 2, 1], [1, 1, 0]]


def test_transfer_tokens_handle_empty_mask():
    mask = torch.zeros((2, 4), dtype=torch.bool)
    result = get_num_transfer_tokens(mask, steps=2)
    assert result.tolist() == [[0, 0], [0, 0]]


def test_generate_extends_prompt_padding_mask_over_generation_region():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_masks = []

        @property
        def device(self):
            return torch.device("cpu")

        def forward(self, input_ids, attention_mask=None):
            self.seen_masks.append(attention_mask.clone())
            logits = torch.zeros(*input_ids.shape, 6)
            logits[..., 1] = 1
            return SimpleNamespace(logits=logits)

    model = RecordingModel()
    input_ids = torch.tensor([[0, 2], [3, 4]])
    attention_mask = torch.tensor([[0, 1], [1, 1]])

    generate(
        input_ids,
        attention_mask,
        model,
        steps=2,
        gen_length=2,
        block_length=2,
        mask_id=5,
    )

    assert model.seen_masks
    expected = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    assert all(torch.equal(mask, expected) for mask in model.seen_masks)
