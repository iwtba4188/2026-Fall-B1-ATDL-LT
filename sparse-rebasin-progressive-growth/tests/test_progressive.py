import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from progressive import (
    assert_density,
    grow_mask,
    mask_statistics,
    transfer_mask,
)


def _masked_model():
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False))
    for layer in model:
        prune.custom_from_mask(layer, "weight", torch.ones_like(layer.weight))
    return model


def test_transfer_and_growth_are_monotonic():
    source = _masked_model()
    target = _masked_model()
    source[0].weight_mask[:2] = 0
    source[1].weight_mask[:, :1] = 0

    transfer_mask(source, target, "LeNet")
    active, total = mask_statistics(target, "LeNet")
    assert active == int(source[0].weight_mask.count_nonzero() + source[1].weight_mask.count_nonzero())
    assert total == source[0].weight_mask.numel() + source[1].weight_mask.numel()

    initial_density = active / total
    target_density = initial_density + 2 / total
    grow_mask(target, "LeNet", target_density)
    assert_density(target, "LeNet", target_density)


def test_growth_rejects_decreasing_density():
    model = _masked_model()
    try:
        grow_mask(model, "LeNet", 0.5)
    except ValueError as error:
        assert "cannot reduce" in str(error)
    else:
        raise AssertionError("decreasing progressive density should fail")


def test_magnitude_growth_uses_original_dense_weights():
    model = _masked_model()
    model[0].weight_orig.data.copy_(
        torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 7.0],
            ]
        )
    )
    model[0].weight_mask.data.zero_()
    model[1].weight_orig.data.zero_()
    active_before, total = mask_statistics(model, "LeNet")

    grow_mask(model, "LeNet", (active_before + 1) / total)

    assert model[0].weight_mask[0, 0].item() == 1
    assert model[0].weight_orig[0, 0].item() == 10.0
