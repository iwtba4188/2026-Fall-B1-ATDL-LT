import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from progressive import (
    assert_density,
    grow_mask,
    mask_retention_statistics,
    mask_statistics,
    snapshot_masks,
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
    assert active == int(
        source[0].weight_mask.count_nonzero() + source[1].weight_mask.count_nonzero()
    )
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


def test_gradient_growth_uses_dense_weight_gradients():
    model = _masked_model()
    model[0].weight_mask.data.zero_()
    model[1].weight_mask.data.zero_()
    model[0].weight_orig.grad = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 7.0],
        ]
    )
    model[1].weight_orig.grad = torch.zeros_like(model[1].weight_orig)
    _, total = mask_statistics(model, "LeNet")

    grow_mask(model, "LeNet", 1 / total, selection="gradient")

    assert model[0].weight_mask[0, 0].item() == 1


def test_gradient_growth_requires_gradients():
    model = _masked_model()

    try:
        grow_mask(model, "LeNet", 0.5, selection="gradient")
    except ValueError as error:
        assert "requires gradients" in str(error)
    else:
        raise AssertionError("gradient growth should require parameter gradients")


def test_mask_retention_statistics_tracks_initial_and_added_nodes():
    model = _masked_model()
    for layer in model:
        layer.weight_mask.data.zero_()
    model[0].weight_mask.data.reshape(-1)[:8] = 1
    initial_masks = snapshot_masks(model, "LeNet")
    grow_mask(model, "LeNet", 0.75)
    grown_masks = snapshot_masks(model, "LeNet")

    for layer in model:
        layer.weight_mask.data.zero_()
    model[0].weight_mask.data.reshape(-1)[0] = 1
    for name, layer in zip(("0", "1"), model):
        added = grown_masks[name] & ~initial_masks[name]
        if added.any():
            index = added.reshape(-1).nonzero(as_tuple=False)[0].item()
            layer.weight_mask.data.reshape(-1)[index] = 1
            break

    stats = mask_retention_statistics(model, "LeNet", initial_masks, grown_masks)

    assert stats["initial_retained"] == 1
    assert stats["added_retained"] == 1
    assert stats["initial_retention"] == 0.5
