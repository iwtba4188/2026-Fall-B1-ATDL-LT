"""Progressive mask growth utilities.

The progressive method transfers a source lottery-ticket mask to an
independently initialized model, grows that mask through explicit density
targets, and leaves subsequent training/pruning to the existing pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def eligible_modules(model: nn.Module, model_type: str) -> list[tuple[str, nn.Module]]:
    """Return layers whose weights participate in the global mask."""
    excluded = ()
    if model_type == "ResNet":
        excluded = ("shortcut",)
    elif model_type == "ResNet50":
        excluded = ("shortcut", "downsample")
    elif model_type not in {"VGG11", "LeNet"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and not any(part in name for part in excluded)
    ]


def _mask(module: nn.Module) -> torch.Tensor:
    if not hasattr(module, "weight_mask"):
        raise ValueError("All progressive layers must have a weight_mask")
    return module.weight_mask.detach()


def mask_statistics(model: nn.Module, model_type: str) -> tuple[int, int]:
    """Return active and total eligible parameter counts."""
    active = total = 0
    for _, module in eligible_modules(model, model_type):
        mask = _mask(module)
        active += int(mask.count_nonzero().item())
        total += mask.numel()
    return active, total


def validate_mask_compatibility(
    source_model: nn.Module, target_model: nn.Module, model_type: str
) -> None:
    source_layers = eligible_modules(source_model, model_type)
    target_layers = eligible_modules(target_model, model_type)
    if [name for name, _ in source_layers] != [name for name, _ in target_layers]:
        raise ValueError("Source and target progressive layers do not match")
    for (name_a, source), (name_b, target) in zip(source_layers, target_layers):
        if source.weight.shape != target.weight.shape:
            raise ValueError(
                f"Layer shape mismatch for {name_a}/{name_b}: "
                f"{tuple(source.weight.shape)} != {tuple(target.weight.shape)}"
            )


def _install_masks(
    model: nn.Module, masks: dict[str, torch.Tensor], model_type: str
) -> None:
    """Install masks without discarding the original dense weights."""
    for name, module in eligible_modules(model, model_type):
        if hasattr(module, "weight_mask"):
            module.weight_mask.data.copy_(
                masks[name].to(
                    device=module.weight_mask.device, dtype=module.weight_mask.dtype
                )
            )
        else:
            prune.custom_from_mask(
                module, "weight", masks[name].to(device=module.weight.device)
            )


def transfer_mask(
    source_model: nn.Module, target_model: nn.Module, model_type: str
) -> nn.Module:
    """Apply a source mask to target weights without copying source weights."""
    validate_mask_compatibility(source_model, target_model, model_type)
    masks = {
        name: _mask(source).to(dtype=target.weight.dtype).clone()
        for (name, source), (_, target) in zip(
            eligible_modules(source_model, model_type),
            eligible_modules(target_model, model_type),
        )
    }
    _install_masks(target_model, masks, model_type)
    return target_model


def grow_mask(
    model: nn.Module,
    model_type: str,
    target_density: float,
    selection: str = "magnitude",
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Grow inactive connections until the requested global density is met.

    Growth never removes an active connection. Magnitude selection uses the
    current effective target weights, ``gradient`` uses the absolute
    gradients of the dense weights, and random selection is available for
    controlled ablations.
    """
    if not 0.0 <= target_density <= 1.0:
        raise ValueError("target_density must be between 0 and 1")
    if selection not in {"magnitude", "random", "gradient"}:
        raise ValueError("selection must be 'magnitude', 'random', or 'gradient'")

    layers = eligible_modules(model, model_type)
    old_active, total = mask_statistics(model, model_type)
    target_active = int(round(target_density * total))
    if target_active < old_active:
        raise ValueError(
            f"Progressive growth cannot reduce active weights: "
            f"{old_active} -> {target_active}"
        )
    add_count = target_active - old_active
    if add_count == 0:
        return old_active, total

    candidate_indices: list[torch.Tensor] = []
    candidate_scores: list[torch.Tensor] = []
    for name, module in layers:
        mask = _mask(module).bool()
        weights = getattr(module, "weight_orig", module.weight)
        if selection == "gradient":
            gradients = weights.grad
            if gradients is None:
                raise ValueError(
                    "gradient selection requires gradients on all progressive layers"
                )
            scores = gradients.detach().abs()
        else:
            scores = weights.detach().abs()
        inactive = ~mask.reshape(-1)
        candidate_indices.append(inactive.nonzero(as_tuple=False).flatten())
        candidate_scores.append(scores.reshape(-1)[inactive])

    candidate_count = sum(indices.numel() for indices in candidate_indices)
    if add_count > candidate_count:
        raise ValueError("Requested density exceeds available inactive weights")

    scores = torch.cat(candidate_scores)
    if selection == "magnitude":
        _, selected = torch.topk(scores, add_count, largest=True, sorted=False)
    elif selection == "random":
        order = torch.randperm(candidate_count, generator=generator)
        selected = order[:add_count].to(device=scores.device)
    else:
        _, selected = torch.topk(scores, add_count, largest=True, sorted=False)

    masks = {name: _mask(module).clone() for name, module in layers}
    offsets = torch.tensor(
        [
            0,
            *torch.cumsum(
                torch.tensor(
                    [indices.numel() for indices in candidate_indices],
                    dtype=torch.long,
                ),
                dim=0,
            ).tolist(),
        ],
        device=scores.device,
    )
    # A candidate at a cumulative boundary belongs to the following layer.
    layer_indices = torch.bucketize(selected, offsets[1:-1], right=True)
    for layer_index, (name, _) in enumerate(layers):
        selected_in_layer = selected[layer_indices == layer_index]
        if selected_in_layer.numel() == 0:
            continue
        local_indices = selected_in_layer - offsets[layer_index]
        flat_mask = masks[name].reshape(-1)
        flat_mask[candidate_indices[layer_index][local_indices]] = 1
    _install_masks(model, masks, model_type)
    return mask_statistics(model, model_type)


def assert_density(
    model: nn.Module, model_type: str, expected_density: float, tolerance: float = 1e-6
) -> None:
    active, total = mask_statistics(model, model_type)
    actual = active / total
    if abs(actual - expected_density) > tolerance:
        raise AssertionError(
            f"Expected density {expected_density:.12f}, got {actual:.12f}"
        )
