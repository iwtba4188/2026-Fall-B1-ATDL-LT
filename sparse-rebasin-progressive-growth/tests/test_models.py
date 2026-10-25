import torch

from models import VGG11_nofc


def test_vgg11_mnist_forward_keeps_spatial_dimensions_valid():
    model = VGG11_nofc(
        {"dataset": "mnist"},
        "VGG11",
        num_classes=10,
        init_weights=False,
    )

    output = model(torch.zeros(2, 1, 28, 28))

    assert output.shape == (2, 10)
