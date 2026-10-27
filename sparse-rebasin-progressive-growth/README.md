# Progressive LT Mask Transfer

This is the code repository is based on the code repository of paper "Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry", [accepted at ICML 2025](https://openreview.net/forum?id=BnfJSwtHLu), and available on arXiv at http://arxiv.org/abs/2505.05143.

The BibTex citation of the original work:

```bibtex
@inproceedings{mohammed2025sparsetraining,
  author = {Adnan, Mohammed and Jain, Rohan and Sharma, Ekansh and Krishnan, Rahul and Ioannou, Yani},
  title = {Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry},
  booktitle = {Forty-second International Conference on Machine Learning (ICML)},
  year = {2025},
  arxivid = {2505.05143},
  eprint = {2505.05143},
  eprinttype = {arXiv},
  venue = {{Vancouver, BC, Canada}},
  eventdate = {2025-07-13/2025-07-19},
}
```

## Quickstart

1. Install the uv package manager following [uv Installation](https://docs.astral.sh/uv/getting-started/installation/).
2. Install packages and a virtual environment

    ```powershell
    uv sync
    ```

3. Write a config file; see examples in `sparse-rebasin-progressive-growth/configs`
4. Under the `sparse-rebasin-progressive-growth` folder, run

    ```powershell
    uv run experiment.py -config <path-to-config> -seed 0 -pretrain True
    ```

5. After running 4. once, run for the progressive version (our method)

    ```powershell
    uv run experiment.py -config <path-to-config> -seed 0 -pretrain False --rewind 0 --method progressive
    ```

    Or for baselines

    ```powershell
    uv run experiment.py -config <path-to-config> -seed 0 -pretrain False --rewind 0 --method baselines
    ```

6. If you didn't modify the network architecture and dataset, you may run 4. just once and run 5. directly using a new config.
7. The result is in the `res` folder.

## Usage

To run experiments for ResNet20 and VGG11 on CIFAR-10/100, or ResNet20 on MNIST, this process is involved in two parts. First ensure that the desired settings are specified in the relative `.yaml` from `configs` and `configs_cifar100` folders for CIFAR-10 and CIFAR-100 (depending which experiment you are running). MNIST uses [`configs/config_mnist.yaml`](configs/config_mnist.yaml).

To run the experiment after setup:

Part 1: Dense training, Pruning, and Permutation Matching.
```
uv run experiment.py -config <FILE_NAME> -seed <INT> -pretrain True
```
Part 2: Sparse training LTH, Naive, and the Permuted solutions.
```
uv run experiment.py -config <FILE_NAME> -seed <INT> -pretrain False --rewind <INT>
```

For example, to pretrain the MNIST ResNet20:

```
uv run experiment.py -config configs/config_mnist.yaml -seed 0 -pretrain True
```

On Windows, the data loaders automatically use `num_workers=0`. This avoids
PyTorch shared-memory allocation failures such as `error code: 1455` when
multiple worker processes collate batches. On Linux, the `num_workers` value
from the selected configuration is used.

### LeNet

LeNet is supported for CIFAR-10, CIFAR-100, and MNIST. Set
`model_type: LeNet` in a configuration, or use
`configs/config_lenet_1_80.yaml`:

```powershell
uv run experiment.py -config configs/config_lenet_1_80.yaml -seed 0 -pretrain True
uv run experiment.py -config configs/config_lenet_1_80.yaml -seed 0 -pretrain False --rewind 0
```

The LeNet configuration uses the same dense-training, iterative-pruning,
activation-matching, and sparse-training pipeline as the VGG11 path.

### Progressive mask growth

The `progressive` method transfers Model A's final lottery-ticket mask to a
freshly initialized Model B, grows the active mask through explicit density
targets, trains after each growth stage, and finally runs the existing
iterative pruning process. Growth targets are global densities over eligible
convolution and linear weights; they must be monotonically increasing.

Add a `progressive` section to the selected configuration:

```yaml
progressive:
  target_densities: [0.20, 0.35, 0.50]
  epochs_per_stage: 5
  selection: magnitude  # magnitude, random, or gradient
  growth_seed: 0
```

After the dense/pruning phase has produced Model A's mask, run:

```powershell
uv run experiment.py -config configs/config_1_80.yaml -seed 0 -pretrain False --method progressive
```

`magnitude` activates the currently inactive Model B connections with the
largest original dense-initialization weights, preserving those weights while
they are masked. `random` is available as an ablation. `gradient` activates
connections with the largest absolute weight gradients from one training batch
before each growth stage. Each stage writes a `progressive_stage_*` state-dict
checkpoint and logs density and active-weight counts to W&B. The final pruning target remains
`pruning.sparsity`, so a progressive run can be compared with the existing
LTH, naive, permuted, and dense Model B controls at matched final sparsity.
