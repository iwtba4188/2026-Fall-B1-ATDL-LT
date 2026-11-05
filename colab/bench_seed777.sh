pwd
ls -l -a
cd /content/2026-Fall-B1-ATDL-LT/sparse-rebasin-progressive-growth

uv run python -c 'import torch; print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))'

# Dense model training
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.3_0.4.yaml -seed 777 -pretrain True --rewind 0

# Baselines
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.3_0.4.yaml -seed 777 -pretrain False --rewind 0 --method baselines

# Progressive growth experiments
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.3_0.4.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.3_0.4_random.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.4_0.6.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.4.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_0.6.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_gradient_1.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_magnitude_0.3_0.4.yaml -seed 777 -pretrain False --rewind 0 --method progressive
uv run experiment.py -config /configs/final/cifar10_vgg11_random_0.3_0.4.yaml -seed 777 -pretrain False --rewind 0 --method progressive
