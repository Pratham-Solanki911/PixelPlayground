import numpy as np
import torch

from alifelab.agents.architectures import ARCHITECTURE_NAMES, build_architecture

shapes = {
    "rank1": (64,),
    "rank2": (6, 20),
    "rank3": (10, 12, 8),
    "rank4": (3, 10, 12, 8),
}

for rank, shape in shapes.items():
    for name in ARCHITECTURE_NAMES:
        arch = build_architecture(name, obs_shape=shape, feature_dim=64)
        x = torch.randn(2, *shape)
        with torch.no_grad():
            y = arch(x)
        assert y.shape == (2, 64), f"{name}:{shape} -> {y.shape}"
        print(f"ok {name:16s} {rank:5s} {tuple(shape)} -> {tuple(y.shape)}")

arch = build_architecture("cnn_ssm", (10, 12, 8))
print("cnn_ssm params", sum(p.numel() for p in arch.parameters()))