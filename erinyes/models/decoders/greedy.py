import torch
from torch import nn


class W2V_PooledDecoder(nn.Module):
    def __init__(self, class_dim: int) -> None:
        self.projector = nn.LazyLinear(class_dim)

    def forward(self, x):
        x = torch.mean(x, dim=1)  # first dim is seq dim
        return self.projector(x)
