import torch
from torch import nn


class PooledSeqClassifier(nn.Module):
    def __init__(self, out_dim: int, is_mhe: bool) -> None:
        self.projector = nn.Sequential(
            nn.LazyLinear(out_features=out_dim),
            nn.Sigmoid() if is_mhe else nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = torch.mean(x, dim=1)  # first dim is seq dim
        return self.projector(x)
