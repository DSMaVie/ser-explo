from enum import Enum

import torch
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam


class MultipleBinaryDecLoss:
    def __init__(self) -> None:
        self.loss_fn = BCELoss()  # TODO: try different reductions!!

    def __call__(self, pred: torch.TensorType, true: torch.TensorType):
        return self.loss_fn(pred.float(), true.float())


class MultiClassDecLoss:
    def __init__(self) -> None:
        self.loss_fn = CrossEntropyLoss()

    def __call__(self, pred: torch.TensorType, true: torch.TensorType):
        return self.loss_fn(pred, true.long())


class Optimizer(Enum):
    Adam = Adam


class LossFn(Enum):
    binary_ce = MultipleBinaryDecLoss
    mc_ce = MultiClassDecLoss
