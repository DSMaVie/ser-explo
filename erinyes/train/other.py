from enum import Enum

from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam


class Optimizer(Enum):
    Adam = Adam


class LossFn(Enum):
    binary_ce = BCELoss
    mhe_ce = CrossEntropyLoss
