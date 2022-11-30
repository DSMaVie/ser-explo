

from enum import Enum, auto


class Dataset(Enum):
    RAV = auto()
    MOS = auto()
    SWBD = auto()
    IEM = auto()


class LabelCategory(Enum):
    EMOTION = auto()
    SENTIMENT = auto()