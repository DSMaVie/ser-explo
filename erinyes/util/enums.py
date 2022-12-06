from enum import Enum, auto


class Dataset(Enum):
    RAV = auto()
    MOS_EMOTION = auto()
    MOS_SENTIMENT = auto()
    SWBD = auto()
    IEM = auto()


class LabelCategory(Enum):
    SENTIMENT = auto()
    EMOTION = auto()


class Split(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
