from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Generic, TypeVar

import transformers

LabelType = TypeVar("LabelType")
EncLabelType = TypeVar("EncLabelType")


class LabelEncodec(ABC, Generic[LabelType, EncLabelType]):
    @abstractmethod
    def encode(self, label: LabelType) -> EncLabelType:
        ...

    @abstractmethod
    def decode(self, label: EncLabelType) -> LabelType:
        ...

    @abstractproperty
    def class_dim(self) -> int:
        ...

    @abstractproperty
    def is_mhe(self) -> bool:
        ...

    def batch_encode(self, label_list: list[LabelType]) -> list[EncLabelType]:
        return [self.encode(label) for label in label_list]

    def batch_decode(self, enc_label_list: list[EncLabelType]) -> list[LabelType]:
        return [self.decode(enc_label) for enc_label in enc_label_list]


class IntEncodec(LabelEncodec[str, int]):
    def __init__(self, classes: list[str]) -> None:
        self.classes = classes

    def encode(self, label: str) -> int:
        return self.classes.index(label)

    def decode(self, enc_label: int) -> str:
        return self.classes[enc_label]

    @property
    def class_dim(self) -> int:
        return len(self.classes)

    @property
    def is_mhe(self) -> bool:
        return False


class SeqIntEncodec(LabelEncodec["list[str, str]", list]):
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizer, classes: list[str]
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.classes = classes

    def encode(self, label: list[str, str]) -> list[str]:
        phoneme_seq, emotion = label
        emo_index = self.classes.index(emotion)
        # todo exceptions! like unknown tokens etc
        return [phon + "_" + emo_index for phon in phoneme_seq]

    def decode(self, label: list[str]) -> list[str, str]:
        return [l.split("_") for l in label]


# class LabelEncodec:
#     def __init__(self, classes: list[str], NofN: bool = False) -> None:
#         self.NofN = NofN
#         self.classes = classes

#     def __ints2mhe(self, n_list: list[int]) -> np.ndarray:
#         mhe_class = np.zeros(len(self.classes), dtype=np.int8)
#         mhe_class[n_list] = 1
#         return mhe_class

#     def __str2int(self, s: str) -> int:
#         return self.classes.index(s)

#     def __int2str(self, n: int) -> str:
#         return self.classes[n]

#     def __mhe2ints(self, mhe_label: list[int]) -> int:
#         return np.where(mhe_label == 1)

#     def encode(self, label: str) -> np.ndarray:
#         if self.NofN:
#             n_list = [self.__str2int(lab) for lab in label.split(".")]
#             return self.__ints2mhe(n_list)
#         else:
#             return np.array([self.__str2int(label)])

#     def decode(self, label: int | list[int]) -> str:
#         if self.NofN and not isinstance(label, list):
#             raise TypeError("Expected list of values to decode!")
#         elif not self.NofN and isinstance(label):
#             raise TypeError("Did not expect label to be of type list!")

#         if self.NofN:
#             n_list = self.__mhe2ints(label)
#             str_list = [self.__int2str(n) for n in n_list]
#             return ".".join(str_list)

#         return self.__int2str(label)

#     def get_class_dim(self):
#         return len(self.classes)

#     def get_is_mhe(self):
#         return self.NofN
