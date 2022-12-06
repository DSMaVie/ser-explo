from __future__ import annotations

import numpy as np


class LabelEncodec:
    def __init__(self, classes: list[str], NofN: bool = False) -> None:
        self.NofN = NofN
        self.classes = classes

    def __ints2mhe(self, n_list: list[int]) -> np.ndarray:
        mhe_class = np.zeros(len(self.classes), dtype=np.int8)
        mhe_class[n_list] = 1
        return mhe_class

    def __str2int(self, s: str) -> int:
        return self.classes.index(s)

    def __int2str(self, n: int) -> str:
        return self.classes[n]

    def __mhe2ints(self, mhe_label: list[int]) -> int:
        return np.where(mhe_label == 1)

    def encode(self, label: str) -> np.ndarray:
        if self.NofN:
            n_list = [self.__str2int(lab) for lab in label.split(".")]
            return self.__ints2mhe(n_list)
        else:
            return np.array([self.__str2int(label)])

    def decode(self, label: int | list[int]) -> str:
        if self.NofN and not isinstance(label, list):
            raise TypeError("Expected list of values to decode!")
        elif not self.NofN and isinstance(label):
            raise TypeError("Did not expect label to be of type list!")

        if self.NofN:
            n_list = self.__mhe2ints(label)
            str_list = [self.__int2str(n) for n in n_list]
            return ".".join(str_list)

        return self.__int2str(label)
