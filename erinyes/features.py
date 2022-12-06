from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import librosa
import numpy as np


class FeatureExtractor(ABC):
    def __init__(self, **additional_args) -> None:
        super().__init__()
        self.additional_args = additional_args

    @abstractmethod
    def __apply__(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """should be overloaded"""

    def extract(
        self, pth_to_data: Path, start: float = 0, duration: float = None
    ) -> np.ndarray:
        try:
            audio, sr = librosa.load(
                pth_to_data, sr=None, offset=start, duration=duration
            )
            return self.__apply__(audio, sr)
        except:
            print("extract:", pth_to_data, start, duration)
            raise RuntimeError(f" Extraction failed. file={pth_to_data}, start={start}, dur={duration}")


class LogMelSpec(FeatureExtractor):
    def __apply__(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return librosa.feature.melspectrogram(y=signal, sr=sr, **self.additional_args)


class FeatureExtractors(Enum):
    logmelspec = LogMelSpec
