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

    @abstractmethod
    def get_feature_dim(self):
        """retrieve the number of features per time step from the extractor."""

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
            raise RuntimeError(
                f"Extraction failed. file={pth_to_data}, start={start}, dur={duration}"
            )


class LogMelSpec(FeatureExtractor):
    def __apply__(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """output shape is TxF"""
        return librosa.feature.melspectrogram(y=signal, sr=sr, **self.additional_args).T

    def get_feature_dim(self):
        return self.additional_args["n_mels"]


class NormalizedRawAudio(FeatureExtractor):
    def __apply__(self, signal: np.ndarray, sr: int) -> np.ndarray:
        mu = np.mean(signal)
        sig = np.std(signal)
        return (signal - mu) / sig

    def get_feature_dim(self):
        return 1


class FeatureExtractors(Enum):
    logmelspec = LogMelSpec
    normedraw = NormalizedRawAudio
