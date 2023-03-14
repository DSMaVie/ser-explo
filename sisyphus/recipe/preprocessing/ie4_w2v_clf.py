from pathlib import Path

import pandas as pd
from matplotlib.transforms import Transform
from recipe.preprocessing.base import PreprocessingJob

from erinyes.data.features import NormalizedRawAudio
from erinyes.data.labels import IntEncodec
from erinyes.preprocess.processor import Preprocessor, PreproTask
from erinyes.preprocess.steps import (
    ConditionalSplitter,
    EmotionFilterNFold,
    LabelNormalizer,
    MinmaxDrop,
    TransformStartStopToDurations,
)
from erinyes.util.enums import Dataset

EMOTIONS = ["Happiness", "Anger", "Sadness", "Neutral"]


class IEM4ProcessorForWav2Vec2(PreprocessingJob):
    def __init__(self) -> None:
        super().__init__()

        self.processor = Preprocessor(
            src=Dataset.IEM,
            name="iem4_w2v_clf",
            steps=[
                PreproTask(
                    "normalize_labels",
                    LabelNormalizer,
                ),
                PreproTask(
                    "filter_emotions",
                    EmotionFilterNFold,
                    args={"keep": EMOTIONS, "fold": {"Excitement": "Happiness"}},
                ),
                PreproTask(
                    "split_on_speaker",
                    ConditionalSplitter,
                    args={
                        "src_col": "Session",
                        "train": range(0, 5),
                        "test": 5,
                    },
                ),
                PreproTask("get_duration_info", TransformStartStopToDurations),
                PreproTask(
                    "minmax_duration",
                    MinmaxDrop,
                    args={"column": "duration", "min": 3, "max": 30},
                ),
            ],
            feature_extractor=PreproTask(
                "raw_extractor", NormalizedRawAudio, args={"resample_to": 16_000}
            ),
            label_encodec=PreproTask(
                "integer_encoding", IntEncodec, args={"classes": EMOTIONS}
            ),
        )

    def preset(self):
        self.utterance_idx.set("file_idx")
        self.label_column.set("Emotion")
