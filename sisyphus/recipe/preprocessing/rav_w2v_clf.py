from pathlib import Path

import pandas as pd
from recipe.preprocessing.base import PreprocessingJob

from erinyes.data.features import NormalizedRawAudio
from erinyes.data.labels import IntEncodec
from erinyes.preprocess.processor import Preprocessor, PreproRecipe
from erinyes.preprocess.steps import (
    ConditionalSplitter,
    EmotionFilterNFold,
    GatherDurations,
    LabelNormalizer,
)
from erinyes.util.enums import Dataset

EMOTIONS = ["Happiness", "Anger", "Sadness", "Fear", "Disgust", "Surprise"]


class RavdessW2VPreproJob(PreprocessingJob):
    def __init__(self) -> None:
        super().__init__()

        self.processor = Preprocessor(
            src=Dataset.RAV,
            name="ravdess_w2v_clf",
            steps=[
                PreproRecipe(
                    "normalize_labels",
                    LabelNormalizer,
                ),
                PreproRecipe(
                    "filter_emotions", EmotionFilterNFold, args={"keep": EMOTIONS}
                ),
                PreproRecipe(
                    "split_on_speaker",
                    ConditionalSplitter,
                    args={
                        "src_col": "Actor",
                        "train": list(range(0, 19)),
                        "val": 19,
                        "test": list(range(19, 25)),
                    },
                ),
                PreproRecipe(
                    "gather_durations",
                    GatherDurations,
                    args={"pth": "rav", "filetype": "wav"},
                ),
            ],
            feature_extractor=PreproRecipe(
                "raw_extractor", NormalizedRawAudio, args={"resample_to": 16_000}
            ),
            label_encodec=PreproRecipe(
                "integer_encoding", IntEncodec, args={"classes": EMOTIONS}
            ),
        )

    def preset(self):
        self.utterance_idx.set("file_idx")
        self.label_column.set("Emotion")
