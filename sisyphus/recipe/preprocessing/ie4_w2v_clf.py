from recipe.preprocessing.base import PreprocessingJob

from erinyes.data.features import NormalizedRawAudio
from erinyes.data.labels import IntEncodec
from erinyes.preprocess.processor import Preprocessor, PreproRecipe
from erinyes.preprocess.steps import (
    ConditionalSplitter,
    EmotionFilterNFold,
    LabelNormalizer,
    MinmaxDrop,
    TransformStartStopToDurations,
    ValFromTrainSplitter,
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
                PreproRecipe(
                    "normalize_labels",
                    LabelNormalizer,
                ),
                PreproRecipe(
                    "filter_emotions",
                    EmotionFilterNFold,
                    args={"keep": EMOTIONS, "fold": {"Excitement": "Happiness"}},
                ),
                PreproRecipe(
                    "split_on_speaker",
                    ConditionalSplitter,
                    args={
                        "src_col": "Session",
                        "train": list(range(0, 5)),
                        "test": 5,
                    },
                ),
                PreproRecipe("produce_val_split", ValFromTrainSplitter),
                PreproRecipe("get_duration_info", TransformStartStopToDurations),
                PreproRecipe(
                    "minmax_duration",
                    MinmaxDrop,
                    args={"column": "duration", "min": 3, "max": 30},
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
