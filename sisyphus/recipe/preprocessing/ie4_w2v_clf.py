from matplotlib.transforms import Transform

from erinyes.data.features import NormalizedRawAudio, Raw
from erinyes.data.labels import IntEncodec
from erinyes.preprocess.processor import Preprocessor, PreproTask
from erinyes.preprocess.steps import (ConditionalSplitter, EmotionFilterNFold,
                                      GatherDurations, LabelNormalizer,
                                      TransformStartStopToDurations)
from erinyes.util.enums import Dataset
from sisyphus import Job

EMOTIONS = ["Happiness", "Anger", "Sadness", "Neutral"]


class IEM4ProcessorForWav2Vec2(Job):
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
            ],
            feature_extractor=PreproTask(
                "raw_extractor", NormalizedRawAudio, args={"resample_to": 16_000}
            ),
            label_encodec=PreproTask(
                "integer_encoding", IntEncodec, args={"classes": EMOTIONS}
            ),
        )

    def run(self):
        data = Env.load().RAW_DIR / self.processor.dataset
        self.processor.run_preprocessing()
