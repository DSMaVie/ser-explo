from __future__ import annotations

import itertools
from pathlib import Path

from transformers import Wav2Vec2Model, Wav2Vec2PhonemeCTCTokenizer

from erinyes.data.features import NormalizedRawAudio, Wav2Vec2OutputFeatureExtractor
from erinyes.data.labels import IntEncodec, SeqIntEncodec
from erinyes.preprocess.processor import Preprocessor, PreproRecipe
from erinyes.preprocess.steps import (
    ConditionalSplitter,
    EmotionFilterNFold,
    GatherDurations,
    LabelNormalizer,
)
from erinyes.preprocess.text import NormalizeText, PhonemizeText, UpdateVocab
from erinyes.util.enums import Dataset
from sisyphus import tk

from ..preprocessing.base import PreprocessingJob

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


class RavdessW2VPreproJobWithPhonemes(PreprocessingJob):
    def __init__(self, path_to_tokenizer: tk.Path) -> None:
        super().__init__()

        self.path_to_tokenizer = Path(path_to_tokenizer)
        self.new_model_loc = self.output_path("model", directory=True)

        self.processor = Preprocessor(
            src=Dataset.RAV,
            name="ravdess_w2v_clf_with_text",
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
                PreproRecipe("normalize_text", NormalizeText),
                PreproRecipe(
                    "tokenize_text", PhonemizeText, delayed_args=["tokenizer"]
                ),
                PreproRecipe(
                    "update_vocab",
                    UpdateVocab,
                    args={
                        "tokenizer_location": self.path_to_tokenizer,
                        "label_col": "Emotion",
                        "new_model_location": Path(self.new_model_loc),
                    },
                ),
            ],
            feature_extractor=PreproRecipe(
                "raw_extractor", NormalizedRawAudio, args={"resample_to": 16_000}
            ),
            label_encodec=PreproRecipe(
                "emo_enriched_phoneme_encoding",
                SeqIntEncodec,
                args={"classes": EMOTIONS},
                delayed_args=["tokenizer"],
            ),
        )

    def preset(self):
        self.utterance_idx.set("file_idx")
        self.label_column.set(("Emotion", "phonemes"))

        tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            self.new_model_loc,
        )
        delayed_args = dict()
        for step in itertools.chain(
            self.processor.steps,
            [self.processor.feature_extractor, self.processor.label_encodec],
        ):
            if step.delayed_args is not None and "tokenizer" in step.delayed_args:
                delayed_args.update({f"{step.name}:tokenizer": tok})

        return delayed_args


class RavdessW2VPreproJobWithModelFeatures(PreprocessingJob):
    def __init__(self, path_to_tokenizer: tk.Path, rqmts: dict | None = None) -> None:
        super().__init__(rqmts=rqmts)

        self.path_to_tokenizer = Path(path_to_tokenizer)

        self.processor = Preprocessor(
            src=Dataset.RAV,
            name="ravdess_w2v_clf_with_text",
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
                "model_output_extractor",
                Wav2Vec2OutputFeatureExtractor,
                args={"resample_to": 16_000},
                delayed_args=["model"],
            ),
            label_encodec=PreproRecipe(
                "integer_encoding", IntEncodec, args={"classes": EMOTIONS}
            ),
        )

    def preset(self):
        self.utterance_idx.set("file_idx")
        self.label_column.set(("Emotion",))

        # tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(self.path_to_tokenizer)
        model = Wav2Vec2Model.from_pretrained(self.path_to_tokenizer)

        delayed_args = dict()
        for step in itertools.chain(
            self.processor.steps,
            [self.processor.feature_extractor, self.processor.label_encodec],
        ):
            if step.delayed_args is not None and "model" in step.delayed_args:
                delayed_args.update({f"{step.name}:model": model})

        return delayed_args
