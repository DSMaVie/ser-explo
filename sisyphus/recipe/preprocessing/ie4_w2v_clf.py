from __future__ import annotations

import itertools
import json
from pathlib import Path

from transformers import Wav2Vec2Model, Wav2Vec2PhonemeCTCTokenizer

from erinyes.data.features import NormalizedRawAudio, Wav2Vec2OutputFeatureExtractor
from erinyes.data.labels import EmoEnrichedPhonemeEncodec, IntEncodec
from erinyes.preprocess.processor import Preprocessor, PreproRecipe
from erinyes.preprocess.steps import (
    ConditionalSplitter,
    EmotionFilterNFold,
    ExcludeExamples,
    LabelNormalizer,
    MinmaxDrop,
    TransformStartStopToDurations,
    ValFromTrainSplitter,
)
from erinyes.preprocess.text import (
    EnsureMinTokens,
    NormalizeText,
    PhonemizeText,
    UpdateVocab,
)
from erinyes.util.enums import Dataset
from sisyphus import tk

from ..preprocessing.base import PreprocessingJob

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
                "integer_encoding",
                IntEncodec,
                args={"classes": EMOTIONS},
            ),
        )

    def preset(self):
        self.utterance_idx.set("file_idx")
        self.label_column.set("Emotion")


class IEM4ProcessorForWav2Vec2WithPhonemes(PreprocessingJob):
    def __init__(self, path_to_tokenizer: tk.Path) -> None:
        super().__init__()

        self.path_to_tokenizer = Path(path_to_tokenizer)

        self.new_model_loc = self.output_path("model", directory=True)
        self.processor = Preprocessor(
            src=Dataset.IEM,
            name="iem4_w2v_clf_with_text",
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
                PreproRecipe("normalize_text", NormalizeText),
                PreproRecipe(
                    "tokenize_text",
                    PhonemizeText,
                    args={"tokenizer_location": Path(self.path_to_tokenizer)},
                ),
                PreproRecipe(
                    "ensure adequate phone number",
                    EnsureMinTokens,
                    args={"column": "phonemes", "min_number": 6},
                ),
                PreproRecipe(
                    "cut troublemakers",
                    ExcludeExamples,
                    args={
                        "exclusion_list": [
                            "Ses01F_script01_3_F012",
                            "Ses01F_script03_1_F020",
                            "Ses01F_script03_1_M036",
                        ],
                        "column": "file_idx",
                    },
                ),
                PreproRecipe(
                    "update_vocab",
                    UpdateVocab,
                    args={
                        "tokenizer_location": Path(self.path_to_tokenizer),
                        "classes": EMOTIONS,
                        "new_model_location": Path(self.new_model_loc),
                    },
                ),
            ],
            feature_extractor=PreproRecipe(
                "raw_extractor", NormalizedRawAudio, args={"resample_to": 16_000}
            ),
            label_encodec=PreproRecipe(
                "emp_enriched_phoneme_encoding",
                EmoEnrichedPhonemeEncodec,
                args={
                    "classes": EMOTIONS,
                    "tokenizer_location": Path(self.new_model_loc),
                },
            ),
        )

    def preset(self):
        self.utterance_idx.set("file_idx")
        self.label_column.set(("Emotion", "phonemes"))

        delayed_args = dict()
        return delayed_args


class IEM4ProcessorForWav2Vec2WithModelFeatures(PreprocessingJob):
    def __init__(self, path_to_tokenizer: tk.Path, rqmts: dict | None = None) -> None:
        super().__init__(rqmts=rqmts)

        self.path_to_tokenizer = Path(path_to_tokenizer)

        self.processor = Preprocessor(
            src=Dataset.IEM,
            name="iem4_w2v_clf_with_text",
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
                "model_output_extractor",
                Wav2Vec2OutputFeatureExtractor,
                args={
                    "resample_to": 16_000,
                    "device": "cuda" if rqmts.get("gpu", False) else "cpu",
                },
                delayed_args=["model"],
            ),
            label_encodec=PreproRecipe(
                "integer_encoding",
                IntEncodec,
                args={"classes": EMOTIONS},
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
            # if step.delayed_args is not None and "tokenizer" in step.delayed_args:
            #     delayed_args.update({f"{step.name}:tokenizer": tok})
            if step.delayed_args is not None and "model" in step.delayed_args:
                delayed_args.update({f"{step.name}:model": model})

        return delayed_args
