from __future__ import annotations

import logging
import re
import string
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2PhonemeCTCTokenizer

from erinyes.data.labels import EmoEnrichedPhonemeEncodec

logger = logging.getLogger(__name__)


class NormalizeText:
    def __init__(self) -> None:
        super().__init__()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        text_keyword = "transcript" if "transcript" in data.columns else "Statement"
        tr_table = str.maketrans(
            "", "", string.punctuation.replace("'", "")
        )  # all punct but not '

        tqdm.pandas(desc="removing comments in brackets")
        data[text_keyword] = data[text_keyword].progress_apply(
            lambda s: re.sub(r"([\[\(].*[\)\]])", "", s),
        )

        tqdm.pandas(desc="stripping punctuation")
        data[text_keyword] = data[text_keyword].progress_apply(
            lambda s: s.translate(tr_table),
        )
        return data


class PhonemizeText:
    def __init__(self, tokenizer_location: Path) -> None:
        super().__init__()
        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(tokenizer_location)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="phonemize")

        text_keyword = "transcript" if "transcript" in data.columns else "Statement"

        data["phonemes"] = data[text_keyword].progress_apply(
            lambda s: self.tokenizer.phonemize(s).rstrip(" |").replace("|", "<NS>")
        )
        # data["phonemes"] = data["phonemes"].progress_apply(
        #     lambda s: f"<s> {s[:-1]}</s>"
        # )  # add beginning and end tokens
        return data


class UpdateVocab:
    def __init__(
        self,
        tokenizer_location: Path,
        new_model_location: Path,
        classes: list[str],
    ) -> None:
        super().__init__()
        self.tokenizer_location = tokenizer_location
        self.new_model_location = new_model_location
        self.classes = classes

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        label_encodec = EmoEnrichedPhonemeEncodec(self.tokenizer_location, self.classes)
        label_encodec.shrink_vocabulary(data, self.new_model_location)
        return data


class EnsureMinTokens:
    def __init__(self, column: str, min_number: int) -> None:
        self.column = column
        self.min_number = min_number

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        to_keep = data[self.column].apply(lambda s: len(s.split()) > self.min_number)

        logger.info(
            f"dropping {100 - len(to_keep)/len(data)*100:.1f} percent of instances"
        )
        # breakpoint()

        return data.loc[to_keep.index]
