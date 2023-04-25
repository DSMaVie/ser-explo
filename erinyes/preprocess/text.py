import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


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


class TokenizeText:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="phonemize")

        text_keyword = "transcript" if "transcript" in data.columns else "Statement"

        data["phonemes"] = data[text_keyword].progress_apply(
            lambda s: self.tokenizer.phonemize(s)
        )
        return data


class UpdateVocab:
    def __init__(self, tokenizer: PreTrainedTokenizer, tokenizer_location:Path) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_location = tokenizer_location

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        all_tokens = np.fromiter(
            (item for sublist in data.phonemes.values for item in sublist), str
        )
        counter = Counter(all_tokens)

        unique_tokens = set(counter.keys()).union(self.tokenizer.all_special_tokens)
        new_vocab = {token: number for number,token in enumerate(unique_tokens)}

        with (self.tokenizer_location / "vocab.json").open("w") as file:
            json.dump(new_vocab, file, indent=2)
