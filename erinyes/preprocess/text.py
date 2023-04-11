import re
import string

import pandas as pd
from tqdm import tqdm_pandas
from transformers.tokenization_utils import PreTrainedTokenizer


class NormalizeText:
    def __init__(self) -> None:
        super().__init__()

    def run(data: pd.DataFrame) -> pd.DataFrame:
        tqdm_pandas()

        text_keyword = "transcript" if "transcript" in data.columns else "Statement"
        tr_table = str.maketrans(
            "", "", string.punctuation.replace("'", "")
        )  # all punct but not '

        data[text_keyword] = data[text_keyword].progress_apply(
            lambda s: re.sub(r"([\[\(].*[\)\]])", "", s),
            desc="removing comments in brackets",
        )
        data[text_keyword] = data[text_keyword].progress_apply(
            lambda s: s.translate(tr_table), desc="stripping punctuation"
        )
        return data


class TokenizeText:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm_pandas()

        text_keyword = "transcript" if "transcript" in data.columns else "Statement"

        data["phonemes"] = data[text_keyword].progress_apply(
            lambda s: self.tokenizer(text=s)
        )
        return data
