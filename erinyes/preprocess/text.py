import json
import re
import shutil
import string
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, Wav2Vec2PhonemeCTCTokenizer


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
            lambda s: self.tokenizer.phonemize(s)
        )
        data["phonemes"] = data["phonemes"].progress_apply(
            lambda s: f"<s> {s[:-1]}</s>"
        )  # add beginning and end tokens
        return data


class UpdateVocab:
    def __init__(
        self,
        # tokenizer: PreTrainedTokenizer,
        tokenizer_location: Path,
        new_model_location: Path,
        label_col: str,
    ) -> None:
        super().__init__()
        # self.tokenizer = toke#nizer
        self.tokenizer_location = tokenizer_location
        self.new_model_location = new_model_location
        self.label_col = label_col

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(self.tokenizer_location)
        tokenizer._additional_special_tokens = ["|"]
        tokenizer.save_pretrained(self.new_model_location)

        unique_tokens = set(" ".join(data.phonemes.values).split(" ")).union(
            set(tokenizer.all_special_tokens)
        )
        unique_tokens = sorted(
            unique_tokens, key=lambda s: s in tokenizer.all_special_tokens, reverse=True
        )
        new_vocab = {token: number for number, token in enumerate(unique_tokens)}
        n_emo = len(data[self.label_col].unique())

        for file in self.tokenizer_location.iterdir():
            if file.name == "vocab.json":
                with (self.new_model_location / file.name).open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(new_vocab, f, indent=2)

            elif file.name == "tokenizer_config.json":
                with file.open("r") as f:
                    conf = json.load(f)
                    conf[
                        "special_tokens_map_file"
                    ] = "../facebook_wav2vec2-base-960h/special_tokens_map.json"
                    with (self.new_model_location / file.name).open("w") as f_out:
                        json.dump(conf, f_out)

            elif file.name == "config.json":
                conf = AutoConfig.from_pretrained(file)
                conf.vocab_size = (len(new_vocab) - len(tokenizer.all_special_tokens)) * n_emo + len(tokenizer.all_special_tokens)
                # breakpoint()
                conf.save_pretrained(self.new_model_location)

            elif file.name == "special_tokens_map.json":
                continue
                #skip bc already serialized

            else:
                shutil.copy(file, self.new_model_location)
        return data
