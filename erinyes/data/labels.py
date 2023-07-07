from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd
import transformers

LabelType = TypeVar("LabelType")
EncLabelType = TypeVar("EncLabelType")


class LabelEncodec(ABC, Generic[LabelType, EncLabelType]):
    @abstractmethod
    def encode(self, label: LabelType) -> EncLabelType:
        ...

    @abstractmethod
    def decode(self, label: EncLabelType) -> LabelType:
        ...

    @abstractproperty
    def class_dim(self) -> int:
        ...

    @abstractproperty
    def is_mhe(self) -> bool:
        ...

    def batch_encode(self, label_list: list[LabelType]) -> list[EncLabelType]:
        return [self.encode(label) for label in label_list]

    def batch_decode(self, enc_label_list: list[EncLabelType]) -> list[LabelType]:
        return [self.decode(enc_label) for enc_label in enc_label_list]


class IntEncodec(LabelEncodec[str, int]):
    def __init__(self, classes: list[str]) -> None:
        self.classes = classes

    def encode(self, label: str) -> int:
        return self.classes.index(label)

    def decode(self, enc_label: int) -> str:
        return self.classes[enc_label]

    @property
    def class_dim(self) -> int:
        return len(self.classes)

    @property
    def is_mhe(self) -> bool:
        return False


class EmoEnrichedPhonemeEncodec(LabelEncodec):
    def __init__(self, tokenizer_location: Path, classes: list[str]) -> None:
        super().__init__()

        self.tokenizer_location = tokenizer_location
        self.tokenizer = transformers.Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            tokenizer_location#, pad_token="<blank>"
        )
        self.classes = classes

    def shrink_vocabulary(self, data: pd.DataFrame, new_location: Path):
        self.tokenizer._additional_special_tokens = ["<NS>"]
        self.tokenizer.save_pretrained(new_location)

        unique_tokens = set(" ".join(data.phonemes.values).split(" ")).union(
            set(self.tokenizer.all_special_tokens)
        )
        unique_tokens = sorted(
            unique_tokens,
            key=lambda s: s in self.tokenizer.all_special_tokens,
            reverse=True,
        )

        # create new vocab from unqie tokens and emotions
        idx = 0
        new_vocab = {}
        for token in unique_tokens:
            if token in self.tokenizer.all_special_tokens:
                new_vocab.update({token: idx})
                idx += 1
            else:
                for emo_idx in range(len(self.classes)):
                    new_vocab.update({f"{token}_{emo_idx}": idx})
                    idx += 1

        # fix serialized files
        for file in self.tokenizer_location.iterdir():
            if file.name == "vocab.json":
                with (new_location / file.name).open("w", encoding="utf-8") as f:
                    json.dump(new_vocab, f, indent=2)

            elif file.name == "tokenizer_config.json":
                with file.open("r") as f:
                    conf = json.load(f)
                    conf[
                        "special_tokens_map_file"
                    ] = "../facebook_wav2vec2-base-960h/special_tokens_map.json"
                    with (new_location / file.name).open("w") as f_out:
                        json.dump(conf, f_out)

            elif file.name == "config.json":
                conf = transformers.AutoConfig.from_pretrained(file)
                conf.vocab_size = len(new_vocab)

                (
                    conf.bos_token_id,
                    conf.eos_token_id,
                    conf.pad_token_id,
                ) = self.tokenizer.convert_tokens_to_ids(["<s>", "</s>", "<pad>"])
                # breakpoint()
                conf.save_pretrained(new_location)

            elif file.name == "special_tokens_map.json":
                continue
                # skip bc already serialized

            else:
                shutil.copy(file, new_location)

        self.tokenizer_location = new_location
        self.tokenizer = transformers.Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            new_location
        )

    def encode(self, phonemes: str, Emotion: str) -> list[str]:
        phonemes = phonemes.strip().split()
        emo_idx = self.classes.index(Emotion)

        emo_enriched_phones = [
            f"{phone}_{emo_idx}"
            if phone not in self.tokenizer.all_special_tokens
            else phone
            for phone in phonemes
        ]
        # breakpoint()
        encoded_phones = [
            self.tokenizer.convert_tokens_to_ids(emo_phone)
            for emo_phone in emo_enriched_phones
        ]
        return encoded_phones

    def decode(self, pred: list[int]) -> list[list[str]]:
        decoded_tokens = self.tokenizer.decode(pred).split()
        split_tokens = [token.split("_") for token in decoded_tokens]
        # breakpoint()
        return [
            [split_tok[0], self.classes[int(split_tok[1])]]
            if len(split_tok) > 1
            else [*split_tok, "No Emotion"]
            for split_tok in split_tokens
        ]

    def is_mhe(self) -> bool:
        return False

    def class_dim(self) -> int:
        # make sure vocab is shrunk beforehand
        return self.tokenizer.vocab_size


# class EmoEnrichedPhonemeEncodec(LabelEncodec["list[str, str]", list]):
#     def __init__(self, tokenizer_location: Path, classes: list[str]) -> None:
#         super().__init__()

#         self.tokenizer = transformers.Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
#             tokenizer_location
#         )
#         self.classes = classes

#         self.special_tokens = self.tokenizer.all_special_tokens
#         self.special_token_ids = self.tokenizer.all_special_ids

#     def encode(self, phonemes: str, Emotion: str) -> list[int]:
#         emo_index = self.classes.index(Emotion)

#         phoneme_ids = [
#             self.tokenizer._convert_token_to_id(tok) for tok in phonemes.split(" ")
#         ]

#         emo_enriched_phoneme_ids = [
#             (emo_index + 1) * (tok_id - len(self.special_tokens))
#             + len(self.special_tokens)  # case emotionally relevant
#             if (tok_id not in self.special_token_ids)  # check case
#             else tok_id  # case not emotionally relevant aka special token
#             for tok_id in phoneme_ids
#         ]

#         return emo_enriched_phoneme_ids

#     def decode(self, label: list[int]) -> list[tuple[str, str]]:
#         divider = len(self.tokenizer.get_vocab()) - len(self.special_token_ids)
#         decoupled_ids = [
#             (0, lab - len(self.special_tokens))
#             if lab in self.special_token_ids
#             else divmod(lab - len(self.special_tokens), divider)
#             for lab in label
#         ]

#         decoupled_ids = [
#             (emo_idx, phone_idx + len(self.special_tokens))
#             for emo_idx, phone_idx in decoupled_ids
#         ]
#         decoded_ids = [
#             (self.tokenizer._convert_id_to_token(phone_idx), self.classes[emo_idx])
#             if phone_idx not in self.special_token_ids
#             else (self.tokenizer._convert_id_to_token(phone_idx), "No Emotion")
#             for emo_idx, phone_idx in decoupled_ids
#         ]
#         return decoded_ids

#     def is_mhe(self) -> bool:
#         return False

#     def class_dim(self) -> int:
#         return (
#             self.tokenizer.vocab_size - len(self.tokenizer.all_special_tokens)
#         ) * len(self.classes) + len(self.tokenizer.all_special_tokens)
