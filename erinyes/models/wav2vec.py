import logging
import os
import shutil
from functools import partial
from turtle import forward

import torch
from torch import nn
from transformers import AutoProcessor, Wav2Vec2Model

from erinyes.util.env import Env

_MODEL_LOC = Env.load().MODEL_DIR / "wav2vec"


class Wav2Vec(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        if not _MODEL_LOC.exists():
            logging.error("Please load the model from the net first!")

        self.processor = Wav2Vec2Model.from_pretrained(_MODEL_LOC)
        self.model = AutoProcessor.from_pretrained(_MODEL_LOC)
        self.clf_head = partial(torch.argmax, dim=-1)

    @staticmethod
    def load_from_web(overwrite: bool = False):
        new_created = False

        if not (_MODEL_LOC).exists():
            os.makedirs(_MODEL_LOC)
            new_created = True

        if overwrite or new_created:
            for pth in _MODEL_LOC.iterdir():
                shutil.rmtree(pth)

            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            model.save_pretrained(_MODEL_LOC)
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            processor.save_pretrained(_MODEL_LOC)

    def forward(self, x):
        x = self.processor(x)
        x = self.model(x)
        return self.clf_head(x)
