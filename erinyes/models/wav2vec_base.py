from __future__ import annotations

import logging
import os

from torch import nn
from transformers import AutoProcessor, Wav2Vec2Model

from erinyes.util.env import Env

_PRETRAINED_MODEL_LOC = Env.load().MODEL_DIR / "wav2vec"


class Wav2Vec(nn.Module):
    def __init__(
        self, frozen: bool = False, classifier: nn.Module | None = None
    ) -> None:
        super().__init__()

        if not _PRETRAINED_MODEL_LOC.exists():
            logging.error("Please load the model from the net first!")

        self.processor = AutoProcessor.from_pretrained(_PRETRAINED_MODEL_LOC)
        self.encoder = Wav2Vec2Model.from_pretrained(_PRETRAINED_MODEL_LOC)
        self.classifier = classifier

        for param in self.encoder.parameters:
            param.requires_grad = not frozen

    @staticmethod
    def load_from_web(overwrite: bool = False):
        new_created = False

        if not _PRETRAINED_MODEL_LOC.exists():
            os.makedirs(_PRETRAINED_MODEL_LOC)
            new_created = True

        if overwrite or new_created:
            for pth in _PRETRAINED_MODEL_LOC.iterdir():
                os.remove(pth)

            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            model.save_pretrained(_PRETRAINED_MODEL_LOC)
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            processor.save_pretrained(_PRETRAINED_MODEL_LOC)

    def forward(self, x):
        x = self.processor(x, sampling_rate=16e3, return_tensors="pt")["input_features"]
        w2v_out = self.model(x).last_hidden_state

        if not self.classifier:
            return w2v_out

        return self.classifier(w2v_out)
