from __future__ import annotations

import logging
import os

from torch import nn
from transformers import AutoProcessor, Wav2Vec2Model

from erinyes.util.env import Env


class Wav2Vec(nn.Module):
    def __init__(
        self, model_loc:str, frozen: bool = False, classifier: nn.Module | None = None
    ) -> None:
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_loc)
        self.encoder = Wav2Vec2Model.from_pretrained(model_loc)
        self.classifier = classifier

        for param in self.encoder.parameters:
            param.requires_grad = not frozen

    def forward(self, x):
        x = self.processor(x, sampling_rate=16e3, return_tensors="pt")["input_features"]
        w2v_out = self.model(x).last_hidden_state

        if not self.classifier:
            return w2v_out

        return self.classifier(w2v_out)
