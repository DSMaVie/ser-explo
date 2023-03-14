from __future__ import annotations

import torch
from torch import nn
from transformers import AutoProcessor, Wav2Vec2Model


class Wav2Vec(nn.Module):
    def __init__(
        self,
        model_loc: str,
        frozen: bool = False,
        classifier: nn.Module | None = None,
        return_conv_features: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(model_loc)
        self.classifier = classifier
        self.return_conv_features = return_conv_features

        for param in self.encoder.parameters():
            param.requires_grad = not frozen

    def forward(self, x):
        w2v_out = self.encoder(x)

        if self.return_conv_features:
            x = w2v_out.extract_features
        else:
            x = w2v_out.last_hidden_state

        if not self.classifier:
            return x

        return self.classifier(x)
