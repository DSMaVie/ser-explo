from __future__ import annotations

import logging

import torch
from torch import nn, softmax
from torch.nn.functional import cross_entropy
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class HFWav2Vec2withClf(Wav2Vec2Model):
    def __init__(
        self,
        config: Wav2Vec2Config,
        clf_hidden_dim: int,
        clf_out_dim: int,
        freeze_encoder: bool = False,
        use_conv_features: bool = False,
    ) -> None:
        super().__init__(config)

        self.classifier = nn.Sequential(
            nn.Linear(out_features=clf_hidden_dim, in_features=clf_hidden_dim),
            nn.Tanh(),
            nn.Linear(out_features=clf_out_dim, in_features=clf_hidden_dim),
            nn.ReLU()
        )
        self.return_conv_features = use_conv_features

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder

    def forward(self, input_values, labels=None, attention_mask=None):
        # logger.info(
        #     f"received input_values of value {input_values} and shape {input_values.shape}"
        # )
        # raise NotImplementedError
        encoder_out = super().forward(
            input_values=input_values, attention_mask=attention_mask
        )
        hidden_states = (
            encoder_out.extract_features
            if self.return_conv_features
            else encoder_out.last_hidden_state
        )

        mpooled_hs = torch.mean(hidden_states, dim=1)
        logits = self.classifier(mpooled_hs)

        loss = None
        if labels is not None:
            loss = cross_entropy(logits, labels.long())

        return SequenceClassifierOutput(logits=logits, loss=loss)

    def __repr__(self):
        return self.__class__.__name__
