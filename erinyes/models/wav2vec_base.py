from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from torch import nn, softmax
from torch.nn.functional import cross_entropy
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class HFWav2VecCTCwithClf(Wav2Vec2ForCTC):
    def __init__(
        self,
        config: Wav2Vec2Config,
        clf_hidden_dim: int,
        clf_out_dim: int,
        freeze_encoder: bool = False,
        use_conv_features: bool = False,
    ) -> None:
        super().__init__(config)

        self.encoder = (
            self.wav2vec2 if not use_conv_features else self.wav2vec2.feature_extractor
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_features=clf_hidden_dim, in_features=clf_hidden_dim),
            nn.Linear(out_features=clf_out_dim, in_features=clf_hidden_dim),
        )
        self.return_conv_features = use_conv_features

        self._modules = OrderedDict(
            {"encoder": self.encoder, "classifier": self.classifier}
        )

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder

    def forward(self, input_values, labels=None, attention_mask=None):
        if self.return_conv_features:
            w2v_out = self.encoder(input_values)
            # logger.info(f"got encoder output of shape {w2v_out.shape}")
            hidden_states = w2v_out.transpose(1, 2)
            # feature and seq dim swapped for some reason. hf implementation does this here as well
        else:
            w2v_out = self.encoder(input_values, attention_mask)
            # logger.info(f"got encoder out of sape {w2v_out.last_hidden_state.shape}")
            hidden_states = w2v_out.last_hidden_state

        mpooled_hs = torch.mean(hidden_states, dim=1)
        logits = self.classifier(mpooled_hs)

        loss = None
        if labels is not None:
            pred = softmax(logits, dim=1)
            loss = cross_entropy(pred, labels.long())

        return SequenceClassifierOutput(
            hidden_states=hidden_states, logits=logits, loss=loss
        )

    def __repr__(self):
        return self.__class__.__name__
