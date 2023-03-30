from __future__ import annotations

import logging

from torch import nn, softmax
import torch
from torch.nn.functional import cross_entropy
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class HFWav2VecCTCwithClf(nn.Module):
    def __init__(
        self,
        model_loc: str,
        clf_hidden_dim: int,
        clf_out_dim: int,
        freeze_encoder: bool = False,
        use_conv_features: bool = False,
    ) -> None:
        super().__init__()

        full_model = Wav2Vec2ForCTC.from_pretrained(model_loc)
        self.encoder = (
            full_model.wav2vec2
            if not use_conv_features
            else full_model.wav2vec2.feature_extractor
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_features=clf_hidden_dim, in_features=clf_hidden_dim),
            nn.Linear(out_features=clf_out_dim, in_features=clf_hidden_dim),
        )
        self.return_conv_features = use_conv_features

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder

    def forward(self, input_values, labels=None):
        w2v_out = self.encoder(input_values)

        if self.return_conv_features:
            # logger.info(f"got encoder output of shape {w2v_out.shape}")
            w2v_out = w2v_out.transpose(1, 2)
            # feature and seq dim swapped for some reason. hf implementation does this here as well
        else:
            # logger.info(f"got encoder out of sape {w2v_out.last_hidden_state.shape}")
            w2v_out = w2v_out.last_hidden_state

        w2v_out = torch.mean(w2v_out, dim=1)
        logits = self.classifier(w2v_out)

        loss = None
        if labels is not None:
            pred = softmax(logits, dim=1)
            loss = cross_entropy(pred, labels.long())

        return SequenceClassifierOutput(hidden_states=w2v_out, logits=logits, loss=loss)

    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing_enable()

    def __repr__(self):
        return self.__class__.__name__


### DEPRECTATED ###
class Wav2Vec(nn.Module):
    def __init__(
        self,
        model_loc: str,
        frozen: bool = False,
        classifier: nn.Module | None = None,
        return_conv_features: bool = False,
    ) -> None:
        raise NotImplementedError
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
