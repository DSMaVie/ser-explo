from __future__ import annotations

from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model


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


class Wav2VecCTC(nn.Module):
    def __init__(
        self,
        model_loc: str,
        frozen: bool = False,
        classifier: nn.Module | None = None,
        return_conv_features: bool = False,
    ) -> None:
        super().__init__()

        full_model = Wav2Vec2ForCTC.from_pretrained(model_loc)
        self.encoder = full_model if not return_conv_features else full_model.wav2vec2.feature_extractor

        self.classifier = classifier
        self.return_conv_features = return_conv_features

        for param in self.encoder.parameters():
            param.requires_grad = not frozen

    def forward(self, x):
        w2v_out = self.encoder(x)

        if self.return_conv_features:
            x = w2v_out.transpose(1,2)
            # feature and seq dim swapped for some reason. hf implementation does this here as well
        else:
            x = w2v_out.logits

        if not self.classifier:
            return x

        return self.classifier(x)
