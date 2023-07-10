import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model, modeling_outputs


class PooledSeqClassifier(nn.Module):
    def __init__(self, out_dim: int, is_mhe: bool, hidden_dim_clf: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.LazyLinear(out_features=hidden_dim_clf),
            nn.ReLU(),
            nn.LazyLinear(out_features=out_dim),
            # nn.Sigmoid() if is_mhe else nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = torch.mean(x, dim=1)  # first dim is seq dim
        return self.projector(x)


class HFPooledSeqClassifier(Wav2Vec2Model):
    # subclass is hacky but works. maybe long init time
    def __init__(
        self,
        config: Wav2Vec2Config,
        hidden_dim_clf: int,
        clf_out_dim: int,
        p_dropout_clf: float,
    ) -> None:
        super().__init__(config)
        # breakpoint()
        self.classifier = nn.Sequential(
            nn.Linear(out_features=hidden_dim_clf, in_features=config.conv_dim[-1]),
            nn.Dropout(p=p_dropout_clf),
            nn.ReLU(),
            nn.Linear(out_features=clf_out_dim, in_features=hidden_dim_clf),
        )

    def forward(self, input_values, labels=None):
        logits = self.classifier(input_values)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.long())

        return modeling_outputs.SequenceClassifierOutput(logits=logits, loss=loss)

    def __repr__(self):
        return self.__class__.__name__
