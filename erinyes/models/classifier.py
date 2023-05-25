import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model, modeling_outputs


class PooledSeqClassifier(nn.Module):
    def __init__(self, out_dim: int, is_mhe: bool, hidden_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.LazyLinear(out_features=hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(out_features=out_dim),
            # nn.Sigmoid() if is_mhe else nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = torch.mean(x, dim=1)  # first dim is seq dim
        return self.projector(x)


class HFPooledSeqClassifier(Wav2Vec2Model):
    # subclass is hacky but works. maybe long init time
    def __init__(self, config: Wav2Vec2Config, clf_out_dim: int) -> None:
        super().__init__(config)
        # breakpoint()
        self.classifier = nn.Sequential(
            nn.Linear(out_features=config.hidden_size, in_features=config.conv_dim[-1]),
            nn.Tanh(),
            nn.Linear(out_features=clf_out_dim, in_features=config.hidden_size),
        )

    def forward(self, input_values, labels=None):
        # logger.info(
        #     f"received input_values of value {input_values} and shape {input_values.shape}"
        # )
        # raise NotImplementedError
        # breakpoint()
        logits = self.classifier(input_values)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.long())

        return modeling_outputs.SequenceClassifierOutput(logits=logits, loss=loss)

    def __repr__(self):
        return self.__class__.__name__
