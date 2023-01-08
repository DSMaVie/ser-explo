from __future__ import annotations

import torch
from torch import nn


class MultiplcativeSoftAttn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.inp_proj = nn.LazyLinear(in_dim, bias=bias)
        self.mem_proj = nn.LazyLinear(in_dim, bias=bias)
        self.final_proj = nn.LazyLinear(out_dim, bias=bias)

        self.weight_vec = nn.parameter.Parameter(torch.empty(in_dim))
        nn.init.uniform_(self.weight_vec, -1, 1)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        """x has shape (Bx,T,F) with B the batch size, T, the series length, and F the output feature dim (twice hidden size if bidirectional!)"""
        if isinstance(x, nn.utils.rnn.PackedSequence):
            x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]

        m = torch.mean(x, dim=1)
        M = torch.stack([m] * x.shape[1], dim=1)
        R = torch.tanh(self.inp_proj(x)) * torch.tanh(self.mem_proj(M))
        alpha_tilde = R @ self.weight_vec
        alpha = nn.functional.softmax(
            alpha_tilde, dim=1
        )  # this softmax should be len masked for packed tensors!!!!
        return torch.einsum("btf,bt -> bf", x, alpha)


class SoftAttnLstmClf(nn.Module):
    def __init__(
        self,
        input_feature_dim: int,
        lstm_hidden_dim: int,
        class_dim: int,
        soft_attn_hidden_dim: int | None = None,
        n_lstm_layers: int = 1,
        mhe: bool = False,
    ) -> None:
        super().__init__()
        if not soft_attn_hidden_dim:
            soft_attn_hidden_dim = 2 * lstm_hidden_dim

        self.encoder = nn.LSTM(
            input_size=input_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=n_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.attn_module = MultiplcativeSoftAttn(
            in_dim=lstm_hidden_dim * 2,
            out_dim=soft_attn_hidden_dim,
            bias=True,
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(out_features=class_dim),
            nn.Sigmoid() if mhe else nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        encoder_out, _ = self.encoder(x)
        attn_out = self.attn_module(encoder_out)
        return self.classifier(attn_out)
