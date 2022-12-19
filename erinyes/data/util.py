from __future__ import annotations

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


def collate_with_pack_pad_to_batch(
    data: list[tuple[torch.TensorType, torch.TensorType]]
) -> tuple[torch.TensorType, torch.TensorType]:
    signals, labels = zip(*sorted(data, key=lambda x: len(x[0]), reverse=True))
    orig_lengths = torch.Tensor([signal.shape[0] for signal in signals])

    padded_seqs = pad_sequence(signals, batch_first=True)
    packed_seqs = pack_padded_sequence(
        padded_seqs, lengths=orig_lengths, batch_first=True
    )

    labels = torch.stack(labels).squeeze()
    return packed_seqs, labels
