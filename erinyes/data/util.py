from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, SubsetRandomSampler

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.util.enums import Split


def collate_with_pack_pad_to_batch(
    data: list[tuple[torch.TensorType, torch.TensorType]], pack: bool = True
) -> tuple[torch.TensorType, torch.TensorType]:
    if pack:
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

    signals, labels = zip(*data)
    seqs = pad_sequence(signals, batch_first=True)

    if pack:
        orig_lengths = torch.Tensor([signal.shape[0] for signal in signals])
        seqs = pack_padded_sequence(seqs, lengths=orig_lengths, batch_first=True)

    labels = torch.stack(labels).squeeze(dim=1)
    return seqs, labels


def get_data_loader(
    data_path: Path,
    batch_size: int,
    split: Split,
    num_workers: int = 0,
    gpu_available: bool = False,
    pack: bool = True,
):
    dataset = Hdf5Dataset(data_path / "processed_data.h5", split=split, load_data=True)
    sampler = SubsetRandomSampler(dataset.get_indices())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=partial(collate_with_pack_pad_to_batch, pack=pack),
        num_workers=num_workers,
        pin_memory=gpu_available,
    )
