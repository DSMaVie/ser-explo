from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, SubsetRandomSampler

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


def pad_collate(
    data: list[tuple[torch.TensorType, torch.TensorType]]
) -> tuple[torch.TensorType, torch.TensorType]:
    signals, labels = zip(*data)
    seqs = pad_sequence(signals, batch_first=True)

    logger.debug(f"collating labels {labels} and seq {seqs}")
    labels = torch.stack(labels).squeeze(dim=1)
    return seqs, labels


def get_data_loader(
    data_path: Path,
    batch_size: int,
    split: Split,
    num_workers: int = 0,
    gpu_available: bool = False,
):
    dataset = Hdf5Dataset(data_path / "processed_data.h5", split=split)
    sampler = SubsetRandomSampler(dataset.available_indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=pad_collate,
        num_workers=num_workers,
        pin_memory=gpu_available,
    )
