from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SubsetRandomSampler

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


def pad_collate(
    data: list[tuple[torch.TensorType, torch.TensorType]],
    return_attention_mask: bool = True,
    padding_token_id: float = 0.0,
    labels_are_seqs: bool = False,
):
    signals, labels = zip(*data)
    seqs = pad_sequence(signals, batch_first=True, padding_value=padding_token_id)
    # logger.info(f"got batch {seqs} of sequences of shape {seqs.shape}")

    return_dict = {"input_values": seqs}
    if return_attention_mask:
        return_dict["attention_mask"] = ~(seqs == padding_token_id)

    if labels_are_seqs:
        return_dict["labels"] = pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # padding value of -100 is unique for hf ctc models. it wont be used for loss calc
    else:
        return_dict["labels"] = torch.stack(labels).squeeze(dim=1).long()

    return return_dict


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
