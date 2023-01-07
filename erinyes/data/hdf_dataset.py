import logging
import os

import h5py
import torch
from torch.utils.data import Dataset

from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class Hdf5Dataset(Dataset):
    def __init__(self, src_path: os.PathLike, split: Split) -> None:
        super().__init__()

        self.src_path = src_path
        self.split = split

    def __len__(self):
        with h5py.File(self.src_path, "r") as file:
            return len(file[self.split.name.lower()].keys())

    def __getitem__(self, idx: str) -> torch.TensorType:
        with h5py.File(self.src_path, "r") as file:

            node = file[f"{self.split.name.lower()}/{idx}"]
            labels = torch.Tensor(node.attrs["label"])
            features = torch.Tensor(node[()])
            
            return features, labels

    def get_indices(self):
        keys = []
        with h5py.File(self.src_path, "r") as file:
            logger.info(
                f"opening file at {self.src_path}. available first lvl keys {list(file.keys())}"
            )
            train_block = file[self.split.name.lower()]
            train_block.visit(
                lambda key: keys.append(key)
                if isinstance(train_block[key], h5py.Dataset)
                else None
            )
        return keys
