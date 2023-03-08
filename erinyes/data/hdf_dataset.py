import logging
import os

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class Hdf5Dataset(Dataset):
    def __init__(
        self, src_path: os.PathLike, split: Split, load_data: bool = False
    ) -> None:
        super().__init__()

        self.src_path = src_path
        self.split = split
        # self.cache = (
        #     {
        #         idx: self.__getitem__(idx, from_disk=True)
        #         for idx in tqdm(self.get_indices(), desc="loading all data")
        #     }
        #     if load_data
        #     else None
        # )



    def __len__(self):
        with h5py.File(self.src_path, "r") as file:
            return len(file[self.split.name.lower()].keys())

    def __getitem__(self, idx: str, from_disk: bool = False) -> torch.TensorType:
        # logger.info(f"loading from index {idx}")
        # if not from_disk and self.cache:
        #     return self.cache[idx]

        with h5py.File(self.src_path, "r") as file:
            node = file[f"{self.split.name.lower()}/{idx}"]
            labels = torch.Tensor(node.attrs["label"])
            features = torch.Tensor(node[:])

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
