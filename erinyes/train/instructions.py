from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import yaml
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.data.util import collate_with_pack_pad_to_batch
from erinyes.models import Models
from erinyes.train.other import LossFn, Optimizer
from erinyes.train.trainer import Trainer
from erinyes.util.enums import Split
from erinyes.util.env import Env
from erinyes.util.types import yamlDict


@dataclass
class TrainingsInstructions:
    model: nn.Module
    data: DataLoader
    trainer_factory: Callable[[nn.Module, DataLoader, Path], Trainer]

    @classmethod
    def from_yaml(
        cls,
        pth_to_train_instructs: Path,
        pth_to_pp_output: Path,
    ):
        with pth_to_train_instructs.open("r") as train_file:
            train_data = yaml.safe_load(train_file)
        pth_to_arch_params = (
            Env.load().INST_DIR / "architecture" / f"{train_data['model']}.yaml"
        )
        with pth_to_arch_params.open("r") as arch_file:
            arch_data = yaml.safe_load(arch_file)

        with (pth_to_pp_output / "feature_extractor.pkl").open("rb") as file:
            feature_extractor = pickle.load(file)
        with (pth_to_pp_output / "label_encoder.pkl").open("rb") as file:
            label_encodec = pickle.load(file)

        dataloader = cls._get_data_loader(
            pth_to_pp_output,
            batch_size=train_data["batch_size"],
        )

        model = cls._get_model(
            model_name=train_data["model"],
            in_dim=feature_extractor.get_feature_dim(),
            out_dim=label_encodec.get_class_dim(),
            is_mhe=label_encodec.get_is_mhe(),
            other_arch_data=arch_data,
        )

        trainer_factory = cls._get_trainer_factory(
            train_args=train_data, is_mhe=label_encodec.get_is_mhe()
        )

        return cls(model=model, data=dataloader, trainer_factory=trainer_factory)

    @staticmethod
    def _get_model(
        model_name: str,
        in_dim: int,
        out_dim: int,
        is_mhe: bool,
        other_arch_data: yamlDict,
    ) -> nn.Module:

        return Models[model_name](
            input_feature_dim=in_dim, class_dim=out_dim, mhe=is_mhe, **other_arch_data
        )

    @staticmethod
    def _get_data_loader(
        pp_path: Path,
        batch_size: int,
        num_workers: int = 0,
        gpu_available: bool = False,
    ):
        dataset = Hdf5Dataset(pp_path / "processed_data.h5", split=Split.TRAIN)
        sampler = SubsetRandomSampler(dataset.get_indices())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_with_pack_pad_to_batch,
            num_workers=num_workers,
            pin_memory=gpu_available,
        )

    @staticmethod
    def _get_trainer_factory(train_args: yamlDict, is_mhe: bool = False) -> Trainer:
        loss = LossFn[
            f"{'mhe' if is_mhe else 'binary'}_{train_args['loss']}"
        ]()  # TODO: make args addable
        opti = Optimizer[train_args["optimizer"]]()  # TODO: make args addable
        return partial(
            Trainer, max_epochs=train_args["max_epochs"], loss_fn=loss, optimizer=opti
        )
