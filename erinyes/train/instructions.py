from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

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

logger = logging.getLogger(__name__)


@dataclass
class TrainingsInstructions:
    model: nn.Module
    train_data: DataLoader
    val_data: DataLoader
    trainer_factory: Callable[..., Trainer]

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

        train_dataloader = cls._get_data_loader(
            pth_to_pp_output, batch_size=train_data["batch_size"], split=Split.TRAIN
        )
        val_dataloader = cls._get_data_loader(
            pth_to_pp_output, batch_size=train_data["batch_size"], split=Split.VAL
        )

        model = cls._get_model(
            model_name=train_data["model"],
            in_dim=feature_extractor.get_feature_dim(),
            out_dim=label_encodec.get_class_dim(),
            is_mhe=label_encodec.get_is_mhe(),
            other_arch_data=arch_data,
        )

        trainer_factory = cls._get_trainer_factory(
            model.parameters(), train_args=train_data, is_mhe=label_encodec.get_is_mhe()
        )

        return cls(
            model=model,
            train_data=train_dataloader,
            val_data=val_dataloader,
            trainer_factory=trainer_factory,
        )

    @staticmethod
    def _get_model(
        model_name: str,
        in_dim: int,
        out_dim: int,
        is_mhe: bool,
        other_arch_data: yamlDict,
    ) -> nn.Module:

        model_inst = Models[model_name].value
        return model_inst(
            input_feature_dim=in_dim, class_dim=out_dim, mhe=is_mhe, **other_arch_data
        )

    @staticmethod
    def _get_data_loader(
        pp_path: Path,
        batch_size: int,
        split: Split,
        num_workers: int = 0,
        gpu_available: bool = False,
    ):
        dataset = Hdf5Dataset(pp_path / "processed_data.h5", split=split)
        logger.info(f"found {len(dataset.get_indices())} examples in the train set")
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
    def _get_trainer_factory(
        model_params: Iterator[nn.Parameter], train_args: yamlDict, is_mhe: bool = False
    ) -> Trainer:
        loss = LossFn[
            f"{'binary' if is_mhe else 'mc'}_{train_args['loss']}"
        ].value()  # pot. args in last bracket
        opti = Optimizer[train_args["optimizer"]].value(
            model_params
        )  # TODO: make args addable
        return partial(
            Trainer, max_epochs=train_args["max_epochs"], loss_fn=loss, optimizer=opti
        )
