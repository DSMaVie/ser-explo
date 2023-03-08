from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Protocol

logger = logging.getLogger(__name__)


class OptimizerType(Protocol):
    def step(self):
        ...

    def zero_grad(self):
        ...


class InTrainingCallback(Protocol):
    def after_batch(self, train_state: Trainer):
        ...

    def after_epoch(self, train_state: Trainer):
        ...


@dataclass
class Trainer:
    def __init__(
        self,
        max_epochs: int,
        loss_fn: Callable[
            [torch.TensorType | list[torch.TensorType]], torch.TensorType
        ],
        optimizer: OptimizerType,
        save_pth: Path,
        model: nn.Module | None = None,
        train_data: DataLoader | None = None,
        gpu_available: bool = False,
        callbacks: list[InTrainingCallback] | None = None,
    ):
        logger.info("initalizing trainer class")
        self.max_epochs = max_epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_pth = save_pth

        self.model = model
        self.train_data = train_data

        self.current_loss = -np.inf
        self.completed_epochs = 0
        self.completed_batches = 0
        self._train_device = "cuda" if gpu_available else "cpu"

        self.callbacks = callbacks
        logger.info("init of trainer done.")

    def fit(self):
        if self.completed_epochs == self.max_epochs:
            warnings.warn("Not fitting. Fit already done!")

        if not self.model or not self.train_data:
            raise AttributeError("Either Model or Train_Data has not been set prior!")

        logger.info(f"using model of class {self.model.__class__}")
        self.model.train()
        self.model.to(device=self._train_device)
        if "cuda" in self._train_device:
            logger.info("send model to gpu")

        logger.info("starting training!")
        for epoch_idx in tqdm(
            range(self.completed_epochs + 1, self.max_epochs),
            desc="Epoch",
            initial=self.completed_epochs,
            total=self.max_epochs,
        ):
            logger.info(f"starting epoch {epoch_idx}")
            for batch_idx, (x, y) in tqdm(
                enumerate(self.train_data, start=self.completed_batches + 1),
                desc="Current Batch in Epoch",
            ):
                self.optimizer.zero_grad()

                x = x.to(self._train_device)
                y = y.to(self._train_device)

                # calc model output and loss
                pred = self.model(x)
                self.current_loss = self.loss_fn(pred, y)

                # optimize
                self.current_loss.backward()
                self.optimizer.step()

                self.completed_batches = batch_idx
                # if self.callbacks:
                #     for cb in self.callbacks:
                #         cb.after_batch(self)

            # reset batch number
            self.completed_epochs = epoch_idx
            self.save_state(self.save_pth / "last")

            # if self.callbacks:
            #     for cb in self.callbacks:
            #         cb.after_epoch(self)

        logger.info("Finished training!")
        return self.model

    def profile(self, data: DataLoader, log_path: str):
        self.model.eval()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with record_function("model_inference"):
                for x, y in data:
                    self.model(x)

            print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
            prof.export_chrome_trace(log_path + "trace.json")

    def save_state(self, pth: Path):
        os.makedirs(pth, exist_ok=True)
        torch.save(self.model, pth / "model.pt")
        torch.save(self.train_data, pth / "train_data.pt")
        torch.save(
            {
                "epoch_idx": self.completed_epochs,
                "batch_idx": self.completed_batches,
                "loss_fn": self.loss_fn,
                "optimizer": self.optimizer,
                "max_epochs": self.max_epochs,
                "after_epoch": self.after_epoch,
                "after_update": self.after_update,
            },
            pth / "train_state.pt",
        )

    @classmethod
    def from_state(cls, pth: Path):
        state_dict = torch.load(pth / "train_state.pt")
        inst = cls(
            max_epochs=state_dict["max_epochs"],
            loss_fn=state_dict["loss_fn"],
            optimizer=state_dict["optimizer"],
            save_pth=pth,
        )
        inst.completed_batches = state_dict["batch_idx"]
        inst.completed_epochs = state_dict["epoch_idx"]

        model_state = torch.load(pth / "model.pt")
        inst.model = model_state
        inst.train_data = torch.load(pth / "train_data.pt")

        inst.after_epoch = state_dict["after_epoch"]
        inst.after_update = state_dict["after_update"]
        return inst
