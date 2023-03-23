from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Protocol

logger = logging.getLogger(__name__)


class InTrainingCallback(Protocol):
    def after_batch(self, train_state: Trainer):
        ...

    def after_epoch(self, train_state: Trainer):
        ...


LossType = Callable[[torch.tensor, torch.tensor], torch.tensor]

RecipeType = TypeVar(
    "RecipeType",
    LossType,
    InTrainingCallback,
    torch.optim.Optimizer,
)


@dataclass
class ObjectRecipe(Generic[RecipeType]):
    name: str
    instance: RecipeType
    args: dict | None = None

    def create_instance(self, *args, **kwargs):
        if self.args is not None:
            return self.instance(*args, **kwargs, **self.args)
        return self.instance(*args, **kwargs)


@dataclass
class Trainer:
    def __init__(
        self,
        max_epochs: int,
        loss_recipe: ObjectRecipe[LossType],
        optimizer_recipe: ObjectRecipe[torch.optim.Optimizer],
        save_pth: Path,
        gpu_available: bool = False,
        callback_recipes: list[ObjectRecipe[InTrainingCallback]] | None = None,
    ):
        logger.info("initalizing trainer class")
        self.max_epochs = max_epochs

        self.loss_recipe = loss_recipe
        self.optimizer_recipe = optimizer_recipe
        self.save_pth = save_pth

        self.current_loss = -np.inf
        self.completed_epochs = 0
        self._train_device = "cuda" if gpu_available else "cpu"

        self.callback_recipes = callback_recipes
        logger.info("init of trainer done.")

        self.is_preped = False

    def prepare(self, model: nn.Module, train_data: DataLoader):
        logger.info("starting preparations.")
        self.model = model
        self.data = train_data

        self.loss = self.loss_recipe.create_instance()
        self.callbacks = [cbr.create_instance() for cbr in self.callback_recipes]

        # preload optimizer
        self.opti = self.optimizer_recipe.create_instance(model.parameters())

        self.is_preped = True
        logger.info("preparation done.")

    def fit(self):
        if self.completed_epochs == self.max_epochs:
            warnings.warn("Not fitting. Fit already done!")

        if not self.is_preped:
            raise AttributeError(
                "Trainer has not been has not been preped prior! Please run Trainer.perpare"
            )

        logger.info(f"using model of class {self.model.__class__}")
        self.model.train()
        self.model.to(device=self._train_device)

        if "cuda" in self._train_device:
            logger.info("send model to gpu")

        logger.info("starting training!")
        for current_epoch_idx in tqdm(
            range(self.completed_epochs + 1, self.max_epochs),
            desc="Epoch",
            initial=self.completed_epochs,
            total=self.max_epochs,
        ):
            logger.info(f"starting epoch {current_epoch_idx}")
            for x, y in tqdm(self.data, desc="Current Batch in Epoch"):
                self.opti.zero_grad()

                x = x.to(self._train_device)
                y = y.to(self._train_device)

                # calc model output and loss
                pred = self.model(x)
                self.current_loss = self.loss(pred, y)

                # optimize
                self.current_loss.backward()
                self.opti.step()

                if self.callbacks:
                    for cb in self.callbacks:
                        cb.after_batch(self)

            # reset batch number
            self.completed_epochs = current_epoch_idx
            self.save_state(self.save_pth / "last")

            if self.callbacks:
                for cb in self.callbacks:
                    cb.after_epoch(self)

        logger.info("Finished training!")
        return self.model

    def profile(self, log_path: str):
        if not self.is_preped:
            raise AttributeError(
                "Trainer has not been has not been preped prior! Please run Trainer.perpare"
            )

        self.model.to(device=self._train_device)

        if "cuda" in self._train_device:
            logger.info("send model to gpu")

        self.model.eval()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with record_function("model_inference"):
                for x, y in self.data:
                    self.model(x)

            print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
            prof.export_chrome_trace(log_path + "trace.json")

    def save_state(self, pth: Path):
        os.makedirs(pth, exist_ok=True)
        torch.save(self.model, pth / "model.pt")
        torch.save(self.data, pth / "data.pt")
        torch.save(
            {
                "completed_epochs": self.completed_epochs,
                "max_epochs": self.max_epochs,
                "gpu_available": self._train_device == "cuda",
                "loss_recipe": self.loss_recipe,
                "optimizer_recipe": self.optimizer_recipe,
                "callback_recipes": self.callback_recipes,
            },
            pth / "train_state.pt",
        )

    @classmethod
    def from_state(cls, pth: Path):
        state_dict = torch.load(pth / "train_state.pt")
        inst = cls(
            max_epochs=state_dict["max_epochs"],
            loss_recipe=state_dict["loss_recipe"],
            optimizer_recipe=state_dict["optimizer_recipe"],
            save_pth=pth.parent,
            gpu_available=state_dict["gpu_available"],
            callback_recipes=state_dict["callback_recipes"],
        )
        inst.completed_epochs = state_dict["completed_epochs"]

        model_state = torch.load(pth / "model.pt")
        train_data = torch.load(pth / "data.pt")

        inst.prepare(model_state, train_data)
        return inst
