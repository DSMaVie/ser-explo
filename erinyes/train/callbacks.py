from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from erinyes.train.trainer import Trainer

logger = logging.getLogger(__name__)


class TrackBestLoss:
    def __init__(self, val_data: DataLoader) -> None:
        self.data = val_data
        self.best_loss = -np.inf

    def __call__(self, trainer: Trainer):
        trainer.model.eval()
        loss = -np.inf
        for batch_idx, (x, y) in tqdm(
            enumerate(self.data),
            desc="Validation: Current Batch in Epoch",
        ):
            x = x.to(trainer._train_device)
            y = y.to(trainer._train_device)

            # calc model output and loss
            pred = trainer.model(x)
            loss += trainer.loss_fn(pred, y.long())

        avg_loss = loss / batch_idx
        if avg_loss > self.best_loss:
            self.best_loss = avg_loss
            trainer.save_state(trainer.save_pth / "best_val_loss")


class ResetIfNoImprovement:
    def __init__(self, val_data: DataLoader) -> None:
        self.data = val_data
        self.best_loss = -np.inf
        self.best_model = None

    def __call__(self, trainer: Trainer):
        trainer.model.eval()
        loss = -np.inf
        for batch_idx, (x, y) in tqdm(
            enumerate(self.data),
            desc="Validation: Current Batch in Epoch",
        ):
            x = x.to(trainer._train_device)
            y = y.to(trainer._train_device)

            # calc model output and loss
            pred = trainer.model(x)
            loss += trainer.loss_fn(pred, y.long())

        avg_loss = loss / batch_idx

        if (
            avg_loss < self.best_loss and self.best_model
        ):  # only do this of best model exists
            logger.info(
                f"Overwriting batch info because best_loss so far is {self.best_loss}, while this time we got an avg validation loss of {avg_loss}."
            )
            trainer.completed_batches -= 1
            trainer.completed_epochs -= 1
            trainer.model = self.best_model
        else:
            self.best_model = trainer.model
            self.best_loss = avg_loss


class CombineCallbacks:
    def __init__(self, callbacks: list[Callable[[Trainer]]]):
        self.callbacks = callbacks

    def __call__(self, trainer: Trainer):
        for cb in self.callbacks:
            cb(trainer)
