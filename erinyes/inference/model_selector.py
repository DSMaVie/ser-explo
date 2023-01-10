from pathlib import Path

import torch

from erinyes.train.trainer import Trainer
import logging

logger = logging.getLogger(__name__)

class ModelSelector:
    def __init__(self, run_folder: Path, out_folder: Path) -> None:
        self.input = run_folder
        self.output = out_folder

    def pick_models(self):
        self.models = {}
        for pth in self.run_folder:
            if "best" in pth.parent.stem:
                monitor = pth.parent.stem.split("best")[1:]
                self.models.update({pth: monitor})

            if "last" in pth.parent.stem:
                self.models.update({pth: "last"})

    def predict(self, test_data: torch.utils.data.DataLoader):
        for pth, label in self.models.items():
            logger.info(f"Extracting best {label} model from {pth}")
            model = Trainer.from_state(pth).model

            logger.info("Extracting Test data and predicting on it.")
            preds, trues = [] , []
            for x, y in test_data:
                preds.extend(model(x).numpy())
                trues.extend(y.numpy())

            file_pth = self.output / f"{label}.txt"
            logger.info(f"writing predictions to.")
            with file_pth.open("a") as file:
                for pred, true in zip(preds, trues):
                    file.write(f"{true} --- {','.join(pred)}")
