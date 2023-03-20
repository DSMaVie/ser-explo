import logging
from itertools import product
from pathlib import Path

from erinyes.data.loader import get_data_loader
from erinyes.train.trainer import Trainer
from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class ModelSelector:
    def __init__(self, run_folder: Path, out_folder: Path, data_loc: Path) -> None:
        self.input = run_folder
        self.output = out_folder
        self.data_loc = data_loc

    def pick_models(self):
        models = {}
        for pth in self.input.iterdir():
            logger.info(f"looking for models in pth {pth}")
            if "best" in pth.stem:
                monitor = pth.stem.split("best")[1:]
                models.update({pth: monitor})

            if "last" in pth.stem:
                models.update({pth: "last"})

        return models

    def load_data(self, num_workers: int = 0, gpu_available: bool = False):
        ret_dict = {}
        for split in Split:
            return {
                split: get_data_loader(
                    self.data_loc,
                    batch_size=128,
                    split=split,
                    num_workers=num_workers,
                    gpu_available=gpu_available,
                    pack=False,
                )
            }

    def predict(self, num_workers: int = 0, gpu_available: bool = False):
        data = self.load_data(num_workers, gpu_available)
        models = self.pick_models()

        for (pth, model), (split, dloader) in product(models, data):
            pass

        # for pth, label in self.models.items():
        #     logger.info(f"Extracting best {label} model from {pth}")
        #     model = Trainer.from_state(pth).model

        #     logger.info("Extracting Test data and predicting on it.")
        #     preds, trues = [], []
        #     for x, y in test_data:
        #         preds.extend(model(x).numpy())
        #         trues.extend(y.numpy())

        #     file_pth = self.output / f"{label}.txt"
        #     logger.info(f"writing predictions to.")
        #     with file_pth.open("a") as file:
        #         for pred, true in zip(preds, trues):
        #             file.write(f"{true} --- {','.join(pred)}")
