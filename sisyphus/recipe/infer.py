import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.util.enums import Split
from sisyphus import Job, Task
from sisyphus import global_settings as gs
from sisyphus import tk

logger = logging.getLogger(__name__)


class ClfInferenceJob(Job):
    def __init__(
        self,
        path_to_model_ckpts: tk.Path,
        path_to_data: tk.Path,
        rqmts: dict,
        model_args: tk.Variable,
        model_class: tk.Variable,
    ):
        self.model_path = Path(path_to_model_ckpts)
        self.data_path = path_to_data
        self.rqmts = rqmts
        self._train_device = "cuda" if self.rqmts.get("gpu", False) else "cpu"

        self.pred_out = self.output_path("inferences", directory=True)
        self.class_labels = self.output_var("class_labels.txt", pickle=True)

        self.model_class = model_class
        self.model_args = model_args

    def run(self):
        # select_classes
        le = torch.load(Path(self.data_path) / "label_encodec.pt")
        self.class_labels.set(le.classes)

        data_path = gs.file_caching(self.data_path.join_right("processed_data.h5"))

        # reconstruct model
        model = (
            (self.model_class.get())
            .from_pretrained(self.model_path, **self.model_args.get())
            .to(self._train_device)
        )

        model.__repr__ = lambda: ""
        logger.info(f"found model {model} at cp at {self.model_path}")

        model.eval()
        # write inferences to disk
        with torch.no_grad():
            for split in Split:
                split_path = Path(self.pred_out.get_path()) / split.name.lower()
                os.makedirs(split_path, exist_ok=True)

                data = Hdf5Dataset(src_path=data_path, split=split)
                logger.info(f"data loaded for split {split}")
                for idx in tqdm(
                    data.available_indices, desc=f"Infering on Split {split}"
                ):
                    x, y = data[idx]
                    x = x.unsqueeze(dim=0)
                    x = x[:, : 20 * 16000].to(self._train_device)

                    model_out = model(input_values=x)
                    logits = model_out.logits.cpu().detach().numpy()[0]

                    res_string = (
                        f"{y.numpy()[0]:n};{','.join(str(logit) for logit in logits)}\n"
                    )

                    with (split_path / f"{idx}.txt").open("w+") as file:
                        file.write(res_string)

    def tasks(self):
        yield Task("run", rqmt=self.rqmts)


class Seq2SeqInferenceJob(Job):
    def __init__(
        self,
        path_to_model_ckpts: tk.Path,
        path_to_data: tk.Path,
        rqmts: dict,
        model_args: tk.Variable,
        model_class: tk.Variable,
    ):
        self.model_path = Path(path_to_model_ckpts)
        self.data_path = path_to_data
        self.rqmts = rqmts
        self._train_device = "cuda" if self.rqmts.get("gpu", False) else "cpu"

        self.pred_out = self.output_path("inferences", directory=True)
        self.class_labels = self.output_var("class_labels.txt", pickle=True)

        self.model_class = model_class
        self.model_args = model_args

    def run(self):
        # select_classes
        le = torch.load(Path(self.data_path) / "label_encodec.pt")
        self.class_labels.set(le.classes)

        data_path = gs.file_caching(self.data_path.join_right("processed_data.h5"))

        # reconstruct model
        model = (
            (self.model_class.get())
            .from_pretrained(self.model_path, **self.model_args.get())
            .to(self._train_device)
        )
        logger.info(f"found model {model} at cp at {self.model_path}")

        model.eval()
        # write inferences to disk
        for split in Split:
            split_path = Path(self.pred_out.get_path()) / split.name.lower()
            os.makedirs(split_path, exist_ok=True)

            data = Hdf5Dataset(src_path=data_path, split=split)
            logger.info(f"data loaded for split {split}")
            for idx in tqdm(data.available_indices, desc=f"Infering on Split {split}"):
                x, y = data[idx]
                x = x.unsqueeze(dim=0).to(self._train_device)

                model_out = model(input_values=x)
                logits_seq = model_out.logits.cpu().detach().numpy()[0]

                res_string = "#" + ",".join(str(val) for val in y.long()[0].numpy()) + "\n"

                with (split_path / f"{idx}.txt").open("w+") as file:
                    file.write(res_string)
                    hid_iter = (
                        ",".join(str(h_value) for h_value in logits) + "\n"
                        for logits in logits_seq
                    )
                    file.writelines(hid_iter)

    def tasks(self):
        yield Task("run", rqmt=self.rqmts)
