import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.util.enums import Split
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class InferenceJob(Job):
    def __init__(
        self, path_to_model_ckpts: tk.Path, path_to_data: tk.Path, rqmts: dict
    ):
        self.model_path = Path(path_to_model_ckpts)
        self.data_path = Path(path_to_data)
        self.rqmts = rqmts

        self.pred_out = self.output_path("inferences", directory=True)
        self.class_labels = self.output_var("class_labels.txt", pickle=True)

    def run(self):
        # select_classes
        le = torch.load(self.data_path / "label_encodec.pt")
        self.class_labels.set(le.classes)

        # select_models
        cp_num = 0
        for cp in self.model_path.iterdir():
            cp_parts = cp.stem.split("-")
            if not cp_parts[0] == "checkpoint":
                continue

            logger.info(f"found cb {cp_parts}")
            cp_num = int(cp_parts[1]) if int(cp_parts[1]) > cp_num else cp_num

        logger.info(f"found cp from step {cp_num}")
        self.model_path = self.model_path / "training" / f"checkpoint-{cp_num}"
        logger.info(f"model path is now set to {self.model_path}")

        # write inferences to disk
        model = AutoModel.from_pretrained(self.model_path).to("cuda")
        logger.info(f"found model {model} at cp at {self.model_path}")
        model.eval()
        for split in Split:
            data = Hdf5Dataset(
                src_path=self.data_path / "processed_data.h5", split=split
            )
            logger.info(f"data loaded for split {split}")
            for idx in tqdm(data.available_indices, desc=f"Infering on Split {split}"):
                x, y = data[idx]
                x.to("cuda")  # dirty
                model_out = model(input_values=x)
                hidden_states = model_out.hidden_states.to("cpu").numpy()

                with (
                    Path(self.pred_out.get_path()) / split.name.lower() / f"{idx}.txt"
                ).open("w") as file:
                    file.write(f"{y:d};{','.join(model_out.logits)}\n")
                    hid_iter = (",".join(hs) + "\n" for hs in hidden_states)
                    file.writelines(hid_iter)

    def tasks(self):
        yield Task("run")
