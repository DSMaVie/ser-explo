import logging
import os
from pathlib import Path

from transformers import AutoModel, AutoProcessor

from erinyes.util.env import Env
from sisyphus import Job, Task

logger = logging.getLogger(__name__)


class DownloadPretrainedModelJob(Job):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name


        self.out = self.output_path( self.model_name.replace("/", "_"), directory=True)

    def download(self):
        model_loc = Path(self.out)

        logger.info(f"downloading {self.model_name} to {model_loc}")
        model = AutoModel.from_pretrained(self.model_name)
        model.save_pretrained(model_loc)

        logger.info("downloading processor alongside.")
        processor = AutoProcessor.from_pretrained(self.model_name)
        processor.save_pretrained(model_loc)

    def tasks(self):
        yield Task("download")
