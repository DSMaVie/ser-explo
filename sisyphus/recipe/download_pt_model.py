from __future__ import annotations

import logging
import os
from pathlib import Path

from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2PhonemeCTCTokenizer,
)

from sisyphus import Job, Task

logger = logging.getLogger(__name__)


class DownloadPretrainedModelJob(Job):
    def __init__(
        self,
        model_name: str,
        is_ctc_model: bool = False,
        rqmts: dict | None = None,
        just_model: bool = False,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.is_ctc_model = is_ctc_model
        self.just_model = just_model
        self.rqmts = rqmts

        self.out = self.output_path(self.model_name.replace("/", "_"), directory=True)

    def download(self):
        model_loc = Path(self.out)

        logger.info(f"downloading {self.model_name} to {model_loc}")
        if self.is_ctc_model:
            model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        else:
            model = Wav2Vec2Model.from_pretrained(self.model_name)

        model.save_pretrained(model_loc)

        if not self.just_model:
            logger.info("downloading processor alongside.")
            processor = AutoProcessor.from_pretrained(self.model_name)
            processor.save_pretrained(model_loc)

    def tasks(self):
        yield Task("download", rqmt=self.rqmts)


class DownloadPretrainedModelWithPhonemeTokenzier(DownloadPretrainedModelJob):
    def download(self):
        super().download()
        logger.info("overwriting tokenizer with phoneme tokenizer")
        logger.info(os.environ)
        Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).save_pretrained(Path(self.out))
