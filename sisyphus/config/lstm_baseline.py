import logging
from pathlib import Path

from recipe.preprocessing import PreprocessingJob
from recipe.train import TrainJob

from erinyes.util.env import Env
from sisyphus import tk

EXPERIMENT_NAME = "lstm_baseline"

logger = logging.getLogger(__name__)


def run_lstm_baseline():
    pp_inst_dir = Env.load().INST_DIR / "pp"
    pp_outputs = {}
    for pth in pp_inst_dir.rglob("*.yaml"):
        logger.info(f"Found instructions at {pth}. Starting PPLob for it.")
        pp_job = PreprocessingJob(tk.Path(str(pth)))
        tk.register_output(f"{EXPERIMENT_NAME}/pp/{pth.stem}", pp_job.out_pth)
        pp_outputs.update({
            pth.stem : pp_job.out_pth
        })

    train_info = tk.Path(str(Env.load().INST_DIR / "train" / f"{EXPERIMENT_NAME}.yaml"))
    train_outs = []
    for pp_name, pp_out_pth in pp_outputs.items():
        train_job = TrainJob(
            pth_to_pp_output=pp_out_pth, pth_to_train_settings=train_info
        )
        train_outs.append(train_job.out_pth)
        tk.register_output(  # TODO: double out to MOS
            f"{EXPERIMENT_NAME}/training/{pp_name}", train_job.out_pth
        )
    return pp_outputs
