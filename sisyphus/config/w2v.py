import logging

from recipe.preprocessing import PreprocessingJob
from recipe.train import TrainJob

from erinyes.util.env import Env
from sisyphus import tk

EXPERIMENT_NAME = "w2v_baselines"

logger = logging.getLogger(__name__)


def run_w2v_baseline():
    pp_inst_dir = Env.load().INST_DIR / "pp" / "raw"
    # pp_outputs = {}
    for pth in pp_inst_dir.rglob("*.yaml"):
        logger.info(f"Found instructions at {pth}. Starting Preprocessing for it.")
        pp_job = PreprocessingJob(tk.Path(str(pth)))

        tk.register_output(f"{EXPERIMENT_NAME}/{pth.stem}/processed", pp_job.out_pth)

        train_info = tk.Path(
            str(Env.load().INST_DIR / "train" / f"{EXPERIMENT_NAME}.yaml")
        )
        train_job = TrainJob(
            pth_to_pp_output=pp_job.out, pth_to_train_settings=train_info
        )

        tk.register_output(f"{EXPERIMENT_NAME}/{pth.stem}/trained", train_job.out_pth)