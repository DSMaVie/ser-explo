import logging

from recipe.preprocessing import PreprocessingJob

from erinyes.util.env import Env
from sisyphus import tk

EXPERIMENT_NAME = "lstm_baseline"

logger = logging.getLogger(__name__)


def run_lstm_baseline():
    inst_dir = Env.load().ROOT_DIR / "data" / "instructions" / "pp"
    outs = []
    for pth in inst_dir.rglob("*.yaml"):
        logger.info(f"Found instructions at {pth}. Starting PPLob for it.")
        pp_job = PreprocessingJob(tk.Path(str(pth)))
        tk.register_output(f"{EXPERIMENT_NAME}/pp/{pth.stem}", pp_job.out_pth)
        print(pp_job.out_pth)
        outs.append(pp_job.out_pth)
    return outs
