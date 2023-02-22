import logging

from erinyes.util.env import Env
from sisyphus import tk
from recipe.preprocessing import PreprocessingJob

EXPERIMENT_NAME = "w2v_baselines"

logger = logging.getLogger(__name__)


def run_w2v_baseline():
    pp_inst_dir = Env.load().INST_DIR / "pp" / "raw"
    # pp_outputs = {}
    for pth in pp_inst_dir.rglob("*.yaml"):
        logger.info(f"Found instructions at {pth}. Starting Preprocessing for it.")
        pp_job = PreprocessingJob(tk.Path(str(pth)))
        tk.register_output(f"{EXPERIMENT_NAME}/pp/raw/{pth.stem}", pp_job.out_pth)
