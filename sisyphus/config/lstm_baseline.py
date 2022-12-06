from recipe.preprocessing import PreprocessingJob

from erinyes.util.env import Env
from sisyphus import tk

EXPERIMENT_NAME = "lstm_baseline"

def run_lstm_baseline():
    env = Env.load()
    for pth in env.RAW_DIR.rglob("*.yaml"):
        print(pth)
        pp_job = PreprocessingJob(pth)
        tk.register_output(f"{EXPERIMENT_NAME}/pp/{pth.stem}", pp_job.out_pth)
        return pp_job.out_pth