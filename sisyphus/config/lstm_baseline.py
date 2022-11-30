from recipe.preprocessing import PreprocessingJob

from erinyes.util.enums import Dataset
from sisyphus import tk

EXPERIMENT_NAME = "lstm_baseline"

def run_lstm_baseline():
    for dataset in Dataset:
        if dataset != Dataset.RAV:
            continue

        pp_job = PreprocessingJob(dataset=dataset)
        tk.register_output(f"{EXPERIMENT_NAME}/{dataset.name}/results", pp_job.out_pth)
        return pp_job.out_pth