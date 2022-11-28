from erinyes.util.enums import Dataset
from sisyphus import tk
from recipe.preprocessing import PreProcessingJob

EXPERIMENT_NAME = "lstm_baseline"

def run_lstm_baseline():
    for dataset in Dataset:
        if dataset != Dataset.RAV:
            continue

        pp_job = PreProcessingJob(dataset=dataset) # argumetns
        tk.register_output(f"{EXPERIMENT_NAME}/{dataset.name}/results", pp_job.output)
        return pp_job.output