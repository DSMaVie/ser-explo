from sisyphus import Job, tk
from pathlib import Path

class InferenceJob(Job):
    def __init__(self,pth_to_model_ckpts:tk.Path, pth_to_data:tk.Path):
        self.model_pth = Path(pth_to_model_ckpts)
        self.data_pth = Path(pth_to_data)