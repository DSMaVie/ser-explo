from pathlib import Path

from erinyes.data.loader import get_data_loader
from erinyes.inference.model_selector import ModelSelector
from erinyes.inference.testbench import Testbench
from erinyes.util.enums import Split
from sisyphus import Job, Task, tk


class InferenceJob(Job):
    def __init__(
        self,
        pth_to_inf_instructs: tk.Path,
        pth_to_model_ckpts: tk.Path,
        pth_to_data: tk.Path,
    ):
        self.model_pth = Path(pth_to_model_ckpts)
        self.data_pth = Path(pth_to_data)
        self.inst_pth = Path(pth_to_inf_instructs)

        self.pred_out = self.output_path("predictions", directory=True)
        self.out = self.output_var("results")

    def pick_models(self):
        selector = ModelSelector(self.model_pth, self.pred_out)
        selector.pick_models()

        test_data = get_data_loader(self.data_pth, batch_size=128, split=Split.TEST)
        selector.predict(test_data=test_data)

    def analyze_models(self):
        tb = Testbench.from_yaml(self.inst_pth)
        res = tb.test(self.pred_out)
        self.out.set(res)

    def tasks(self):
        yield Task("pick_models")
        yield Task("analyze_models")
