from sisyphus import Job, tk, Task


class TrainJob(Job):
    def __init__(
        self, pth_to_arch_file: tk.Path, pth_to_train_settings: tk.Path
    ) -> None:
        super().__init__()

        self.pth_to_arch_file = pth_to_arch_file
        self.pth_to_train_settings = pth_to_train_settings


    def run(self):
        ...


    def tasks(self):
        yield Task("run")