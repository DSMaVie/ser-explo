import getpass
import importlib
import sys
import typing
sys.path.append("/usr/local/")
sys.path.append("/usr/local/cache-manager/")

cm = importlib.import_module("cache-manager")


# WORK_DIR = "work"
# IMPORT_PATHS = ["config", "recipe/"]


def TODO():
    raise Exception("Missing parameter.")


def check_engine_limits(current_rqmt, task):
    if "gpu" not in current_rqmt or current_rqmt["gpu"] == 0:
        current_rqmt["qsub_args"] = current_rqmt.get("qsub_args", []) + [
            "-l",
            "hostname=(*-10|*-11|*-12|*-13|*-14|*-15|*-16|*-17|*-18|*-19|*-20|*-21|*-22|*-23|*-24|*-25)",
        ]

    current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
    return current_rqmt


def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.son_of_grid_engine import SonOfGridEngine

    default_rqmt = {
        "cpu": 1,
        "mem": 1,
        "gpu": 0,
        "time": 1,
        "qsub_args": ["-l", "hostname=!(*-cn-214|*-cn-215|*-cn-216)"],
    }

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=2, mem=8),
            "long": SonOfGridEngine(default_rqmt=default_rqmt),
        },  # , gateway='cluster-cn-02')},
        default_engine="long",
    )


def file_caching(file: str) -> str:
    from sisyphus import tk
    path = file.get_path() if isinstance(file, tk.Path) else file
    return cm.cacheFile(path)

MAIL_ADDRESS = getpass.getuser()

# Application specific settings
PYTHON_EXE = "/work/smt4/thulke/vieweg/SER/.venv/bin/python"  # Add path to your python executable

DEFAULT_ENVIRONMENT_SET["HF_HOME"] = "/work/smt4/thulke/vieweg/.cache"
DEFAULT_ENVIRONMENT_SET[
    "TRANSFORMERS_CACHE"
] = "/work/smt4/thulke/vieweg/.cache/torch/transformers"
DEFAULT_ENVIRONMENT_SET[
    "HF_DATASETS_CACHE"
] = "/work/smt4/thulke/vieweg/.cache/huggingface/datasets"
DEFAULT_ENVIRONMENT_SET[
    "HF_METRICS_CACHE"
] = "/work/smt4/thulke/vieweg/.cache/huggingface/metrics"

SIS_COMMAND = [PYTHON_EXE, sys.argv[0]]
MAX_PARALLEL = 20
SGE_SSH_COMMANDS = [" source /u/standard/settings/sge_settings.sh; "]

# how many seconds should be waited before ...
WAIT_PERIOD_JOB_FS_SYNC = 30  # finishing a job
WAIT_PERIOD_BETWEEN_CHECKS = 30  # checking for finished jobs
WAIT_PERIOD_CACHE = 30  # stoping to wait for actionable jobs to appear
WAIT_PERIOD_SSH_TIMEOUT = 30  # retrying ssh connection
WAIT_PERIOD_QSTAT_PARSING = 30  # retrying to parse qstat output
WAIT_PERIOD_HTTP_RETRY_BIND = 30  # retrying to bind to the desired port
WAIT_PERIOD_JOB_CLEANUP = 30  # cleaning up a job
WAIT_PERIOD_MTIME_OF_INPUTS = (
    60  # wait X seconds long before starting a job to avoid file system sync problems
)

PRINT_ERROR_LINES = 1
SHOW_JOB_TARGETS = False
CLEAR_ERROR = False  # set true to automatically clean jobs in error state

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"

versions = {"cuda": "10.1", "acml": "4.4.0", "cudnn": "7.6"}

DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join(
    [
        "/usr/local/cudnn-{cuda}-v{cudnn}/lib64".format(**versions),
        "/usr/local/cuda-{cuda}/lib64".format(**versions),
        "/usr/local/cuda-{cuda}/extras/CUPTI/lib64".format(**versions),
        "/usr/local/acml-{acml}/cblas_mp/lib".format(**versions),
        "/usr/local/acml-{acml}/gfortran64/lib".format(**versions),
        "/usr/local/acml-{acml}/gfortran64_mp/lib/".format(**versions),
    ]
)

DEFAULT_ENVIRONMENT_SET[
    "PHONEMIZER_ESPEAK_LIBRARY"
] = "/u/vieweg/.local/lib/libespeak-ng.so"

DEFAULT_ENVIRONMENT_SET["PATH"] = (
    "/usr/local/cuda-10.1/bin:" + DEFAULT_ENVIRONMENT_SET["PATH"]
)
DEFAULT_ENVIRONMENT_SET["HDF5_USE_FILE_LOCKING"] = "FALSE"
