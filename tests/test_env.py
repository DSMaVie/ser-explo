from pathlib import Path
from erinyes.util.env import Env

def test_env():
    env = Env.load()
    assert isinstance(env.RAW_DIR, Path)
    assert isinstance(env.ROOT_DIR, Path)
    assert isinstance(env.MODEL_DIR, Path)
