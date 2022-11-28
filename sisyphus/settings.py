def engine():
    from sisyphus.localengine import LocalEngine
    return LocalEngine(cpu=4, gpu=0, mem=16)