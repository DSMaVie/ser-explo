def engine():
    from sisyphus.localengine import LocalEngine
    return LocalEngine(cpu=2, gpu=0, mem=8)