def engine():
    from sisyphus.localengine import LocalEngine

    return LocalEngine(cpu=5, gpu=0, mem=8)


JOB_AUTO_CLEANUP = False
