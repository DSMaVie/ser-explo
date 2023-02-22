def engine():
    from dotenv import dotenv_values, find_dotenv
    # from krylov_engine import KrylovEngine

    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine

    secrets = dotenv_values(find_dotenv(".secrets.env"))

    return EngineSelector(
        {
            "local": LocalEngine(cpus=2, gpus=0, mem=4),
            # "krylov": KrylovEngine(
            #     project_name=secrets["PROJECT_NAME"],
            #     default_rqmt={"cpu": 4, "gpu": 1, "mem": 20, "time":1 ,"docker_image":secrets["DOCKER_IMAGE"]},
            # ),
        },
        default_engine="local",
    )


JOB_AUTO_CLEANUP = False
