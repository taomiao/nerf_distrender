config = {
    "models_repo": "/home/PJLAB/taomiao/PycharmProjects/distrender/models",
    "models": {
        "model_test": {
            "load": False,
            "do_opt": False,
            "use_pipe": False,
            "use_ddp": False
        },
        "model_leftsmall": {
            "load": False,
            "do_opt": False,
            "use_pipe": False,
            "use_ddp": False
        },
        "model_leftsmall_multi_stages": {
            "load": True,
            "do_opt": False,
            "use_pipe": True,
            "use_ddp": True
        },
    }
}
