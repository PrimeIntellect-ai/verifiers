import verifiers as vf


def load_environment(
    openenv_project: str = "openenv/echo-env",
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        openenv_project=openenv_project,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
    )
