import verifiers as vf


def load_environment(
    gym: str | list[str | dict] = "arc_1d",
    num_train_examples: int = 2000,
    num_eval_examples: int = 2000,
    seed: int = 0,
):
    vf_env = vf.ReasoningGymEnv(
        gym=gym,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
    )
    return vf_env
