from pathlib import Path

import verifiers as vf


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        openenv_project=Path(__file__).parent / "proj",
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
    )
