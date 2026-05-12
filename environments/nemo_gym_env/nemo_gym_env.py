import verifiers as vf


NEMO_ENV = "example_single_tool_call"


def load_environment(
    num_examples: int = -1,
    timeout_seconds: float | None = None,
) -> vf.Environment:
    limit = None if num_examples < 0 else num_examples
    taskset = vf.NeMoGymTaskset(
        nemo_env=NEMO_ENV,
        limit=limit,
    )
    harness = vf.NeMoGymHarness(
        nemo_env=NEMO_ENV,
        timeout_seconds=timeout_seconds,
    )
    return vf.Env(taskset=taskset, harness=harness)
