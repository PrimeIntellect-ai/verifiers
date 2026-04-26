import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def load_taskset(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
) -> vf.Taskset:
    def source():
        return load_example_dataset(dataset_name, dataset_split, n=num_train_examples)

    parser = vf.Parser(extract_fn=extract_boxed_answer)
    return vf.Taskset(
        source=source,
        rubric=vf.MathRubric(parser=parser),
        name="math-python-th",
    )


def load_harness(
    max_turns: int = 100,
    max_startup_wait_seconds: int = 60,
    pip_install_packages: str = "numpy sympy scipy",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 60,
    sandbox_timeout_per_command_seconds: int = 60,
    sandbox_client_max_workers: int = 50,
) -> vf.Harness:
    pip_install_prompt = (
        f"In addition to the Python standard library, you have access to: {pip_install_packages}."
        if pip_install_packages.strip()
        else "You may only use the Python standard library."
    )
    system_prompt = (
        "Use Python for all calculations. Give your answer inside \\boxed{}."
        "\n\n"
        f"{pip_install_prompt}"
    )

    @vf.cleanup(priority=50)
    async def cleanup_python_state(
        _task: vf.Task, state: vf.State, _resources: vf.Resources
    ) -> None:
        if "python_state" in state:
            state["python_cleanup"] = {
                "execution_count": state["python_state"]["execution_count"],
                "ready": state["python_state"]["ready"],
            }

    python = vf.SandboxPythonTool(
        sandbox=vf.SandboxSpec(
            image="python:3.11-slim",
            cpu_cores=sandbox_cpu_cores,
            memory_gb=sandbox_memory_gb,
            disk_size_gb=sandbox_disk_size_gb,
            gpu_count=sandbox_gpu_count,
            timeout_minutes=sandbox_timeout_minutes,
        ),
        command_timeout=sandbox_timeout_per_command_seconds,
        max_startup_wait_seconds=max_startup_wait_seconds,
        pip_install_packages=pip_install_packages,
        sandbox_key="math-python",
        sandbox_runtime={"client_max_workers": sandbox_client_max_workers},
    )
    return vf.Harness(
        system_prompt=system_prompt,
        tools=[
            vf.Toolset(
                tools=[python],
                channels={"cleanup": {"harness": cleanup_python_state}},
                name="math_python_tools",
            )
        ],
        run=vf.RunConfig(max_turns=max_turns, stop_errors=(vf.SandboxError,)),
    )


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    max_turns: int = 100,
    max_startup_wait_seconds: int = 60,
    pip_install_packages: str = "numpy sympy scipy",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 60,
    sandbox_timeout_per_command_seconds: int = 60,
    sandbox_client_max_workers: int = 50,
) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_train_examples=num_train_examples,
        ),
        harness=load_harness(
            max_turns=max_turns,
            max_startup_wait_seconds=max_startup_wait_seconds,
            pip_install_packages=pip_install_packages,
            sandbox_cpu_cores=sandbox_cpu_cores,
            sandbox_memory_gb=sandbox_memory_gb,
            sandbox_disk_size_gb=sandbox_disk_size_gb,
            sandbox_gpu_count=sandbox_gpu_count,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            sandbox_timeout_per_command_seconds=sandbox_timeout_per_command_seconds,
            sandbox_client_max_workers=sandbox_client_max_workers,
        ),
    )
