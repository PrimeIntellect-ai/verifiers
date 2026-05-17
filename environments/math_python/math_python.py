import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


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
    sandbox_client_max_workers: int | None = None,
    v1: bool = False,
    **kwargs,
):
    if v1:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported v1 load_environment kwargs: {unexpected}")

        from math_python_v1 import (
            MathPythonEnvConfig,
            MathPythonHarnessConfig,
            MathPythonTasksetConfig,
            build_system_prompt,
            load_v1_environment,
        )

        return load_v1_environment(
            config=MathPythonEnvConfig(
                taskset=MathPythonTasksetConfig(
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                    num_train_examples=num_train_examples,
                    system_prompt=build_system_prompt(pip_install_packages),
                ),
                harness=MathPythonHarnessConfig(
                    max_turns=max_turns,
                    toolsets={
                        "python": {
                            "fn": "math_python_v1:load_toolset",
                            "pip_install_packages": pip_install_packages,
                            "sandbox_cpu_cores": sandbox_cpu_cores,
                            "sandbox_memory_gb": sandbox_memory_gb,
                            "sandbox_disk_size_gb": sandbox_disk_size_gb,
                            "sandbox_gpu_count": sandbox_gpu_count,
                            "sandbox_timeout_minutes": sandbox_timeout_minutes,
                            "sandbox_timeout_per_command_seconds": sandbox_timeout_per_command_seconds,
                        }
                    },
                ),
            )
        )

    def build_dataset():
        return load_example_dataset(dataset_name, dataset_split, n=num_train_examples)

    pip_install_prompt = (
        f"In addition to the Python standard library, you have access to: {pip_install_packages}."
        if pip_install_packages.strip()
        else "You may only use the Python standard library."
    )
    system_prompt = (
        "Use Python for all calculations. Give your answer inside \\boxed{}."
    )
    system_prompt += "\n\n" + pip_install_prompt

    parser = vf.Parser(extract_fn=extract_boxed_answer)
    math_rubric = vf.MathRubric(parser=parser)
    return vf.PythonEnv(
        dataset=build_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=math_rubric,
        max_turns=max_turns,
        # python env args
        max_startup_wait_seconds=max_startup_wait_seconds,
        pip_install_packages=pip_install_packages,
        # sandbox env args
        cpu_cores=sandbox_cpu_cores,
        memory_gb=sandbox_memory_gb,
        disk_size_gb=sandbox_disk_size_gb,
        gpu_count=sandbox_gpu_count,
        timeout_minutes=sandbox_timeout_minutes,
        timeout_per_command_seconds=sandbox_timeout_per_command_seconds,
        sandbox_client_max_workers=sandbox_client_max_workers,
        **kwargs,
    )
