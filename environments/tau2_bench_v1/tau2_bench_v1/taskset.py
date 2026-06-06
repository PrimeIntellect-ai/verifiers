import os
import shutil
import subprocess
import sys
from pathlib import Path

import verifiers.v1 as vf

DEFAULT_USER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_USER_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_USER_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_ERRORS = 10


class Tau2TasksetConfig(vf.TasksetConfig):
    id: str | None = "tau2_telecom"
    user: vf.User | None = vf.User(
        server=vf.MCPServerSpec(
            command=[sys.executable, "-m", "tau2_bench_v1.servers.user"]
        )
    )
    domain: str = "telecom"
    user_model: str = DEFAULT_USER_MODEL
    user_args: vf.JsonData | None = None
    user_base_url: str = DEFAULT_USER_BASE_URL
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR
    max_steps: int = DEFAULT_MAX_STEPS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_turns: int = DEFAULT_MAX_STEPS


class Tau2Task(vf.Task):
    domain: str
    system_prompt: str
    tau2_task: vf.JsonData


class Tau2Taskset(vf.Taskset[Tau2TasksetConfig]):
    task_type = Tau2Task

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return list(
            load_tasks(domain=self.config.domain, max_turns=self.config.max_turns)
        )

    @vf.reward(weight=1.0)
    async def tau2_reward(self, state: vf.State) -> float:
        tau2 = state.scratch.get("tau2")
        if not isinstance(tau2, dict):
            return 0.0
        return float(tau2.get("reward", 0.0) or 0.0)

    @vf.metric
    async def tau2_num_steps(self, state: vf.State) -> float:
        tau2 = state.scratch.get("tau2")
        return float(tau2.get("step_count", 0.0) if isinstance(tau2, dict) else 0.0)

    @vf.metric
    async def tau2_num_errors(self, state: vf.State) -> float:
        tau2 = state.scratch.get("tau2")
        return float(tau2.get("num_errors", 0.0) if isinstance(tau2, dict) else 0.0)


def download_tau2_data() -> None:
    from tau2.utils.utils import DATA_DIR

    data_dir = Path(DATA_DIR)
    if os.path.exists(data_dir) and os.path.exists(data_dir / "tau2" / "domains"):
        return
    os.makedirs(data_dir, exist_ok=True)
    temp_dir = Path("/tmp/tau2_bench_v1")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/sierra-research/tau2-bench.git",
                str(temp_dir),
            ],
            check=True,
            capture_output=True,
        )
        source_data = temp_dir / "data"
        if source_data.exists():
            shutil.copytree(source_data, data_dir, dirs_exist_ok=True)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def load_tasks(domain: str, max_turns: int):
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    from tau2.registry import registry
    from tau2.run import load_tasks as load_tau2_tasks

    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    system_prompt = SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=environment.policy,
    )
    for index, task in enumerate(
        load_tau2_tasks(task_set_name=domain, task_split_name="base")
    ):
        yield {
            "row_id": index,
            "task_id": task.id,
            "domain": domain,
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "prompt": [],
            "tau2_task": task.model_dump(mode="json", exclude_none=True),
        }


def load_taskset(config: Tau2TasksetConfig) -> Tau2Taskset:
    return Tau2Taskset(config=config)
