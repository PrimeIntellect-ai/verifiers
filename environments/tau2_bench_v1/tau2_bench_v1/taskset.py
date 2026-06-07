import os
import shutil
import subprocess
from pathlib import Path

from pydantic import TypeAdapter
import verifiers.v1 as vf

from .servers.user import UserConfig

DEFAULT_USER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_USER_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_USER_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_ERRORS = 10


class Tau2TasksetConfig(vf.TasksetConfig):
    id: str | None = "tau2_telecom"
    user: vf.UserConfig | None = UserConfig()
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
    tau2_user: vf.JsonData


class Tau2Taskset(vf.Taskset[Tau2TasksetConfig]):
    task_type = Tau2Task

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return list(
            load_tasks(
                domain=self.config.domain,
                max_turns=self.config.max_turns,
                user_model=self.config.user_model,
                user_args=self.config.user_args or {},
                user_base_url=self.config.user_base_url,
                user_api_key_var=self.config.user_api_key_var,
                max_errors=self.config.max_errors,
            )
        )

    @vf.reward(weight=1.0)
    async def tau2_reward(self, task: Tau2Task, state: vf.State) -> float:
        tau2 = state.extras.get("tau2")
        if not isinstance(tau2, dict):
            return 0.0
        messages_data = tau2.get("messages")
        if not isinstance(messages_data, list):
            return 0.0
        from tau2.data_model.message import Message as TauMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.data_model.tasks import Task as TauTask
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.utils.utils import get_now

        messages = TypeAdapter(list[TauMessage]).validate_python(messages_data)
        now = get_now()
        simulation = SimulationRun(
            id=state.id,
            task_id=task.task_id or task.tau2_task["id"],
            start_time=now,
            end_time=now,
            duration=state.timing.total,
            termination_reason=TerminationReason.USER_STOP,
            messages=messages,
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=TauTask.model_validate(task.tau2_task),
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=task.domain,
        )
        tau2["reward"] = float(reward_info.reward)
        tau2["reward_info"] = reward_info.model_dump(mode="json", exclude_none=True)
        return float(reward_info.reward)

    @vf.metric
    async def tau2_num_steps(self, state: vf.State) -> float:
        tau2 = state.extras.get("tau2")
        return float(tau2.get("step_count", 0.0) if isinstance(tau2, dict) else 0.0)

    @vf.metric
    async def tau2_num_errors(self, state: vf.State) -> float:
        tau2 = state.extras.get("tau2")
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


def load_tasks(
    domain: str,
    max_turns: int,
    user_model: str,
    user_args: vf.JsonData,
    user_base_url: str,
    user_api_key_var: str,
    max_errors: int,
):
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
            "tau2_user": {
                "model": user_model,
                "args": user_args,
                "base_url": user_base_url,
                "api_key_var": user_api_key_var,
                "max_errors": max_errors,
            },
        }


def load_taskset(config: Tau2TasksetConfig) -> Tau2Taskset:
    return Tau2Taskset(config=config)
