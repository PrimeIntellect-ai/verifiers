import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

os.environ.setdefault(
    "TAU2_DATA_DIR", str(Path.home() / ".cache" / "tau2-bench-v1" / "data")
)

import verifiers.v1 as vf
from tau2.data_model.message import (
    AssistantMessage,
    Message,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Description as TauDescription
from tau2.data_model.tasks import Task as TauTask
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import load_tasks, run_task
from tau2.user.base import UserState
from tau2.utils import llm_utils
from tau2.utils.utils import DATA_DIR
from verifiers.utils.client_utils import load_prime_config

TAU2_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU2_REVISION = "337326e62d8e0ca74c353b004a9c5d748e0ba914"
# Tau2's workflow variant uses the Telecom tasks with its procedural support policy.
Tau2Domain = Literal["airline", "retail", "telecom", "telecom-workflow"]
_RESULT_PREFIX = "__TAU2_RESULT__="
_RUN_CONFIG = "TAU2_RUN_CONFIG"


class Tau2TasksetConfig(vf.TasksetConfig):
    domain: Tau2Domain = "telecom"


class Tau2Task(vf.Task, TauTask):
    domain: Tau2Domain
    tau_description: TauDescription | None = None


_tau_to_litellm_messages = llm_utils.to_litellm_messages
_tau_flip_roles = UserState.flip_roles


def _to_litellm_messages(messages: list[Message]) -> list[dict]:
    converted = _tau_to_litellm_messages(messages)
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage) and message.raw_data:
            converted[index] = message.raw_data["message"]
    return converted


def _flip_roles(self: UserState):
    flipped = _tau_flip_roles(self)
    for message, flipped_message in zip(self.messages, flipped, strict=True):
        if isinstance(flipped_message, AssistantMessage):
            flipped_message.raw_data = message.raw_data
    return flipped


class Tau2Taskset(vf.Taskset[Tau2Task, Tau2TasksetConfig]):
    def load_tasks(self) -> list[Tau2Task]:
        data_domain = (
            "telecom"
            if self.config.domain == "telecom-workflow"
            else self.config.domain
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        marker = DATA_DIR / ".tau2_revision"
        with (DATA_DIR / ".tau2_bootstrap.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not (
                (DATA_DIR / "tau2" / "domains" / data_domain).exists()
                and marker.exists()
                and marker.read_text() == TAU2_REVISION
            ):
                with tempfile.TemporaryDirectory(prefix="tau2_bench_v1_") as temp_dir:
                    subprocess.run(
                        ["git", "init", temp_dir],
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            temp_dir,
                            "fetch",
                            "--depth",
                            "1",
                            TAU2_REPOSITORY,
                            TAU2_REVISION,
                        ],
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            temp_dir,
                            "checkout",
                            "FETCH_HEAD",
                            "--",
                            "data",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    shutil.copytree(
                        Path(temp_dir) / "data", DATA_DIR, dirs_exist_ok=True
                    )
                    marker.write_text(TAU2_REVISION)

        tasks = load_tasks(
            task_set_name=self.config.domain,
            task_split_name="base",
        )
        return [
            Tau2Task(
                **task.model_dump(exclude={"description"}),
                idx=index,
                name=task.id,
                description=str(task.description) if task.description else None,
                prompt=DEFAULT_FIRST_AGENT_MESSAGE.content or "",
                domain=self.config.domain,
                tau_description=task.description,
            )
            for index, task in enumerate(tasks)
        ]

    @vf.reward
    async def tau2_reward(self, trace: vf.Trace) -> float:
        simulation = SimulationRun.model_validate(trace.info["tau2"]["simulation"])
        reward = simulation.reward_info
        assert reward is not None
        trace.info["tau2"]["evaluation"] = reward.model_dump(mode="json")
        return float(reward.reward)


class Tau2HarnessConfig(vf.HarnessConfig):
    id: Literal["tau2-bench-v1"] = "tau2-bench-v1"
    runtime: vf.SubprocessConfig = vf.SubprocessConfig()


class Tau2Harness(vf.Harness[Tau2HarnessConfig]):
    SUPPORTS_TASK_TOOLS = False

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace[Tau2Task],
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        del mcp_urls
        task = trace.task
        model = ctx.model.rsplit("/", 1)[-1]
        llm = f"openai/{ctx.model.removeprefix('openai/')}"
        llm_args: dict[str, object] = {
            "api_base": endpoint,
            "api_key": secret,
            "timeout": None,
        }
        if model.startswith("gpt-"):
            llm = f"openai/responses/{model}"
            llm_args["include"] = ["reasoning.encrypted_content"]
        prime = load_prime_config()
        user_args: dict[str, object] = {
            "temperature": 0,
            "api_base": prime.get("inference_url", "https://api.pinference.ai/api/v1"),
            "api_key": os.getenv("PRIME_API_KEY") or prime.get("api_key"),
            "timeout": 86400,
        }
        team_id = os.getenv("PRIME_TEAM_ID") or prime.get("team_id")
        if team_id:
            user_args["extra_headers"] = {"X-Prime-Team-ID": team_id}
        run_config = {
            "domain": task.domain,
            "task": {
                **task.model_dump(
                    mode="json",
                    include=set(TauTask.model_fields) - {"description"},
                ),
                "description": task.tau_description.model_dump(mode="json")
                if task.tau_description
                else None,
            },
            "agent_llm": llm,
            "agent_llm_args": llm_args,
            "user_llm_args": user_args,
        }
        result = await runtime.run_program(
            [sys.executable, "-m", __name__],
            {**self.config.env, _RUN_CONFIG: json.dumps(run_config)},
        )
        if result.exit_code != 0:
            return result
        simulation = SimulationRun.model_validate_json(
            next(
                line.removeprefix(_RESULT_PREFIX)
                for line in reversed(result.stdout.splitlines())
                if line.startswith(_RESULT_PREFIX)
            )
        )
        simulation.id = trace.id
        trace.info["tau2"] = {"simulation": simulation.model_dump(mode="json")}
        trace.stop(
            "user_completed"
            if simulation.termination_reason == TerminationReason.USER_STOP
            else f"tau2_{simulation.termination_reason.value}"
        )
        return result


__all__ = ["Tau2Harness", "Tau2Taskset"]


if __name__ == "__main__":
    # Tau stores LiteLLM's response metadata on generated messages. Reuse those messages
    # when it rebuilds model history so encrypted reasoning survives across turns.
    setattr(llm_utils, "to_litellm_messages", _to_litellm_messages)
    setattr(UserState, "flip_roles", _flip_roles)

    config = json.loads(os.environ[_RUN_CONFIG])
    simulation = run_task(
        domain=config["domain"],
        task=TauTask.model_validate(config["task"]),
        agent="llm_agent",
        user="user_simulator",
        llm_agent=config["agent_llm"],
        llm_args_agent=config["agent_llm_args"],
        llm_user="openai/openai/gpt-4.1",
        llm_args_user=config["user_llm_args"],
        max_steps=500,  # Let long scenarios finish while still bounding loops.
    )
    print(f"{_RESULT_PREFIX}{simulation.model_dump_json()}")
