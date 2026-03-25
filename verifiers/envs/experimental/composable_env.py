"""ComposableEnv — orchestration layer that wires Task + Agent.

Inherits from ``Environment`` (for evaluate / generate / dataset / rubric
/ state management) and ``SandboxMixin`` (for sandbox lifecycle).
Implements its own ``rollout()`` — no ``@final`` constraint because we
bypass ``MultiTurnEnv`` entirely.

Typical usage::

    task = MySweTaskAdapter(R2EGymTask())
    agent = LLMAgent(tools=[bash, str_replace])
    env = ComposableEnv(task=task, agent=agent, dataset=task.get_dataset())

The rollout flow is:

1. ``init_state()`` — standard Environment state setup
2. Create sandbox with Task's docker image
3. ``task.setup()`` — prepare sandbox (repo checkout, venv, etc.)
4. ``agent.setup()`` — lightweight agent install (upload scripts, etc.)
5. ``agent.run()`` — the agent loop (prompt → tools → repeat)
6. ``task.evaluate()`` — score the result
7. Cleanup sandbox
"""

from __future__ import annotations

import logging
import math
import time
import uuid

from prime_sandboxes import CreateSandboxRequest

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.experimental.agent import Agent
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin, SandboxMonitorRubric
from verifiers.envs.experimental.task import Task
from verifiers.types import (
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
)
from verifiers.utils.message_utils import concat_messages

logger = logging.getLogger(__name__)


class ComposableRubric(SandboxMonitorRubric):
    """Rubric that reads the reward pre-computed by Task.evaluate().

    Extends ``SandboxMonitorRubric`` so sandbox OOM/timeout metrics are
    included in the same rubric (avoiding the RubricGroup assert issue
    where metric-only rubrics have no reward funcs).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.task_reward)

    async def task_reward(self, state: State) -> float:
        """Return the reward computed by the Task during rollout."""
        return float(state.get("reward") or 0.0)


class ComposableEnv(SandboxMixin, vf.Environment):
    """Environment that composes a Task (WHAT) with an Agent (HOW).

    Parameters
    ----------
    task:
        Provides dataset, docker image, sandbox setup, and evaluation.
    agent:
        Drives the solving loop (tools, LLM calls, context management).
    docker_image_override:
        If set, used instead of ``task.get_image()`` for all instances.
        Useful for pre-composed images (task + agent baked together).
    cpu_cores, memory_gb, disk_size_gb, gpu_count:
        Sandbox resource limits.
    timeout_seconds:
        Maximum wall-clock time per rollout.
    test_timeout:
        Timeout (seconds) passed to ``task.evaluate()``.
    """

    def __init__(
        self,
        task: Task,
        agent: Agent,
        *,
        docker_image_override: str | None = None,
        cpu_cores: int = 4,
        memory_gb: int = 8,
        disk_size_gb: int = 32,
        gpu_count: int = 0,
        timeout_seconds: float = 3600.0,
        test_timeout: int = 300,
        environment_vars: dict[str, str] | None = None,
        labels: list[str] | None = None,
        # SandboxMixin params
        max_retries: int = 5,
        sandbox_creations_per_minute: float | None = 128,
        # Environment params
        **kwargs,
    ):
        # Auto-extract dataset from TaskSet if not provided
        if "dataset" not in kwargs and hasattr(task, "get_dataset"):
            kwargs["dataset"] = task.get_dataset()
        # Inject ComposableRubric as the primary rubric unless user provided one
        if "rubric" not in kwargs:
            kwargs["rubric"] = ComposableRubric()
        super().__init__(**kwargs)

        self.task = task
        self.agent = agent
        self.docker_image_override = docker_image_override
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_seconds = timeout_seconds
        self.test_timeout = test_timeout
        self.environment_vars = environment_vars or {}
        self.labels = labels or []

        self.init_sandbox_client(
            max_retries=max_retries,
            sandbox_creations_per_minute=sandbox_creations_per_minute,
        )

    # -- rollout ------------------------------------------------------------

    @property
    def _needs_sandbox(self) -> bool:
        """Whether this rollout requires a sandbox."""
        task_needs = getattr(self.task, "needs_sandbox", True)
        agent_needs = getattr(self.agent, "needs_sandbox", True)
        return task_needs or agent_needs

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        if self._needs_sandbox:
            return await self._rollout_with_sandbox(input, client, model, sampling_args)
        else:
            return await self._rollout_without_sandbox(input, client, model, sampling_args)

    async def _rollout_without_sandbox(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Lightweight rollout: prompt → agent → evaluate.  No sandbox."""
        state = await self.init_state(input, client, model, sampling_args)
        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        try:
            prompt = self._get_agent_prompt(state)
            steps = await self.agent.run(prompt, state)

            state["trajectory"] = steps
            self._render_completion(state)

            try:
                reward = await self.task.evaluate(None, "", state)
                if isinstance(reward, dict):
                    state["role_rewards"] = reward
                    state["reward"] = sum(reward.values()) / len(reward) if reward else 0.0
                else:
                    state["reward"] = reward
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {rollout_id}: {e}")
                state["reward"] = 0.0

        except vf.Error as e:
            state["error"] = e
        except Exception as e:
            state["error"] = vf.InfraError(str(e))
            self.logger.error(f"Rollout {rollout_id} failed: {e}")
        finally:
            duration_s = time.time() - state["timing"]["start_time"]
            num_turns = len(state.get("trajectory", []))
            self.logger.info(
                f"Finished rollout_id={rollout_id} | "
                f"example_id={state.get('example_id')} | "
                f"turns={num_turns} | "
                f"reward={state.get('reward')} | "
                f"duration={duration_s:.1f}s"
            )
            state["is_completed"] = True
            end_time = time.time()
            start_time = state["timing"]["start_time"]
            state["timing"]["generation_ms"] = (end_time - start_time) * 1000
            state["timing"]["total_ms"] = (end_time - start_time) * 1000

        return state

    async def _rollout_with_sandbox(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Full rollout with sandbox lifecycle."""
        state = await self.init_state(input, client, model, sampling_args)
        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id
        state["_run_background_job"] = self.run_background_job

        try:
            info = state.get("info") or {}
            image = self.docker_image_override or self._resolve_image(info)

            env_vars = dict(self.environment_vars)
            env_vars.update(self.task.get_env_vars())
            if hasattr(self.agent, "get_env_vars"):
                env_vars.update(self.agent.get_env_vars(state))

            request = CreateSandboxRequest(
                name=rollout_id,
                docker_image=image,
                cpu_cores=self.cpu_cores,
                memory_gb=self.memory_gb,
                disk_size_gb=self.disk_size_gb,
                gpu_count=self.gpu_count,
                timeout_minutes=max(1, math.ceil(self.timeout_seconds / 60)),
                environment_vars=env_vars,
                labels=self.labels,
            )
            await self.create_sandbox(state, request)
            sandbox_id = state["sandbox_id"]

            self.logger.info(
                f"Started rollout_id={rollout_id} | example_id={state.get('example_id')} | image={image}"
            )

            await self.task.setup(self.sandbox_client, sandbox_id, state)
            await self.agent.setup(self.sandbox_client, sandbox_id, state)

            if hasattr(self.agent, "inject_tool_args"):
                self.agent.inject_tool_args(
                    state=state,
                    sandbox_client=self.sandbox_client,
                    sandbox_id=sandbox_id,
                )

            # Inject task-provided extra tools into ReActAgent
            if hasattr(self.agent, "add_tool") and hasattr(self.task, "get_extra_tools"):
                for tool_entry in self.task.get_extra_tools() or []:
                    if isinstance(tool_entry, tuple):
                        func, skip = tool_entry
                        self.agent.add_tool(func, args_to_skip=skip)
                    else:
                        self.agent.add_tool(tool_entry)

            prompt = self._get_agent_prompt(state)
            steps = await self.agent.run(prompt, state)

            state["trajectory"] = steps
            self._render_completion(state)

            try:
                reward = await self.task.evaluate(
                    self.sandbox_client, sandbox_id, state
                )
                if isinstance(reward, dict):
                    state["role_rewards"] = reward
                    state["reward"] = sum(reward.values()) / len(reward) if reward else 0.0
                else:
                    state["reward"] = reward
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {rollout_id}: {e}")
                state["reward"] = 0.0

        except vf.Error as e:
            state["error"] = e
        except Exception as e:
            state["error"] = vf.InfraError(str(e))
            self.logger.error(f"Rollout {rollout_id} failed: {e}")
        finally:
            duration_s = time.time() - state["timing"]["start_time"]
            num_turns = len(state.get("trajectory", []))
            self.logger.info(
                f"Finished rollout_id={rollout_id} | "
                f"example_id={state.get('example_id')} | "
                f"turns={num_turns} | "
                f"reward={state.get('reward')} | "
                f"duration={duration_s:.1f}s"
            )

            sandbox_id = state.get("sandbox_id")
            if sandbox_id:
                await self.delete_sandbox(sandbox_id)

            state["is_completed"] = True
            end_time = time.time()
            start_time = state["timing"]["start_time"]
            state["timing"]["generation_ms"] = (end_time - start_time) * 1000
            state["timing"]["total_ms"] = (end_time - start_time) * 1000

        return state

    # -- helpers ------------------------------------------------------------

    def _resolve_image(self, info: dict) -> str:
        """Resolve the Docker image: task → agent default → fallback."""
        task_needs = getattr(self.task, "needs_sandbox", True)
        if task_needs:
            try:
                return self.task.get_image(info)
            except Exception:
                pass
        # Agent's default image (e.g. BinaryAgent.default_image)
        agent_image = getattr(self.agent, "default_image", None)
        if agent_image:
            return agent_image
        return "python:3.11-slim"

    def _get_agent_prompt(self, state: State) -> Messages:
        """Build the prompt the agent will see.

        By default uses the Task's ``get_prompt()`` with the example info.
        Override in subclasses (e.g. ``UserSimEnv``) to transform or
        withhold the full prompt from the solving agent.
        """
        info = state.get("info") or {}
        try:
            return self.task.get_prompt(info)
        except Exception:
            # Fallback to dataset-provided prompt
            return state["prompt"]

    def _render_completion(self, state: State) -> None:
        """Build ``state["completion"]`` from the trajectory steps."""
        trajectory = state.get("trajectory", [])
        if not trajectory:
            state["completion"] = []
            return

        all_completions: list[Messages] = []
        for step in trajectory:
            all_completions.append(step.get("completion", []))
        state["completion"] = concat_messages(all_completions)
