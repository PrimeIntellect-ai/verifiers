"""UserSimEnv — multi-agent environment with a simulated user.

Two agents collaborate on a task:

* **User agent** (frozen) — has the full problem context, responds to
  questions in character according to a persona.
* **Developer agent** (trainable) — has sandbox access and tools, but does
  NOT see the full problem statement.  Must use ``ask_user`` to extract
  information from the user agent.

The developer agent can be *any* ``Agent`` implementation (``ReActAgent``,
``BinaryAgent``, etc.) — the ``ask_user`` tool is injected at rollout time.

Usage::

    task = SweTaskAdapter(R2EGymTask())
    developer = ReActAgent(tools=[bash, str_replace], max_turns=200)
    env = UserSimEnv(
        task=task,
        developer_agent=developer,
        user_system_prompt="You are a non-technical user who reported a bug...",
        dataset=task.get_dataset(),
    )
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Any

from prime_sandboxes import CreateSandboxRequest

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.experimental.agent import ReActAgent  # noqa: F401 (used in docstring)
from verifiers.envs.experimental.composable_env import ComposableEnv
from verifiers.envs.experimental.task import Task
from verifiers.types import (
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
)
from verifiers.utils.response_utils import parse_response_message

logger = logging.getLogger(__name__)

DEFAULT_USER_SYSTEM_PROMPT = """\
You are a user who reported a software issue. You have the following context \
about the problem:

{problem_statement}

Guidelines:
- Answer the developer's questions based on your knowledge of the issue.
- Stay in character — you are the person who experienced this bug.
- Reveal information naturally when asked, but don't volunteer everything at once.
- If the developer asks about something you don't know, say so.
- Keep your responses concise and natural.
"""

DEFAULT_DEV_STARTER_PROMPT = """\
A user has reported a software issue in this repository. You need to \
diagnose and fix the problem.

Use the `ask_user` tool to ask the user questions about the issue they \
experienced. Use `execute_bash` and `edit_via_str_replace` to investigate \
the codebase and implement a fix.

Start by asking the user what problem they encountered.\
"""


class UserSimEnv(ComposableEnv):
    """Multi-agent environment: user simulator + developer agent.

    Parameters
    ----------
    task:
        The underlying task (SWE bug, Lean proof, etc.).
    developer_agent:
        The agent being trained/evaluated — gets tools + ``ask_user``.
    user_system_prompt:
        System prompt template for the user agent.  Use ``{problem_statement}``
        as a placeholder for the full task instruction.
    dev_starter_prompt:
        The initial message the developer agent sees (vague, no details).
    user_model:
        Model to use for the user agent.  If ``None``, uses the same model
        as the developer.
    max_user_turns:
        Maximum number of ``ask_user`` calls per rollout.
    """

    def __init__(
        self,
        task: Task,
        developer_agent: Any,  # any Agent implementation
        *,
        user_system_prompt: str = DEFAULT_USER_SYSTEM_PROMPT,
        dev_starter_prompt: str = DEFAULT_DEV_STARTER_PROMPT,
        user_model: str | None = None,
        max_user_turns: int = 20,
        **kwargs,
    ):
        # Pass developer_agent as the agent to ComposableEnv
        super().__init__(task=task, agent=developer_agent, **kwargs)
        self.developer_agent = developer_agent
        self.user_system_prompt_template = user_system_prompt
        self.dev_starter_prompt = dev_starter_prompt
        self.user_model = user_model
        self.max_user_turns = max_user_turns

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id
        state["_run_background_job"] = self.run_background_job
        state["user_interactions"] = []

        try:
            info = state.get("info") or {}
            image = self.docker_image_override or self.task.get_image(info)

            env_vars = dict(self.environment_vars)
            env_vars.update(self.task.get_env_vars())

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

            # Task setup
            await self.task.setup(self.sandbox_client, sandbox_id, state)

            async with self._agent_run_lock:
                # Agent setup
                await self.developer_agent.setup(self.sandbox_client, sandbox_id, state)

                # Build user agent with full problem context
                full_prompt = self.task.get_prompt(info)
                problem_text = ""
                for msg in full_prompt:
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    if content:
                        problem_text += str(content) + "\n"

                user_system_prompt = self.user_system_prompt_template.format(
                    problem_statement=problem_text.strip()
                )

                # Track user turns for this rollout
                user_turn_count = 0
                user_model = self.user_model or model

                # Create ask_user tool
                async def ask_user(question: str) -> str:
                    """Ask the user about the issue they reported. The user is a real person who experienced the bug — ask them questions to understand the problem."""
                    nonlocal user_turn_count
                    user_turn_count += 1

                    if user_turn_count > self.max_user_turns:
                        return "The user is no longer available. Please proceed with the information you have."

                    # Build user conversation
                    user_messages: Messages = [
                        {"role": "system", "content": user_system_prompt},  # type: ignore[list-item]
                        {"role": "user", "content": question},  # type: ignore[list-item]
                    ]

                    # Add previous interactions for context
                    for prev in state["user_interactions"]:
                        user_messages.insert(-1, {"role": "user", "content": prev["question"]})  # type: ignore[arg-type]
                        user_messages.insert(-1, {"role": "assistant", "content": prev["answer"]})  # type: ignore[arg-type]

                    # Call the user model
                    user_client = state["client"]
                    response = await user_client.get_response(
                        prompt=user_messages,
                        model=user_model,
                        tools=None,
                        sampling_args={},
                        state=state,
                    )
                    completion = await parse_response_message(response)
                    answer = ""
                    if completion:
                        content = completion[-1].content if hasattr(completion[-1], "content") else str(completion[-1])
                        answer = str(content) if content else ""

                    state["user_interactions"].append({
                        "question": question,
                        "answer": answer,
                        "turn": user_turn_count,
                    })

                    self.logger.debug(
                        f"ask_user turn={user_turn_count}: Q={question[:80]}... A={answer[:80]}..."
                    )
                    return answer

                # Inject ask_user into developer agent
                self.developer_agent.add_tool(ask_user)
                try:
                    # Inject hidden args for other tools
                    if hasattr(self.developer_agent, "inject_tool_args"):
                        self.developer_agent.inject_tool_args(
                            state=state,
                            sandbox_client=self.sandbox_client,
                            sandbox_id=sandbox_id,
                        )

                    # Run developer agent with vague prompt
                    dev_prompt: Messages = [
                        {"role": "user", "content": self.dev_starter_prompt},  # type: ignore[list-item]
                    ]
                    steps = await self.developer_agent.run(dev_prompt, state)
                finally:
                    # Always clean up ask_user so it does not leak across rollouts.
                    self.developer_agent.remove_tool("ask_user")

            state["trajectory"] = steps
            state["user_turn_count"] = user_turn_count
            self._render_completion(state)

            # Evaluate
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
            user_turns = state.get("user_turn_count", 0)
            self.logger.info(
                f"Finished rollout_id={rollout_id} | "
                f"example_id={state.get('example_id')} | "
                f"agent_turns={num_turns} | "
                f"user_turns={user_turns} | "
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
