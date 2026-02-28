"""
Multi-agent environment with turn order management.

Two modes:
    TaskSet mode:  MultiAgentEnv(task=my_task, agents={"a": agent_a, "b": agent_b})
    Subclass mode: class MyEnv(MultiAgentEnv) — override hooks directly, no TaskSet

TaskSet mode: Game logic lives in the TaskSet. Agents define who responds and how.
Subclass mode: Worker envs override build_actor_prompt/on_turn_complete directly.
              Agents are injected by Registry via inject_agents().

Both modes: MultiAgentEnv runs the loop: pick actor → build prompt → get response → repeat.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.clients import Client, OpenAIChatCompletionsClient, resolve_client
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import (
    ClientConfig,
    Messages,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.async_utils import maybe_retry
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.save_utils import state_to_output


class MultiAgentEnv(MultiTurnEnv):
    """
    Multi-agent environment.

    TaskSet mode:  MultiAgentEnv(task=my_task, agents={"a": agent_a})
    Subclass mode: class MyEnv(MultiAgentEnv) — set self.actors/self.name,
                   override hooks, agents injected by Registry.
    """

    def __init__(
        self,
        task: Any | None = None,
        agents: dict[str, Any] | list[Any] | None = None,
        **kwargs,
    ):
        self._task = task

        # Normalize agents to dict (or empty for worker envs)
        if agents is not None:
            if isinstance(agents, dict):
                self._agents = agents
            else:
                self._agents = {a.id: a for a in agents}
        else:
            self._agents = {}

        # TaskSet mode: wire dataset/rubric/roles
        if task is not None:
            self.actors = list(self._agents.keys())

            for role in task.roles:
                if role not in self._agents:
                    raise ValueError(
                        f"Task role '{role}' has no agent. "
                        f"Available agents: {list(self._agents.keys())}"
                    )

            if "dataset" not in kwargs and "eval_dataset" not in kwargs:
                kwargs["dataset"] = task.get_examples()
            if "rubric" not in kwargs and task.rubric is not None:
                kwargs["rubric"] = task.rubric
            self.name = task.name

        # Worker envs: dummy dataset to satisfy Environment base class
        if "dataset" not in kwargs and "eval_dataset" not in kwargs:
            kwargs["dataset"] = Dataset.from_dict({
                "prompt": [[{"role": "user", "content": ""}]],
                "answer": [""],
                "example_id": [0],
                "task": ["dummy"],
            })

        super().__init__(**kwargs)

        # Auto-register actors with MultiAgentRubric for training support
        self._register_actors_with_rubric()

    # -------------------------------------------------------------------------
    # Actor / Agent Lookup
    # -------------------------------------------------------------------------

    def get_actor(self, actor_id: str) -> Any:
        """Get an agent by ID."""
        if actor_id in self._agents:
            return self._agents[actor_id]
        raise KeyError(
            f"Agent '{actor_id}' not found. "
            f"Available: {list(self._agents.keys())}"
        )

    def inject_agents(self, agents: dict[str, Any]) -> None:
        """Add agents from Registry. Doesn't overwrite existing."""
        for aid, agent in agents.items():
            if aid not in self._agents:
                self._agents[aid] = agent
        self._register_actors_with_rubric()

    def _register_actors_with_rubric(self) -> None:
        """Register all agents with MultiAgentRubric for training support."""
        from verifiers.rubrics.multiagent_rubric import MultiAgentRubric

        if hasattr(self, "rubric") and isinstance(self.rubric, MultiAgentRubric):
            for agent_id, agent in self._agents.items():
                self.rubric.register_actor(
                    agent_id, is_trainable=agent.is_trainable
                )

    # -------------------------------------------------------------------------
    # Turn Management (delegates to TaskSet, or subclass overrides)
    # -------------------------------------------------------------------------

    def get_initial_actor(self, state: State) -> str:
        if self._task is not None:
            return self._task.get_initial_role(state)
        raise NotImplementedError("Subclass must override get_initial_actor()")

    def get_next_actor(self, state: State) -> str:
        if self._task is not None:
            return self._task.get_next_role(state)
        raise NotImplementedError("Subclass must override get_next_actor()")

    def get_active_actors(self, state: State) -> list[str]:
        if self._task is not None:
            return self._task.get_active_roles(state)
        current = state["extras"].get("current_actor_id")
        if current is None:
            return [self.get_initial_actor(state)]
        return [self.get_next_actor(state)]

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)

        state["child_states"] = []
        state["extras"] = {
            "current_actor_id": None,
            "actor_history": [],
            "episode_id": state.get("trajectory_id", uuid.uuid4().hex),
            "parent_episode_id": None,
        }

        # Make registry accessible from state (for TaskSet spawn access)
        if hasattr(self, "registry"):
            state["registry"] = self.registry

        if self._task is not None:
            state = await self._task.setup_state(state)
        return state

    # -------------------------------------------------------------------------
    # Game Hooks (delegates to TaskSet)
    # -------------------------------------------------------------------------

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        if self._task is not None:
            return await self._task.build_prompt(actor_id, state)
        raise NotImplementedError("Subclass must override build_actor_prompt()")

    async def on_turn_complete(self, state: State) -> None:
        if self._task is not None:
            return await self._task.on_turn_complete(state)

    async def on_game_end(self, state: State) -> None:
        if self._task is not None:
            return await self._task.on_game_end(state)

    # -------------------------------------------------------------------------
    # Stop Condition
    # -------------------------------------------------------------------------

    @vf.stop
    async def task_stopped(self, state: State) -> bool:
        if self._task is not None:
            return await self._task.should_stop(state)
        return False  # Subclass uses @vf.stop decorator on its own methods

    # -------------------------------------------------------------------------
    # Tool Support (env_response + tool call detection)
    # -------------------------------------------------------------------------

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Execute tool calls from the last assistant message.

        Default returns [] (no tools). Override in subclass to add tool execution.
        Called by the tool loop in rollout() when the model makes tool calls.
        """
        return []

    def _last_completion_has_tool_calls(self, state: State) -> bool:
        """Check if the last trajectory step's completion includes tool calls."""
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return False
        last_completion = trajectory[-1].get("completion", [])
        if not last_completion:
            return False
        last_msg = last_completion[-1]
        return (
            hasattr(last_msg, "tool_calls")
            and last_msg.tool_calls is not None
            and len(last_msg.tool_calls) > 0
        )

    # -------------------------------------------------------------------------
    # Trajectory Management
    # -------------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        current_actor_id = state["extras"]["current_actor_id"]
        if current_actor_id:
            trajectory_step["extras"]["actor_id"] = current_actor_id
            turn_index = len(state["trajectory"])
            state["extras"]["actor_history"].append((current_actor_id, turn_index))
        await super().add_trajectory_step(state, trajectory_step)

    # -------------------------------------------------------------------------
    # Main Rollout Loop
    # -------------------------------------------------------------------------

    async def rollout(
        self,
        input,
        client,
        model,
        sampling_args=None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        while not await self.is_completed(state):
            active_actors = self.get_active_actors(state)

            for actor_id in active_actors:
                state["extras"]["current_actor_id"] = actor_id

                try:
                    prompt_messages = await self.build_actor_prompt(actor_id, state)

                    actor = self.get_actor(actor_id)
                    merged_args = actor.merge_sampling_args(sampling_args or {})

                    # Inject agent's system_prompt
                    if actor.system_prompt:
                        prompt_messages = [
                            {"role": "system", "content": actor.system_prompt},
                            *prompt_messages,
                        ]

                    # Wrap raw AsyncOpenAI in Client if needed
                    actor_client = actor.client
                    if actor_client is not None and not isinstance(actor_client, Client):
                        actor_client = OpenAIChatCompletionsClient(actor_client)

                    used_model = actor.model or state.get("model", "default")
                    self.logger.info(f"[{actor_id}] using model: {used_model}")

                    response = await self.get_model_response(
                        state,
                        prompt_messages,
                        client=actor_client,
                        model=actor.model,
                        sampling_args=merged_args,
                    )

                    await self.add_model_response(state, prompt_messages, response)

                    # Tool loop: same actor continues until no more tool calls.
                    # env_response() returns [] by default (no tools).
                    # Subclasses override env_response() to execute tools.
                    while self._last_completion_has_tool_calls(state):
                        last_step = state["trajectory"][-1]
                        messages = list(last_step["prompt"]) + list(last_step["completion"])
                        tool_results = await self.env_response(messages, state)
                        if not tool_results:
                            break
                        prompt_messages = messages + list(tool_results)
                        response = await self.get_model_response(
                            state,
                            prompt_messages,
                            client=actor_client,
                            model=actor.model,
                            sampling_args=merged_args,
                        )
                        await self.add_model_response(state, prompt_messages, response)

                    await self.on_turn_complete(state)

                except vf.OverlongPromptError:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                    break
                except vf.Error as e:
                    state["error"] = e
                    break

            if await self.is_completed(state):
                break

        await self.on_game_end(state)
        await self.render_completion(state)
        return state

    async def render_completion(self, state: State):
        """Build completion from all actors' turns, tagged with actor IDs."""
        if not state.get("trajectory"):
            state["completion"] = []
            return
        all_messages = []
        for step in state["trajectory"]:
            actor_id = step["extras"].get("actor_id", "")
            for msg in step.get("completion", []):
                tagged = dict(msg)
                if actor_id and tagged.get("content"):
                    tagged["content"] = f"[{actor_id}] {tagged['content']}"
                all_messages.append(tagged)
        state["completion"] = all_messages

    # -------------------------------------------------------------------------
    # Per-Actor State Creation
    # -------------------------------------------------------------------------

    SHARED_STATE_FIELDS = {
        "client",
        "model",
        "trajectory_id",
        "sampling_args",
    }

    def create_actor_state(
        self,
        parent_state: State,
        actor_id: str,
        actor_trajectory: list[TrajectoryStep],
    ) -> State:
        actor_state = State()

        for key in parent_state.keys():
            if key in self.SHARED_STATE_FIELDS:
                actor_state[key] = parent_state[key]

        if "timing" in parent_state:
            actor_state["timing"] = dict(parent_state["timing"])

        actor_state["answer"] = parent_state.get("answer", "")
        actor_state["task"] = parent_state.get("task", "")
        actor_state["example_id"] = parent_state.get("example_id", 0)
        actor_state["info"] = parent_state.get("info", {})

        actor_state["trajectory"] = actor_trajectory

        actor_state["extras"] = {
            **parent_state.get("extras", {}),
            "current_actor_id": actor_id,
        }

        actor_state["child_states"] = []
        actor_state["reward"] = None
        actor_state["advantage"] = None
        actor_state["metrics"] = None

        actor = self.get_actor(actor_id)
        actor_state["is_trainable"] = actor.is_trainable

        if actor_trajectory:
            raw_prompt = actor_trajectory[0].get("prompt", [])
            prompt_ref = raw_prompt
            for i in range(len(raw_prompt) - 1, -1, -1):
                if raw_prompt[i].get("role") == "system":
                    prompt_ref = raw_prompt[i:]
                    break
            actor_state["prompt"] = prompt_ref

            all_completions = []
            for step in actor_trajectory:
                step_completion = step.get("completion", [])
                all_completions.extend(step_completion)
            actor_state["completion"] = all_completions
        else:
            actor_state["prompt"] = parent_state.get("prompt", [])
            actor_state["completion"] = []

        return actor_state

    def create_actor_states(self, state: State, actor_ids: list[str] | None = None) -> list[State]:
        if actor_ids is None:
            actor_ids = self.actors

        actor_states = []
        for actor_id in actor_ids:
            actor_trajectory = [
                step for step in state.get("trajectory", [])
                if step.get("extras", {}).get("actor_id") == actor_id
            ]
            new_state = self.create_actor_state(state, actor_id, actor_trajectory)
            actor_states.append(new_state)

        return actor_states

    # -------------------------------------------------------------------------
    # Training Support
    # -------------------------------------------------------------------------

    @property
    def outputs_per_input(self) -> int:
        """Number of RolloutOutputs produced per rollout input.

        For multi-agent: one output per trainable actor per game.
        Prime-rl reads this to determine how many game inputs to create.
        """
        trainable = [a for a in self._agents.values() if a.is_trainable]
        return len(trainable) or 1

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client=None,
        **kwargs,
    ) -> list[RolloutOutput]:
        """Run a group of rollouts, splitting into per-actor outputs.

        Each game is split into per-actor states (trainable only) before
        scoring, so the rubric sees correct actor IDs and each actor gets
        its own reward. Non-trainable actors are excluded entirely.
        """
        # Server mode: delegate (server has the same override)
        env_client = env_client or getattr(self, "env_client", None)
        if env_client is not None:
            resolved_config = (
                resolve_client_config(client)
                if isinstance(client, ClientConfig)
                else None
            )
            if resolved_config is None:
                raise ValueError(
                    f"client must be ClientConfig in server mode, got {type(client)}"
                )
            return await env_client.run_group(
                group_inputs,
                resolved_config,
                model,
                sampling_args,
                max_retries,
                state_columns,
            )

        resolved_client = resolve_client(client)
        state_columns = list(state_columns or [])

        trainable_ids = [
            aid for aid, a in self._agents.items() if a.is_trainable
        ]
        num_trainable = len(trainable_ids) or 1

        # Run fewer games when multiple actors are trainable,
        # so total outputs = len(group_inputs)
        games_count = max(1, len(group_inputs) // num_trainable)
        game_inputs = group_inputs[:games_count]

        async def attempt() -> list[State]:
            game_states = await asyncio.gather(*[
                self.rollout(inp, resolved_client, model, sampling_args)
                for inp in game_inputs
            ])
            # Split each game into per-actor states (trainable only)
            actor_states = []
            for state in game_states:
                for astate in self.create_actor_states(
                    state, actor_ids=trainable_ids
                ):
                    for step in astate.get("trajectory", []):
                        step["extras"]["game_stop_condition"] = state.get("stop_condition")
                        step["extras"]["game_error"] = str(state.get("error") or "")
                    actor_states.append(astate)

            if self.score_rollouts:
                await self.rubric.score_group(actor_states)
            else:
                await self.rubric.dummy_score_group(actor_states)
            return actor_states

        actor_states = await maybe_retry(
            attempt, max_retries=max_retries
        )()
        return [
            state_to_output(s, state_columns) for s in actor_states
        ]

    # -------------------------------------------------------------------------
    # Result Building (for generate/eval)
    # -------------------------------------------------------------------------

    def _prepare_rollout_results(
        self,
        all_states: list[State],
        model: str,
        client: AsyncOpenAI,
        state_columns: list[str] | None,
        results_path,
        gen_sampling_args: SamplingArgs,
        start_time: float,
    ):
        result = super()._prepare_rollout_results(
            all_states, model, client, state_columns,
            results_path, gen_sampling_args, start_time
        )
        result["actor_id"] = [
            s.get("extras", {}).get("current_actor_id", "unknown")
            for s in all_states
        ]
        return result
