import asyncio
import inspect
import logging
import time
from typing import Any, cast

import verifiers as vf
from verifiers.types import (
    GroupRewardFunc,
    MultiAgentRewardFunc,
    RewardFunc,
    RolloutScore,
    State,
)
from verifiers.utils.async_utils import maybe_await


class Rubric:
    """
    Rubric class for reward functions.

    Each reward function takes:
    - prompt: list[dict[str, str]] | str
    - completion: list[dict[str, str]] | str
    - answer: Any (metadata for scoring)
    - task (optional): str (type of task)
    - **kwargs: additional kwargs

    Returns:
    - float | list[float] | dict[str, float]
    """

    def __init__(
        self,
        funcs: list[RewardFunc | GroupRewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: vf.Parser | None = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.funcs = funcs or []
        self.weights = weights or []
        if not self.weights:
            self.weights = [1.0] * len(self.funcs)
        elif len(self.weights) != len(self.funcs):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of functions ({len(self.funcs)})"
            )

        self.parser = parser or vf.Parser()

        # class objects for reward functions
        self.class_objects = {}
        if self.parser:
            self.class_objects["parser"] = self.parser

    # public helpers
    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.funcs.append(func)
        self.weights.append(weight)

    def add_metric(self, func: RewardFunc, weight: float = 0.0):
        self.funcs.append(func)
        self.weights.append(weight)

    def add_class_object(self, name: str, obj: Any):
        self.class_objects[name] = obj

    # private helpers
    def _get_reward_func_names(self) -> list[str]:
        return [getattr(func, "__name__", repr(func)) for func in self.funcs]

    def _get_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs]

    def _get_reward_weights(self) -> list[float]:
        return self.weights

    def _is_group_func(self, func: RewardFunc) -> bool:
        """Check if a function is a GroupRewardFunc by inspecting its signature."""
        if self._is_multiagent_func(func):
            return False
        sig = inspect.signature(func)
        # GroupRewardFunc has plural parameters: states, prompts, completions, etc.
        param_names = set(sig.parameters.keys())
        group_indicators = {
            "states",
            "prompts",
            "completions",
            "answers",
            "tasks",
            "infos",
        }
        returns_list = inspect.signature(func).return_annotation is list
        return bool(param_names & group_indicators) or returns_list

    def _is_multiagent_func(self, func: RewardFunc) -> bool:
        """Check if a function is a MultiAgentRewardFunc by inspecting its return annotation."""
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation
        # Check for dict[str, float] or Dict[str, float] return type
        origin = getattr(return_annotation, "__origin__", None)
        return origin is dict

    # individual-level reward helpers
    def _get_individual_reward_func_names(self) -> list[str]:
        return [
            getattr(func, "__name__", repr(func))
            for func in self.funcs
            if not self._is_group_func(func)
        ]

    def _get_individual_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs if not self._is_group_func(func)]

    def _get_individual_reward_weights(self) -> list[float]:
        return [
            weight
            for func, weight in zip(self.funcs, self.weights)
            if not self._is_group_func(func)
        ]

    async def _call_individual_reward_func(
        self,
        func: RewardFunc,
        state: State,
    ) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """

        sig = inspect.signature(func)

        merged = dict(
            prompt=state["prompt"],
            completion=state["completion"],
            answer=state.get("answer", ""),
            state=state,
            task=state["task"],
            info=state.get("info", {}),
        )
        merged.update(self.class_objects)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = float(await maybe_await(func, **merged))
            except Exception as e:
                self.logger.error(
                    f"Error calling reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                )
                ans = 0.0
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = float(await maybe_await(func, **allowed))
            except Exception as e:
                self.logger.error(
                    f"Error calling reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                )
                ans = 0.0
        return ans

    # group-level reward helpers
    def _get_group_reward_func_names(self) -> list[str]:
        return [
            getattr(func, "__name__", repr(func))
            for func in self.funcs
            if self._is_group_func(func)
        ]

    def _get_group_reward_funcs(self) -> list[GroupRewardFunc]:
        return cast(
            list[GroupRewardFunc],
            [func for func in self.funcs if self._is_group_func(func)],
        )

    def _get_group_reward_weights(self) -> list[float]:
        return [
            weight
            for func, weight in zip(self.funcs, self.weights)
            if self._is_group_func(func)
        ]

    async def _call_group_reward_func(
        self,
        func: GroupRewardFunc,
        states: list[State],
    ) -> list[float]:
        """
        Invoke `func` with only the required arguments.
        """

        sig = inspect.signature(func)
        merged = dict(
            prompts=[state["prompt"] for state in states],
            completions=[state["completion"] for state in states],
            answers=[state.get("answer", "") for state in states],
            states=states,
            tasks=[state["task"] for state in states],
            infos=[state.get("info", {}) for state in states],
        )
        merged.update(self.class_objects)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = await maybe_await(func, **merged)
            except Exception as e:
                self.logger.error(
                    f"Error calling group reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                )
                ans = [0.0] * len(states)
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = await maybe_await(func, **allowed)
            except Exception as e:
                self.logger.error(
                    f"Error calling group reward function {func.__name__}: {e}"  # type: ignore[unresolved-attribute]
                )
                ans = [0.0] * len(states)
        return ans

    # multi-agent reward helpers
    def _get_multiagent_reward_funcs(self) -> list[MultiAgentRewardFunc]:
        return cast(
            list[MultiAgentRewardFunc],
            [func for func in self.funcs if self._is_multiagent_func(func)],
        )

    async def _call_multiagent_reward_func(
        self,
        func: MultiAgentRewardFunc,
        state: State,
    ) -> dict[str, float]:
        """Invoke a multi-agent reward function that returns per-agent rewards."""
        sig = inspect.signature(func)
        merged = dict(
            prompt=state["prompt"],
            completion=state["completion"],
            answer=state.get("answer", ""),
            state=state,
            task=state["task"],
            info=state.get("info", {}),
        )
        merged.update(self.class_objects)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = await maybe_await(func, **merged)
            except Exception as e:
                self.logger.error(
                    f"Error calling multi-agent reward function {func.__name__}: {e}"
                )
                ans = {}
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = await maybe_await(func, **allowed)
            except Exception as e:
                self.logger.error(
                    f"Error calling multi-agent reward function {func.__name__}: {e}"
                )
                ans = {}
        return ans

    async def dummy_score_rollout(self, state: State):
        """Score a single rollout with dummy rewards."""
        state["reward"] = 0.0
        state["metrics"] = {}

    async def score_rollout(self, state: State):
        """
        Evaluate all reward functions for a single rollout.
        """
        reward_funcs = self._get_individual_reward_funcs()
        group_reward_funcs = self._get_group_reward_funcs()
        assert len(reward_funcs) > 0 and len(group_reward_funcs) == 0, (
            "Rubric.score_rollout requires at least one individual-level reward function and no group-level reward functions"
        )
        start_time = time.time()
        reward_scores = []
        for func in reward_funcs:
            reward_scores.append(
                await self._call_individual_reward_func(
                    func=func,
                    state=state,
                )
            )
        rewards = RolloutScore(
            metrics={
                func.__name__: reward
                for func, reward in zip(reward_funcs, reward_scores)
            },
            reward=sum(
                [
                    reward * weight
                    for reward, weight in zip(
                        reward_scores, self._get_individual_reward_weights()
                    )
                ]
            ),
        )
        end_time = time.time()
        state["timing"]["scoring_ms"] = (end_time - start_time) * 1000
        state["timing"]["total_ms"] += state["timing"]["scoring_ms"]
        state["reward"] = rewards["reward"]
        state["metrics"] = rewards["metrics"]

    async def dummy_score_group(self, states: list[State]):
        """Score a group of rollouts together with dummy rewards."""
        for state in states:
            await self.dummy_score_rollout(state)

    async def score_group(self, states: list[State]):
        """
        Score a group of rollouts together.

        All reward functions are executed in order, parallelizing across states.
        Supports multi-agent reward functions that return per-agent rewards.
        """
        start_time = time.time()
        num_states = len(states)
        if num_states == 0:
            self.logger.warning("No states to score")
            return
        aggregated_rewards = [0.0] * num_states
        # Per-agent rewards for multi-agent envs: list of dict[agent_id, reward]
        aggregated_agent_rewards: list[dict[str, float]] = [
            {} for _ in range(num_states)
        ]
        aggregated_metrics: dict[str, list[float]] = {}

        # process functions in order
        for func, weight in zip(self.funcs, self.weights):
            is_group = self._is_group_func(func)
            is_multiagent = self._is_multiagent_func(func)

            if is_multiagent:
                # MultiAgentRewardFunc: returns dict[str, float] per state
                multiagent_func = cast(MultiAgentRewardFunc, func)
                score_tasks = [
                    self._call_multiagent_reward_func(multiagent_func, state)
                    for state in states
                ]
                agent_scores_list = await asyncio.gather(*score_tasks)

                func_name = func.__name__
                for i, agent_scores in enumerate(agent_scores_list):
                    # Aggregate per-agent rewards
                    for agent_id, score_value in agent_scores.items():
                        if agent_id not in aggregated_agent_rewards[i]:
                            aggregated_agent_rewards[i][agent_id] = 0.0
                        aggregated_agent_rewards[i][agent_id] += score_value * weight
                    # Also compute a rollout-level reward (mean of agent rewards)
                    if agent_scores:
                        mean_score = sum(agent_scores.values()) / len(agent_scores)
                        aggregated_rewards[i] += mean_score * weight
                        if func_name not in aggregated_metrics:
                            aggregated_metrics[func_name] = [0.0] * num_states
                        aggregated_metrics[func_name][i] = mean_score
            elif is_group:
                # GroupRewardFunc: score all states together
                group_func = cast(GroupRewardFunc, func)
                scores = await self._call_group_reward_func(group_func, states)
                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    score_value = scores[i]
                    aggregated_rewards[i] += score_value * weight
                    aggregated_metrics[func_name][i] = score_value
                    # Also add to each agent's rewards (for multi-agent compatibility)
                    agent_ids = set(
                        t.get("extras", {}).get("agent_id")
                        for t in states[i]["trajectory"]
                        if t.get("extras", {}).get("agent_id")
                    )
                    for agent_id in agent_ids:
                        if agent_id not in aggregated_agent_rewards[i]:
                            aggregated_agent_rewards[i][agent_id] = 0.0
                        aggregated_agent_rewards[i][agent_id] += score_value * weight
            else:
                reward_func = cast(RewardFunc, func)
                score_tasks = [
                    self._call_individual_reward_func(reward_func, state)
                    for state in states
                ]
                scores = await asyncio.gather(*score_tasks)

                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    score_value = scores[i]
                    aggregated_rewards[i] += score_value * weight
                    aggregated_metrics[func_name][i] = score_value
                    # Also add to each agent's rewards (for multi-agent compatibility)
                    agent_ids = set(
                        t.get("extras", {}).get("agent_id")
                        for t in states[i]["trajectory"]
                        if t.get("extras", {}).get("agent_id")
                    )
                    for agent_id in agent_ids:
                        if agent_id not in aggregated_agent_rewards[i]:
                            aggregated_agent_rewards[i][agent_id] = 0.0
                        aggregated_agent_rewards[i][agent_id] += score_value * weight

        # update states with aggregated results
        end_time = time.time()
        scoring_ms = (end_time - start_time) * 1000
        avg_reward = sum(aggregated_rewards) / num_states

        # For multi-agent: compute opponent-conditioned baselines
        # Group states by opponent behavior to isolate each agent's learning signal
        has_multiagent = any(aggregated_agent_rewards[i] for i in range(num_states))
        opponent_baselines: dict[
            str, dict[str, float]
        ] = {}  # {agent_id: {opponent_sig: baseline}}

        # DEBUG
        print(f"[DEBUG] has_multiagent={has_multiagent}, num_states={num_states}")
        print(f"[DEBUG] aggregated_agent_rewards={aggregated_agent_rewards[:3]}")
        # DEBUG: inspect trajectory structure
        if states and states[0].get("trajectory"):
            traj = states[0]["trajectory"]
            print(f"[DEBUG] trajectory len={len(traj)}")
            for idx, t in enumerate(traj):
                print(
                    f"[DEBUG] step {idx}: agent_id={t.get('extras', {}).get('agent_id')}, completion={t.get('completion')}"
                )

        if has_multiagent:
            # Build opponent-conditioned baselines for each agent
            # For each agent, group rollouts by what the opponent(s) did
            agent_ids_in_group: set[str] = set()
            for agent_rewards in aggregated_agent_rewards:
                agent_ids_in_group.update(agent_rewards.keys())

            for agent_id in agent_ids_in_group:
                # For each state, extract opponent's actions (actions by agents != agent_id)
                opponent_groups: dict[
                    str, list[tuple[int, float]]
                ] = {}  # {opponent_signature: [(state_idx, agent_reward)]}
                for i, state in enumerate(states):
                    # Get opponent's action signature from trajectory
                    opponent_actions = []
                    for t in state["trajectory"]:
                        step_agent_id = t.get("extras", {}).get("agent_id")
                        if step_agent_id and step_agent_id != agent_id:
                            # Use completion content as opponent action signature
                            opponent_actions.append(str(t.get("completion", "")))
                    opponent_sig = "|".join(opponent_actions)

                    # Get this agent's reward for this state
                    agent_reward = aggregated_agent_rewards[i].get(
                        agent_id, aggregated_rewards[i]
                    )
                    if opponent_sig not in opponent_groups:
                        opponent_groups[opponent_sig] = []
                    opponent_groups[opponent_sig].append((i, agent_reward))

                # Compute baseline for each opponent behavior group
                opponent_baselines[agent_id] = {}
                for opponent_sig, rewards_list in opponent_groups.items():
                    if rewards_list:
                        baseline = sum(r for _, r in rewards_list) / len(rewards_list)
                        opponent_baselines[agent_id][opponent_sig] = baseline

                # DEBUG
                print(
                    f"[DEBUG] agent_id={agent_id}, opponent_groups keys={list(opponent_groups.keys())[:3]}"
                )
                print(
                    f"[DEBUG] opponent_baselines[{agent_id}]={opponent_baselines[agent_id]}"
                )

        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            state["advantage"] = aggregated_rewards[i] - avg_reward

            # Store per-agent rewards if any multi-agent funcs were used
            agent_rewards = aggregated_agent_rewards[i]
            if agent_rewards:
                state["agent_rewards"] = agent_rewards

            # Assign per-step rewards and advantages based on agent_id (for multi-agent)
            for t in state["trajectory"]:
                if t["reward"] is None:
                    if agent_rewards:
                        agent_id = t.get("extras", {}).get("agent_id")
                        t["reward"] = agent_rewards.get(agent_id, state["reward"])
                    else:
                        t["reward"] = state["reward"]

                # Compute per-agent advantage with opponent-conditioned baseline
                if t["advantage"] is None:
                    agent_id = t.get("extras", {}).get("agent_id")
                    if agent_id and agent_id in opponent_baselines:
                        # Get opponent's action signature for this state
                        opponent_actions = []
                        for t2 in state["trajectory"]:
                            step_agent_id = t2.get("extras", {}).get("agent_id")
                            if step_agent_id and step_agent_id != agent_id:
                                opponent_actions.append(str(t2.get("completion", "")))
                        opponent_sig = "|".join(opponent_actions)
                        # Use opponent-conditioned baseline
                        baseline = opponent_baselines[agent_id].get(
                            opponent_sig, avg_reward
                        )
                        t["advantage"] = t["reward"] - baseline
                        # DEBUG (only first few)
                        if i < 2:
                            print(
                                f"[DEBUG] i={i} agent={agent_id} reward={t['reward']} baseline={baseline} adv={t['advantage']}"
                            )
                    else:
                        t["advantage"] = t["reward"] - avg_reward
                        if i < 2:
                            print(
                                f"[DEBUG] i={i} agent={agent_id} NOT in opponent_baselines, using avg_reward"
                            )

            state["metrics"] = {
                func_name: values[i] for func_name, values in aggregated_metrics.items()
            }
            state["timing"]["scoring_ms"] = scoring_ms
            state["timing"]["total_ms"] += state["timing"]["scoring_ms"]
