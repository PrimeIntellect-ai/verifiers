import asyncio
import inspect
import logging
import operator
import time
from typing import Any, AsyncContextManager, cast

import verifiers as vf
from verifiers.types import (
    AdvantageMode,
    GateExpr,
    GroupRewardFunc,
    RewardFunc,
    RolloutScore,
    State,
)
from verifiers.utils.async_utils import maybe_await

# Operator mapping for gate evaluation
_GATE_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


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
        advantage_mode: AdvantageMode = "grpo",
        gates: dict[str, GateExpr] | None = None,
        epsilon: float = 1e-8,
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

        # GDPO parameters (arXiv:2601.05242)
        self.advantage_mode: AdvantageMode = advantage_mode
        self.gates = gates or {}
        self.epsilon = epsilon

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
        return [func.__name__ for func in self.funcs]  # type: ignore[possibly-missing-attribute]

    def _get_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs]

    def _get_reward_weights(self) -> list[float]:
        return self.weights

    def _is_group_func(self, func: RewardFunc) -> bool:
        """Check if a function is a GroupRewardFunc by inspecting its signature."""
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

    # individual-level reward helpers
    def _get_individual_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.funcs if not self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

    def _get_individual_reward_funcs(self) -> list[RewardFunc]:
        return [func for func in self.funcs if not self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

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
        score_sem: AsyncContextManager,
    ) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """

        async def _call():
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

        async with score_sem:
            return await _call()

    # group-level reward helpers
    def _get_group_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.funcs if self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

    def _get_group_reward_funcs(self) -> list[GroupRewardFunc]:
        return [func for func in self.funcs if self._is_group_func(func)]  # type: ignore[possibly-missing-attribute]

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
        score_sem: AsyncContextManager,
    ) -> list[float]:
        """
        Invoke `func` with only the required arguments.
        """

        async def _call():
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

        async with score_sem:
            return await _call()

    # GDPO helper methods (arXiv:2601.05242)
    def _evaluate_gate(self, expr: GateExpr, metrics: dict[str, float]) -> bool:
        """
        Recursively evaluate a gate expression (AND/OR/NOT with arbitrary nesting).

        Args:
            expr: A GateCondition or nested boolean expression
            metrics: Dictionary of reward function scores for the current sample

        Returns:
            True if the gate condition passes, False otherwise
        """
        if "func" in expr:
            # Base condition: compare metrics[func] with threshold
            func_name = expr["func"]
            op_str = expr.get("op", ">=")
            threshold = expr.get("value", 0.0)

            if func_name not in metrics:
                self.logger.warning(
                    f"Gate references '{func_name}' but it's not in metrics. Defaulting to False."
                )
                return False

            op_func = _GATE_OPS.get(op_str)
            if op_func is None:
                self.logger.warning(
                    f"Unknown gate operator '{op_str}'. Defaulting to False."
                )
                return False

            return op_func(metrics[func_name], threshold)

        elif "AND" in expr:
            return all(self._evaluate_gate(sub, metrics) for sub in expr["AND"])

        elif "OR" in expr:
            return any(self._evaluate_gate(sub, metrics) for sub in expr["OR"])

        elif "NOT" in expr:
            return not self._evaluate_gate(expr["NOT"], metrics)

        else:
            self.logger.warning(
                f"Invalid gate expression: {expr}. Defaulting to False."
            )
            return False

    def _zscore_normalize_group(self, values: list[float]) -> list[float]:
        """
        Z-score normalize values WITHOUT epsilon (per-reward group normalization).

        Per paper Equation 4: A_k = (r_k - μ) / σ
        If σ = 0 (all values identical), returns zeros (no gradient signal).

        Args:
            values: List of values to normalize within a group

        Returns:
            List of z-score normalized values
        """
        n = len(values)
        if n == 0:
            return []

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance**0.5

        # If std ≈ 0, all values are identical -> no signal, return zeros
        if std < 1e-12:
            return [0.0] * n

        return [(v - mean) / std for v in values]

    def _zscore_normalize_batch(self, values: list[float]) -> list[float]:
        """
        Z-score normalize values WITH epsilon (batch-wise normalization).

        Per paper Equation 6: Â_sum = (A_sum - μ) / (σ + ε)
        Epsilon provides numerical stability for batch normalization.

        Args:
            values: List of values to normalize across batch

        Returns:
            List of z-score normalized values
        """
        n = len(values)
        if n == 0:
            return []

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance**0.5

        return [(v - mean) / (std + self.epsilon) for v in values]

    def _compute_gdpo_advantages(
        self,
        states: list[State],
        aggregated_metrics: dict[str, list[float]],
    ) -> list[float]:
        """
        Compute GDPO advantages per paper (arXiv:2601.05242).

        Algorithm:
        1. Apply reward conditioning/gating (Eq. 8)
        2. Per-reward GROUP-wise z-score normalization (Eq. 4) - within each question's rollouts
        3. Sum normalized advantages with weights (Eq. 5)
        4. BATCH-wise z-score normalization with epsilon (Eq. 6)

        Args:
            states: List of states (needed to group by example_id)
            aggregated_metrics: Raw scores per reward function {func_name: [scores]}

        Returns:
            List of final advantages for each state
        """
        num_states = len(states)
        if num_states == 0:
            return []

        # Build index mapping: example_id -> list of state indices
        example_groups: dict[int, list[int]] = {}
        for idx, state in enumerate(states):
            example_id = state.get(
                "example_id", idx
            )  # fallback to idx if no example_id
            if example_id not in example_groups:
                example_groups[example_id] = []
            example_groups[example_id].append(idx)

        # Step 1: Apply gating conditions (Eq. 8)
        # r̃_k = r_k if r_l >= t else 0
        gated_metrics: dict[str, list[float]] = {}
        for func_name, scores in aggregated_metrics.items():
            gated = list(scores)  # copy
            if func_name in self.gates:
                for i in range(num_states):
                    # Evaluate gate on RAW metrics (not gated)
                    sample_metrics = {
                        fn: vals[i] for fn, vals in aggregated_metrics.items()
                    }
                    if not self._evaluate_gate(self.gates[func_name], sample_metrics):
                        gated[i] = 0.0
            gated_metrics[func_name] = gated

        # Step 2: Per-reward GROUP-wise z-score normalization (Eq. 4)
        # A_k^(i,j) = (r_k^(i,j) - μ_k^i) / σ_k^i
        # where μ_k^i and σ_k^i are computed within question i's G rollouts
        normalized: dict[str, list[float]] = {
            func_name: [0.0] * num_states for func_name in gated_metrics
        }

        for func_name, scores in gated_metrics.items():
            for group_indices in example_groups.values():
                # Extract scores for this group
                group_scores = [scores[i] for i in group_indices]
                # Normalize within group (no epsilon per Eq. 4)
                group_normalized = self._zscore_normalize_group(group_scores)
                # Write back to normalized array
                for local_idx, global_idx in enumerate(group_indices):
                    normalized[func_name][global_idx] = group_normalized[local_idx]

        # Step 3: Sum normalized advantages with weights (Eq. 5)
        # A_sum^(i,j) = Σ_k w_k · A_k^(i,j)
        func_weights = {f.__name__: w for f, w in zip(self.funcs, self.weights)}
        A_sum = [0.0] * num_states
        for func_name, norm_scores in normalized.items():
            weight = func_weights.get(func_name, 1.0)
            for i in range(num_states):
                A_sum[i] += norm_scores[i] * weight

        # Step 4: BATCH-wise z-score normalization with epsilon (Eq. 6)
        # Â_sum^(i,j) = (A_sum^(i,j) - μ_batch) / (σ_batch + ε)
        return self._zscore_normalize_batch(A_sum)

    async def dummy_score_rollout(self, state: State):
        """Score a single rollout with dummy rewards."""
        state["reward"] = 0.0
        state["metrics"] = {}

    async def score_rollout(self, state: State, score_sem: AsyncContextManager):
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
                    score_sem=score_sem,
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

    async def score_group(self, states: list[State], score_sem: AsyncContextManager):
        """
        Score a group of rollouts together.

        All reward functions are executed in order, parallelizing across states.
        """
        start_time = time.time()
        num_states = len(states)
        if num_states == 0:
            self.logger.warning("No states to score")
            return
        aggregated_rewards = [0.0] * num_states
        aggregated_metrics: dict[str, list[float]] = {}

        # process functions in order
        for func, weight in zip(self.funcs, self.weights):
            is_group = self._is_group_func(func)
            if is_group:
                # GroupRewardFunc: score all states together
                group_func = cast(GroupRewardFunc, func)
                scores = await self._call_group_reward_func(
                    group_func, states, score_sem=score_sem
                )
                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    score_value = scores[i]
                    aggregated_rewards[i] += score_value * weight
                    aggregated_metrics[func_name][i] = score_value
            else:
                reward_func = cast(RewardFunc, func)
                score_tasks = [
                    self._call_individual_reward_func(
                        reward_func, state, score_sem=score_sem
                    )
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

        # update states with aggregated results
        end_time = time.time()
        scoring_ms = (end_time - start_time) * 1000

        # Compute advantages based on mode
        if self.advantage_mode == "gdpo":
            # GDPO: per-reward normalized, gated advantages (arXiv:2601.05242)
            advantages = self._compute_gdpo_advantages(states, aggregated_metrics)
        else:
            # GRPO (default): simple mean subtraction
            avg_reward = sum(aggregated_rewards) / num_states
            advantages = [r - avg_reward for r in aggregated_rewards]

        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            state["advantage"] = advantages[i]
            for t in state["trajectory"]:
                if t["advantage"] is None:
                    t["advantage"] = state["advantage"]
                if t["reward"] is None:
                    t["reward"] = state["reward"]
            state["metrics"] = {
                func_name: values[i] for func_name, values in aggregated_metrics.items()
            }
            state["timing"]["scoring_ms"] = scoring_ms
            state["timing"]["total_ms"] += state["timing"]["scoring_ms"]
