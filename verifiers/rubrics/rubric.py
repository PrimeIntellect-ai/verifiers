import asyncio
import inspect
import logging

from verifiers.parsers.parser import Parser
from verifiers.types import (
    Info,
    Messages,
    RewardFunc,
    RolloutScore,
    RolloutScores,
    State,
)


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
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
        parallelize_scoring: bool = True,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")

        self.reward_funcs = funcs or []
        self.reward_weights = weights or []
        self.parser = parser or Parser()

        for key, value in kwargs.items():
            setattr(self, key, value)
        if not self.reward_weights:
            self.reward_weights = [1.0] * len(self.reward_funcs)
        self.parallelize_scoring = parallelize_scoring

    def get_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.reward_funcs]

    def get_reward_funcs(self) -> list[RewardFunc]:
        return self.reward_funcs  # type: ignore

    def get_reward_weights(self) -> list[float]:
        return self.reward_weights  # type: ignore

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.reward_funcs.append(func)
        self.reward_weights.append(weight)

    async def call_reward_func(
        self,
        func: RewardFunc,
        parser: Parser,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        task: str = "default",
        info: Info | None = None,
        **kwargs,
    ) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """
        info = info or {}
        sig = inspect.signature(func)

        common = dict(
            parser=parser,
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
            task=task,
            info=info,
        )
        ans = 0.0
        merged = {**common, **kwargs}
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = func(**merged)
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = func(**allowed)
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        return ans

    async def score_rollout(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        task: str = "default",
        info: Info | None = None,
        **kwargs,
    ) -> RolloutScore:
        """
        Evaluate all reward functions asynchronously for a single rollout.
        """
        if self.parallelize_scoring:
            score_tasks = [
                self.call_reward_func(
                    func=func,
                    parser=self.parser,
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    task=task,
                    info=info,
                    **kwargs,
                )
                for func in self.get_reward_funcs()
            ]
            reward_scores = await asyncio.gather(*score_tasks)
        else:
            reward_scores = []
            for func in self.get_reward_funcs():
                score = await self.call_reward_func(
                    func=func,
                    parser=self.parser,
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    task=task,
                    info=info,
                    **kwargs,
                )
                reward_scores.append(score)
        rewards = RolloutScore(
            metrics={
                func.__name__: reward
                for func, reward in zip(self.get_reward_funcs(), reward_scores)
            },
            reward=sum(
                [
                    reward * weight
                    for reward, weight in zip(reward_scores, self.get_reward_weights())
                ]
            ),
        )
        return rewards

    async def score_rollouts(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        answers: list[str],
        states: list[State],
        tasks: list[str],
        infos: list[Info],
        **kwargs,
    ) -> RolloutScores:
        """
        Compute reward scores for a group of rollouts.

        Default behavior:
        - evaluate each rollout asynchronously
        - return list of dictionaries of reward function names and their scores

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - scores computed using global state stored in Rubric class
        """
        from tqdm.asyncio import tqdm_asyncio

        rollout_tasks = [
            self.score_rollout(*pcasti, **kwargs)
            for pcasti in zip(prompts, completions, answers, states, tasks, infos)
        ]
        rewards = await tqdm_asyncio.gather(
            *rollout_tasks,
            total=len(prompts),
            desc=f"Evaluating {len(prompts)} rollouts",
        )

        if not rewards:
            reward_func_names = self.get_reward_func_names()
            return RolloutScores(
                reward=[],
                metrics={name: [] for name in reward_func_names},
            )

        return RolloutScores(
            reward=[reward.reward for reward in rewards],
            metrics={
                k: [item.metrics[k] for item in rewards] for k in rewards[0].metrics
            },
        )
