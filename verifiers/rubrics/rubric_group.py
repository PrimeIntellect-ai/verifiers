from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    Info,
    Messages,
    RewardFunc,
    RolloutScore,
    RolloutScores,
    State,
)


class RubricGroup(Rubric):
    """
    Class for aggregating multiple rubrics.
    """

    def __init__(self, rubrics: list[Rubric], **kwargs):
        if not rubrics:
            raise ValueError("RubricGroup must have at least one rubric")
        super().__init__(**kwargs)
        self.rubrics = rubrics
        self.logger.info(f"Initialized RubricGroup with {len(rubrics)} rubrics")

    def get_reward_func_names(self) -> list[str]:
        names = []
        for rubric in self.rubrics:
            names.extend(rubric.get_reward_func_names())
        return names

    def get_reward_funcs(self) -> list[RewardFunc]:
        funcs = []
        for rubric in self.rubrics:
            funcs.extend(rubric.get_reward_funcs())
        return funcs

    def get_reward_weights(self) -> list[float]:
        weights = []
        for rubric in self.rubrics:
            weights.extend(rubric.get_reward_weights())
        return weights

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        assert len(self.rubrics) > 0, "RubricGroup must have at least one rubric"
        self.logger.warning("Adding reward function to the first rubric in the group.")
        self.rubrics[0].add_reward_func(func, weight)

    async def score_rollout(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        task: str = "default",
        info: Info | None = None,
        example_id: int | None = None,
        **kwargs,
    ) -> RolloutScore:
        total_reward = 0.0
        aggregated_metrics: dict[str, float] = {}
        for rubric in self.rubrics:
            score = await rubric.score_rollout(
                prompt,
                completion,
                answer,
                state,
                task,
                info,
                example_id,
                **kwargs,
            )
            total_reward += score.reward
            for key, value in score.metrics.items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value
        return RolloutScore(reward=total_reward, metrics=aggregated_metrics)

    async def score_rollouts(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        answers: list[str],
        states: list[State],
        tasks: list[str],
        infos: list[Info],
        example_ids: list[int] | None = None,
        max_concurrent: int = -1,
        use_tqdm: bool = True,
        **kwargs,
    ) -> RolloutScores:
        """
        Run all rubrics sequentially and return the aggregated scores.

        Reward functions with the same name are summed up.
        """
        example_ids = example_ids or list(range(len(prompts)))
        all_scores = RolloutScores(
            reward=[],
            metrics={},
        )
        for rubric in self.rubrics:
            rubric_scores = await rubric.score_rollouts(
                prompts,
                completions,
                answers,
                states,
                tasks,
                infos,
                example_ids,
                max_concurrent=max_concurrent,
                use_tqdm=use_tqdm,
                **kwargs,
            )
            # aggregate reward (element-wise sum across rubrics)
            if not all_scores.reward:
                all_scores.reward = rubric_scores.reward
            else:
                all_scores.reward = [
                    a + b for a, b in zip(all_scores.reward, rubric_scores.reward)
                ]
            for key, value in rubric_scores.metrics.items():
                if key in all_scores.metrics:
                    # element-wise sum
                    all_scores.metrics[key] = [
                        a + b for a, b in zip(all_scores.metrics[key], value)
                    ]
                else:
                    all_scores.metrics[key] = value
        return all_scores
