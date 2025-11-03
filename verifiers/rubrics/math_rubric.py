from math_verify import parse, verify  # type: ignore[unresolved-import]

from verifiers.parsers.maybe_think_parser import MaybeThinkParser
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc
from verifiers.utils.data_utils import extract_boxed_answer


class MathRubric(Rubric):
    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
    ):
        parser = parser or MaybeThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer_reward_func)

    def correct_answer_reward_func(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        try:
            response = parser.parse_answer(completion) or ""
            if response == "":
                return 0.0
            if verify(
                parse(f"\\boxed{{{answer}}}", parsing_timeout=5),
                parse(f"\\boxed{{{response}}}", parsing_timeout=5),
                timeout_seconds=5,
            ):
                return 1.0
            else:
                return 0.0
        except BaseException:
            return 0.0
