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
        parsing_timeout_seconds: int = 5,
        verify_timeout_seconds: int = 5,
    ):
        parser = parser or MaybeThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer)
        self.parsing_timeout_seconds = parsing_timeout_seconds
        self.verify_timeout_seconds = verify_timeout_seconds

    def correct_answer(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        try:
            response = parser.parse_answer(completion) or ""
            if response == "":
                return 0.0
            parsed_answer = parse(
                f"\\boxed{{{answer}}}", parsing_timeout=self.parsing_timeout_seconds
            )
            parsed_response = parse(
                f"\\boxed{{{response}}}", parsing_timeout=self.parsing_timeout_seconds
            )
            if verify(
                parsed_answer,
                parsed_response,
                timeout_seconds=self.verify_timeout_seconds,
            ):
                return 1.0
            else:
                return 0.0
        except BaseException:
            return 0.0
