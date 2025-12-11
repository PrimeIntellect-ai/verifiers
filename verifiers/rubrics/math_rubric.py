import asyncio

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
        timeout_seconds: float = 5,
    ):
        parser = parser or MaybeThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer)
        self.timeout_seconds = timeout_seconds

    async def correct_answer(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""

        async def _correct_answer() -> float:
            try:
                response = (
                    await asyncio.to_thread(parser.parse_answer, completion)
                ) or ""
                if response == "":
                    return 0.0

                def parse_answer():
                    return parse(
                        f"\\boxed{{{answer}}}",
                        parsing_timeout=None,  # type: ignore
                    )

                parsed_answer = await asyncio.to_thread(parse_answer)

                def parse_response():
                    return parse(
                        f"\\boxed{{{response}}}",
                        parsing_timeout=None,  # type: ignore
                    )

                parsed_response = await asyncio.to_thread(parse_response)

                def verify_result():
                    return verify(
                        parsed_answer,
                        parsed_response,
                        timeout_seconds=None,
                    )

                result = await asyncio.to_thread(verify_result)
                if result:
                    return 1.0
                else:
                    return 0.0
            except BaseException:
                return 0.0

        try:
            return await asyncio.wait_for(
                _correct_answer(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            return 0.0
