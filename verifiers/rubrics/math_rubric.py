import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

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
        max_workers: int = 10,
        timeout_seconds: float = 5,
    ):
        parser = parser or MaybeThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer)
        self.timeout_seconds = timeout_seconds
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="math-rubric",
        )

        # suppress math_verify timeout warnings (we handle timeouts ourselves via asyncio.wait_for)
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    async def run_in_executor(self, func: Callable, *args) -> Any:
        """Run a sync function in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def correct_answer(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""

        async def _correct_answer() -> float:
            try:
                response = (
                    await self.run_in_executor(lambda: parser.parse_answer(completion))
                    or ""
                )
                if response == "":
                    self.logger.warning("Parsed response is empty")
                    return 0.0

                parsed_answer = await self.run_in_executor(
                    lambda: parse(f"\\boxed{{{answer}}}", parsing_timeout=None)  # type: ignore
                )
                parsed_response = await self.run_in_executor(
                    lambda: parse(f"\\boxed{{{response}}}", parsing_timeout=None)  # type: ignore
                )
                result = await self.run_in_executor(
                    lambda: verify(parsed_answer, parsed_response, timeout_seconds=None)
                )
                return 1.0 if result else 0.0
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                self.logger.warning(
                    f"Math verification failed with {type(e).__name__}: {e!r}"
                )
                return 0.0

        try:
            return await asyncio.wait_for(
                _correct_answer(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Math verification timed out after {self.timeout_seconds:.1f}s"
            )
            return 0.0

    def __del__(self):
        """Shutdown the thread pool executor when the object is garbage collected."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
