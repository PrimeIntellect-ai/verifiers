import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from math_verify import parse, verify

from verifiers.parsers.maybe_think_parser import MaybeThinkParser
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc
from verifiers.utils.data_utils import extract_boxed_answer


class MathRubric(Rubric):
    HARD_TIMEOUT_SECONDS: float = 120.0

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
        max_workers: int = 50,
        timeout_seconds: float = 5,
    ):
        parser = parser or MaybeThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer)
        self.timeout_seconds = timeout_seconds

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="math-verify",
        )

        # suppress math_verify timeout warnings (we handle timeouts ourselves)
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    async def correct_answer(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""

        def verify_response() -> tuple[float, float]:
            """
            Verify a response against an answer using math_verify.

            Times itself internally so event loop lag doesn't affect scoring.
            """
            start = time.perf_counter()
            response = parser.parse_answer(completion) or ""
            if response == "":
                elapsed = time.perf_counter() - start
                return 0.0, elapsed

            try:
                parsed_answer = parse(
                    f"\\boxed{{{answer}}}", parsing_timeout=cast(int, None)
                )
                parsed_response = parse(
                    f"\\boxed{{{response}}}", parsing_timeout=cast(int, None)
                )
                is_correct = verify(
                    parsed_answer, parsed_response, timeout_seconds=None
                )
                elapsed = time.perf_counter() - start

                return float(is_correct), elapsed
            except BaseException:
                elapsed = time.perf_counter() - start
                return 0.0, elapsed

        loop = asyncio.get_running_loop()

        try:
            reward, elapsed = await asyncio.wait_for(
                loop.run_in_executor(self.executor, verify_response),
                timeout=self.HARD_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Math verification hit hard timeout after {self.HARD_TIMEOUT_SECONDS:.0f}s. Worker may be hung or main event loop experiences severe lag."
            )
            return 0.0
        except Exception as e:
            self.logger.warning(f"Math verification failed: {e}")
            return 0.0

        if elapsed > self.timeout_seconds:
            self.logger.debug(
                f"Math verification exceeded time limit after {elapsed:.2f}s (>{self.timeout_seconds:.1f}s)"
            )
            return 0.0

        return reward

    def __del__(self):
        """Shutdown the thread pool executor when the object is garbage collected."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
