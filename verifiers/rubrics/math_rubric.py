import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from math_verify import parse, verify  # type: ignore[unresolved-import]

from verifiers.parsers.maybe_think_parser import MaybeThinkParser
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc
from verifiers.utils.data_utils import extract_boxed_answer


@dataclass
class MathVerifyResult:
    """Result of math verification with timing information."""

    reward: float
    elapsed: float
    error: str | None = None


def verify_response(response: str, answer: str) -> MathVerifyResult:
    """Verify a response against an answer using math_verify."""
    start = time.perf_counter()
    try:
        parsed_answer = parse(f"\\boxed{{{answer}}}", parsing_timeout=None)  # type: ignore[arg-type]
        parsed_response = parse(f"\\boxed{{{response}}}", parsing_timeout=None)  # type: ignore[arg-type]
        is_correct = verify(parsed_answer, parsed_response, timeout_seconds=None)
        elapsed = time.perf_counter() - start

        return MathVerifyResult(reward=float(is_correct), elapsed=elapsed)
    except BaseException as e:
        elapsed = time.perf_counter() - start
        return MathVerifyResult(
            reward=0.0, elapsed=elapsed, error=f"{type(e).__name__}: {e!r}"
        )


class MathRubric(Rubric):
    HARD_TIMEOUT_SECONDS: float = 120.0

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
        max_workers: int = 1,
        timeout: float = 5,
    ):
        parser = parser or MaybeThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer)
        self.timeout = timeout

        # use 'spawn' for clean process isolation (no inherited state)
        ctx = multiprocessing.get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        )

        # suppress math_verify timeout warnings (we handle timeouts ourselves)
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    async def correct_answer(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""

        response = parser.parse_answer(completion) or ""
        if response == "":
            self.logger.warning("Parsed response is empty")
            return 0.0

        loop = asyncio.get_running_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, verify_response, response, answer),
                timeout=self.HARD_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Math verification hard timeout after {self.HARD_TIMEOUT_SECONDS:.0f}s. Worker may be hung or main event loop experiences severe lag."
            )
            return 0.0
        except asyncio.CancelledError:
            raise

        if result.error is not None:
            self.logger.warning(f"Math verification failed: {result.error}")

        # Enforce timeout based on actual verification wall-clock time (measured in worker)
        if result.elapsed > self.timeout:
            self.logger.debug(
                f"Math verification exceeded time limit: "
                f"{result.elapsed:.2f}s > {self.timeout:.1f}s"
            )
            return 0.0

        return result.reward

    def __del__(self):
        """Shutdown the thread pool executor when the object is garbage collected."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
