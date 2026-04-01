import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import cast

from math_verify import parse, verify

from verifiers.parsers.maybe_think_parser import MaybeThinkParser
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.thread_utils import register_executor, unregister_executor


def verify_response(
    response: str,
    answer: str,
    max_verify_chars: int,
) -> tuple[float, float]:
    """
    Verify a response against an answer using math_verify.

    Top-level function so it can be pickled for ProcessPoolExecutor.
    Times itself internally so event loop lag doesn't affect scoring.
    """
    start = time.perf_counter()
    if response == "":
        elapsed = time.perf_counter() - start
        return 0.0, elapsed

    if len(response) > max_verify_chars:
        elapsed = time.perf_counter() - start
        return 0.0, elapsed

    try:
        parsed_answer = parse(f"\\boxed{{{answer}}}", parsing_timeout=cast(int, None))
        parsed_response = parse(
            f"\\boxed{{{response}}}", parsing_timeout=cast(int, None)
        )
        is_correct = verify(parsed_answer, parsed_response, timeout_seconds=None)
        elapsed = time.perf_counter() - start
        return float(is_correct), elapsed
    except BaseException:
        elapsed = time.perf_counter() - start
        return 0.0, elapsed


class MathRubric(Rubric):
    HARD_TIMEOUT_SECONDS: float = 120.0
    MAX_VERIFY_CHARS: int = 50_000

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
        max_workers: int = 8,
        timeout_seconds: float = 5,
        max_verify_chars: int = MAX_VERIFY_CHARS,
    ):
        from functools import partial

        parser = parser or MaybeThinkParser(
            extract_fn=partial(extract_boxed_answer, strict=True)
        )
        self.max_verify_chars = max_verify_chars
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer)
        self.timeout_seconds = timeout_seconds

        self.executor = ProcessPoolExecutor(max_workers=1)
        self.executor_name = f"math-verify-{id(self)}"
        register_executor(
            self.executor_name,
            self.executor,
            scaling_fn=lambda c: min(
                max(1, c // 128), min(max_workers, os.cpu_count() or 1)
            ),
        )

        # suppress math_verify timeout warnings (we handle timeouts ourselves)
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    async def correct_answer(
        self, parser: Parser, completion: Messages, answer: str, **kwargs
    ) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        response = parser.parse_answer(completion) or ""

        if len(response) > self.max_verify_chars:
            self.logger.warning(
                f"Skipping math verification: parsed response too long "
                f"({len(response)} chars > {self.max_verify_chars} limit)"
            )
            return 0.0

        self.logger.debug(
            f"Math verify input: response={response[:200]!r}, answer={answer[:200]!r}"
        )

        loop = asyncio.get_running_loop()

        try:
            reward, elapsed = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    verify_response,
                    response,
                    answer,
                    self.max_verify_chars,
                ),
                timeout=self.HARD_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Math verification hit hard timeout after {self.HARD_TIMEOUT_SECONDS:.0f}s. "
                f"response={response[:200]!r}, answer={answer[:200]!r}"
            )
            return 0.0
        except Exception as e:
            self.logger.warning(
                f"Math verification failed: {e}. "
                f"response={response[:200]!r}, answer={answer[:200]!r}"
            )
            return 0.0

        if elapsed > self.timeout_seconds:
            self.logger.debug(
                f"Math verification exceeded time limit after {elapsed:.2f}s (>{self.timeout_seconds:.1f}s). "
                f"response={response[:200]!r}, answer={answer[:200]!r}"
            )
            return 0.0

        return reward

    async def teardown(self):
        """Shut down the PPE cleanly so _python_exit doesn't hang.

        On Python <3.13, workers forked from a threaded process (asyncio +
        ZMQ) can deadlock. Killing them first lets the manager thread exit,
        so _python_exit()'s join() returns immediately.
        """
        await super().teardown()
        if hasattr(self, "executor_name"):
            unregister_executor(self.executor_name)
        if hasattr(self, "executor"):
            if sys.version_info < (3, 13):
                procs = list(
                    (getattr(self.executor, "_processes", None) or {}).values()
                )
                for p in procs:
                    p.kill()
                for p in procs:
                    p.join(timeout=5)
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        """Best-effort cleanup if teardown() was never called."""
        if hasattr(self, "executor_name"):
            unregister_executor(self.executor_name)
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
