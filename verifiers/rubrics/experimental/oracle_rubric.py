import inspect
import logging
from typing import Any, Callable

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State
from verifiers.types import GroupRewardFunc, RewardFunc
from verifiers.utils.async_utils import maybe_await


class OracleRubric(Rubric):
    """Scores generations with an external oracle (backend endpoint).

    Parallel to JudgeRubric: instantiate with the oracle backend and an
    optional oracle_fn, then register reward functions via add_reward_func.
    Reward functions receive ``oracle`` injected automatically and call it
    directly, the same way toxicity and judge examples call ``judge``::

        async def my_score(oracle, prompt, completion, answer, state, **kwargs):
            result = await oracle(prompt, completion, answer, state)
            threshold = answer.get("threshold", 0) if isinstance(answer, dict) else 0
            return 1.0 if float(result) >= threshold else 0.0

        rubric = vf.OracleRubric(oracle=my_backend, oracle_fn=call_backend)
        rubric.add_reward_func(my_score)

    Args:
        oracle: Backend client or callable. Passed as ``oracle`` kwarg to
            oracle_fn (analogous to ``judge_client`` in JudgeRubric).
        parser: Response parser. Defaults to a standard Parser.
        funcs: Optional list of reward functions (also registerable via
            add_reward_func).
        weights: Optional weights for each reward function.
        oracle_fn: Function that calls the backend. Receives ``oracle``
            (the backend), ``prompt``, ``completion``, ``answer``, ``state``,
            and ``response`` as available kwargs. Returns raw oracle output.
            If omitted, the backend is called directly with the parsed response.
        cache_measurements: Cache oracle outputs within a rollout. Default True.
    """

    def __init__(
        self,
        oracle: Any = None,
        parser: Parser | None = None,
        funcs: list[RewardFunc | GroupRewardFunc] | None = None,
        weights: list[float] | None = None,
        oracle_fn: Callable[..., Any] | None = None,
        cache_measurements: bool = True,
    ):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.oracle_backend = oracle
        self.oracle_fn = oracle_fn
        self.cache_measurements = cache_measurements
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            "Initialized OracleRubric (backend=%s, oracle_fn=%s, cache=%s)",
            type(self.oracle_backend).__name__ if self.oracle_backend is not None else "None",
            self.oracle_fn is not None,
            self.cache_measurements,
        )

        self.class_objects = {
            "parser": self.parser,
            "oracle": self.oracle,
            "oracle_backend": self.oracle_backend,
        }

    def _call_with_supported_kwargs(
        self,
        func: Callable[..., Any],
        **kwargs,
    ) -> Any:
        sig = inspect.signature(func)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        ):
            return maybe_await(func, **kwargs)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return maybe_await(func, **allowed)

    def _cache_key(self, prompt: Messages, response: str, answer: Any) -> str:
        return repr((prompt, response, answer))

    async def oracle(
        self,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
    ) -> Any:
        """Call oracle backend and return raw output.

        Parallel to judge() in JudgeRubric. Reward functions receive this
        method as ``oracle`` and call it directly::

            result = await oracle(prompt, completion, answer, state)

        Handles caching automatically within a rollout.
        """
        response = self.parser.parse_answer(completion) or ""
        cache_key = self._cache_key(prompt, response, answer)
        cached = state.get("oracle_cache") if state else None
        if (
            self.cache_measurements
            and isinstance(cached, dict)
            and cache_key in cached
        ):
            return cached[cache_key]

        try:
            if self.oracle_fn is not None:
                result = await self._call_with_supported_kwargs(
                    self.oracle_fn,
                    oracle=self.oracle_backend,
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    response=response,
                )
            elif hasattr(self.oracle_backend, "predict") and callable(
                self.oracle_backend.predict
            ):
                self.logger.info("Calling oracle backend .predict(response)")
                result = await maybe_await(self.oracle_backend.predict, response)
            elif callable(self.oracle_backend):
                self.logger.info("Calling oracle backend as callable(response)")
                result = await maybe_await(self.oracle_backend, response)
            else:
                raise TypeError(
                    "OracleRubric requires `oracle` to be callable or paired with `oracle_fn`."
                )
        except Exception as e:
            self.logger.warning(
                f"Oracle call failed. Oracle: {self.oracle_backend}, Error: {e}"
            )
            raise RuntimeError(
                f"Oracle call failed: {e}. Check oracle availability and configuration."
            ) from e

        if state is not None and self.cache_measurements:
            if not isinstance(cached, dict):
                cached = {}
            cached[cache_key] = result
            state["oracle_cache"] = cached

        return result