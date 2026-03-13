import inspect
import logging
from typing import Any, Callable

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State
from verifiers.utils.async_utils import maybe_await


class OracleRubric(Rubric):
    """Scores generations with an external oracle (model/engine endpoint).
    
    Parallel to JudgeRubric but with flexible backend support for any model,
    API, or inference engine. Scores by comparing oracle output against either:
    - A ground truth value/target (from answer)
    - A threshold (improvement-based or absolute)
    """

    def __init__(
        self,
        oracle: Any = None,
        parser: Parser | None = None,
        oracle_fn: Callable[..., Any] | None = None,
        backend_caller: Callable[..., Any] | None = None,
        oracle_input_fn: Callable[..., Any] | None = None,
        process_output: Callable[..., Any] | None = None,
        property_extractor: Callable[..., Any] | None = None,
        score_function: Callable[..., bool | float] | None = None,
        comparator: Callable[..., bool | float] | None = None,
        target_extractor: Callable[[Any], Any] | None = None,
        threshold_extractor: Callable[[Any], float | None] | None = None,
        expose_oracle_property_metric: bool = False,
        cache_measurements: bool = True,
    ):
        super().__init__(parser=parser)
        if oracle_fn is not None and backend_caller is not None:
            raise ValueError(
                "Provide either `oracle_fn` or `backend_caller`, not both."
            )
        if score_function is not None and comparator is not None:
            raise ValueError(
                "Provide either `score_function` or `comparator`, not both."
            )

        self.oracle_backend = oracle
        self.oracle_fn = oracle_fn or backend_caller
        self.oracle_input_fn = oracle_input_fn or self._default_oracle_input
        self.process_output = process_output
        self.property_extractor = property_extractor
        self._score_function = score_function or comparator or self._default_comparator
        self._score_metric_name = self._resolve_score_metric_name(
            score_function=score_function,
            comparator=comparator,
        )
        self.comparator = self._score_function
        if score_function is not None and comparator is None:
            # In direct score_function mode, extractor hooks are optional.
            self.target_extractor = target_extractor
            self.threshold_extractor = threshold_extractor
        else:
            # Backward-compatible comparator mode keeps old defaults.
            self.target_extractor = target_extractor or self._default_target_extractor
            self.threshold_extractor = (
                threshold_extractor or self._default_threshold_extractor
            )
        self.expose_oracle_property_metric = expose_oracle_property_metric
        self.cache_measurements = cache_measurements
        self.logger = logging.getLogger(__name__)

        if self.expose_oracle_property_metric:
            self.add_metric(self.oracle_property)
        self._score_reward_entry = self._make_score_reward_entry()
        self.add_reward_func(self._score_reward_entry)

        self.class_objects = {
            "parser": self.parser,
            "oracle": self.oracle_backend,
            "oracle_backend": self.oracle_backend,
            "oracle_fn": self.oracle_fn,
            "oracle_input_fn": self.oracle_input_fn,
            "process_output": self.process_output,
            "property_extractor": self.property_extractor,
            "score_function": self._score_function,
            "score_metric_name": self._score_metric_name,
            "comparator": self._score_function,
            "target_extractor": self.target_extractor,
            "threshold_extractor": self.threshold_extractor,
            "expose_oracle_property_metric": self.expose_oracle_property_metric,
        }

    def _resolve_score_metric_name(
        self,
        score_function: Callable[..., Any] | None,
        comparator: Callable[..., Any] | None,
    ) -> str:
        if score_function is not None:
            name = getattr(score_function, "__name__", "")
            if name and not name.startswith("<"):
                return name
        if comparator is not None:
            return "score_function"
        return "score_function"

    def _make_score_reward_entry(self) -> Callable[..., Any]:
        async def _score_reward_entry(
            prompt: Messages,
            completion: Messages,
            answer: Any,
            state: State | None = None,
            **kwargs,
        ) -> float:
            return await self.score_function(
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
                **kwargs,
            )

        _score_reward_entry.__name__ = self._score_metric_name
        return _score_reward_entry

    def _default_oracle_input(
        self,
        response: str,
        **kwargs,
    ) -> str:
        return response

    def _default_property_extractor(
        self,
        oracle_output: Any,
        **kwargs,
    ) -> Any:
        return oracle_output

    def _default_target_extractor(self, answer: Any) -> Any:
        if isinstance(answer, dict):
            if "target" in answer:
                return answer["target"]
            if "answer" in answer:
                return answer["answer"]
            return None
        return answer

    def _default_threshold_extractor(self, answer: Any) -> float | None:
        if isinstance(answer, dict) and ("threshold" in answer or "tolerance" in answer):
            threshold = answer.get("threshold", answer.get("tolerance"))
            return None if threshold is None else float(threshold)
        return None

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

    async def oracle(
        self,
        response: str,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
    ) -> Any:
        """Call oracle and return raw output.
        
        This is the main inference point in OracleRubric, parallel to judge() in JudgeRubric.
        Prepares input, invokes oracle backend, handles caching and errors.
        
        Args:
            response: Parsed response from completion
            prompt: Original prompt/question
            completion: Full completion from model
            answer: Expected answer (may contain target/threshold)
            state: Current state (for caching)
            
        Returns:
            Raw oracle output (type depends on oracle backend)
            
        Raises:
            RuntimeError: On oracle invocation failure
        """
        # Prepare oracle input
        oracle_input = await self._call_with_supported_kwargs(
            self.oracle_input_fn,
            response=response,
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
        )

        # Check cache
        # TODO: Harden cache key for multi-prediction servers by including normalized
        # oracle_input (and optionally a caller/extractor discriminator) so different
        # prediction types do not collide when prompt/response/answer are similar.
        cache_key = self._cache_key(prompt, response, answer)
        cached = state.get("oracle_cache") if state else None
        if (
            self.cache_measurements
            and isinstance(cached, dict)
            and cache_key in cached
        ):
            return cached[cache_key]

        # Invoke oracle with error handling
        try:
            oracle_output = await self._invoke_oracle_backend(
                oracle_input=oracle_input,
                response=response,
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )
            if self.process_output is not None:
                oracle_output = await self._call_with_supported_kwargs(
                    self.process_output,
                    oracle_output=oracle_output,
                    oracle=self.oracle_backend,
                    oracle_backend=self.oracle_backend,
                    oracle_input=oracle_input,
                    response=response,
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                )
        except Exception as e:
            self.logger.warning(
                f"Oracle invocation failed. Oracle: {self.oracle_backend}, Error: {str(e)}"
            )
            raise RuntimeError(
                f"Oracle invocation failed. Check oracle availability and configuration. "
                f"Oracle: {self.oracle_backend}, Error: {str(e)}"
            ) from e

        # Cache result
        if state is not None and self.cache_measurements:
            if not isinstance(cached, dict):
                cached = {}
            cached[cache_key] = oracle_output
            state["oracle_cache"] = cached

        return oracle_output

    async def _invoke_oracle_backend(
        self,
        oracle_input: Any,
        response: str,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
    ) -> Any:
        """Invoke the oracle backend (model, API, or engine).
        
        Supports multiple oracle types:
        - Callable with oracle_fn wrapper
        - Objects with .predict() method (sklearn, torch, etc.)
        - Async callables
        """
        if self.oracle_fn is not None:
            return await self._call_with_supported_kwargs(
                self.oracle_fn,
                oracle=self.oracle_backend,
                oracle_backend=self.oracle_backend,
                oracle_input=oracle_input,
                response=response,
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )

        if hasattr(self.oracle_backend, "predict") and callable(self.oracle_backend.predict):
            return await maybe_await(self.oracle_backend.predict, oracle_input)

        if callable(self.oracle_backend):
            return await maybe_await(self.oracle_backend, oracle_input)

        raise TypeError(
            "OracleRubric requires `oracle` to be callable, expose `.predict()`, or be paired with `oracle_fn`."
        )

    def _coerce_number(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _default_comparator(
        self,
        property_value: Any,
        target: Any,
        threshold: float | None,
        **kwargs,
    ) -> bool:
        if threshold is not None:
            property_number = self._coerce_number(property_value)
            target_number = self._coerce_number(target)

            if target not in (None, ""):
                if property_number is not None and target_number is not None:
                    return abs(property_number - target_number) <= threshold
                return property_value == target

            if property_number is None:
                raise TypeError(
                    "Threshold-only oracle comparisons require a numeric property value."
                )
            return property_number >= threshold

        if target in (None, ""):
            raise ValueError(
                "OracleRubric needs either an answer/target or a threshold to score the rollout."
            )

        property_number = self._coerce_number(property_value)
        target_number = self._coerce_number(target)
        if property_number is not None and target_number is not None:
            return property_number == target_number

        return property_value == target

    def _cache_key(self, prompt: Messages, response: str, answer: Any) -> str:
        return repr((prompt, response, answer))

    async def measure_property(
        self,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
    ) -> tuple[Any, Any]:
        """Measure oracle property by invoking oracle and extracting result.
        
        Uses the oracle() method as the inference point, then extracts the
        relevant property from the oracle output.
        """
        response = self.parser.parse_answer(completion) or ""

        # Invoke oracle (handles caching internally)
        oracle_output = await self.oracle(
            response=response,
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
        )

        # If oracle_fn returns a direct value (float/bool/etc.), pass it through.
        # property_extractor is only applied for dict-shaped server responses.
        if isinstance(oracle_output, dict) and self.property_extractor is not None:
            property_value = await self._call_with_supported_kwargs(
                self.property_extractor,
                oracle_output=oracle_output,
                oracle=self.oracle_backend,
                oracle_backend=self.oracle_backend,
                response=response,
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )
        else:
            property_value = oracle_output

        if state is not None:
            state["oracle_response"] = oracle_output
            state["oracle_property_value"] = property_value

        return property_value, oracle_output

    async def oracle_property(
        self,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
        **kwargs,
    ) -> float:
        property_value, _ = await self.measure_property(prompt, completion, answer, state)
        if isinstance(property_value, bool):
            return float(property_value)
        if isinstance(property_value, (int, float)):
            return float(property_value)
        return 0.0

    async def score_function(
        self,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
        **kwargs,
    ) -> float:
        property_value, oracle_output = await self.measure_property(
            prompt,
            completion,
            answer,
            state,
        )
        target = self.target_extractor(answer) if callable(self.target_extractor) else None
        threshold = (
            self.threshold_extractor(answer)
            if callable(self.threshold_extractor)
            else None
        )
        result = await self._call_with_supported_kwargs(
            self._score_function,
            property_value=property_value,
            oracle_output=oracle_output,
            target=target,
            threshold=threshold,
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
            **kwargs,
        )
        score = float(result)
        if state is not None:
            state["oracle_target"] = target
            state["oracle_threshold"] = threshold
            state["oracle_match"] = bool(result)
            state["oracle_score"] = score
        return score

    async def correct_answer(
        self,
        prompt: Messages,
        completion: Messages,
        answer: Any,
        state: State | None = None,
        **kwargs,
    ) -> float:
        """Backward-compatible alias for score_function."""
        return await self.score_function(
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
            **kwargs,
        )