"""REDSearcher composable search taskset.

This ports the released REDSearcher text RL query set into the composable
search taskset family. The public artifact is a simple QA dataset, so scoring
uses the paper/repo's answer-matching LLM-as-judge convention rather than
dataset-provided verifier scripts.
"""

import asyncio
import logging
import os
import re
import unicodedata
from typing import Any, NoReturn

import verifiers as vf
from verifiers.utils.sandbox_delete import delete_sandbox_for_rollout
from datasets import Dataset, load_dataset
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import setup_openai_client

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "Zchu/REDSearcher_RL_1K"
DEFAULT_SPLIT = "train"
DEFAULT_ANSWER_FILE = "/task/answer.txt"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_JUDGE_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_JUDGE_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-mini"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"

# Matches DeepTraceHub's released BROWSECOMP judge prompt, the closest public
# reference for REDSearcher's RL reward while the RL trainer remains unreleased.
_JUDGE_PROMPT = """\
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e. just return A or B, without adding any other text.
"""

_CONTEXT_LENGTH_ERROR_PHRASES = (
    "this model's maximum context length is",
    "is longer than the model's context length",
    "is longer than the maximum model length",
    "exceeds the model's context length",
    "exceed the configured limit",
    "exceeds the configured limit",
    "exceeded model",
    "prompt_too_long",
    "context length",
    "maximum model length",
)

_REDSEARCHER_JUDGE_ERROR_TYPES = (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

_REDSEARCHER_TRANSIENT_JUDGE_ERROR_TYPES = (
    APIConnectionError,
    APITimeoutError,
    ConflictError,
    InternalServerError,
    RateLimitError,
)


def _is_context_length_error(exc: BadRequestError) -> bool:
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", "") or ""
    error_text = f"{response_text}\n{exc}".lower()
    return any(phrase in error_text for phrase in _CONTEXT_LENGTH_ERROR_PHRASES)


def _raise_redsearcher_judge_error(exc: Exception, *, model: str) -> NoReturn:
    if isinstance(exc, BadRequestError) and _is_context_length_error(exc):
        raise vf.OverlongPromptError(
            f"REDSearcher judge prompt exceeded model context for {model}: {exc}"
        ) from exc
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            ConflictError,
        ),
    ):
        raise vf.InfraError(
            f"REDSearcher judge transient request failed for {model}: {exc}"
        ) from exc
    if isinstance(exc, APIResponseValidationError):
        raise vf.InvalidModelResponseError(
            f"REDSearcher judge SDK response validation failed for {model}: {exc}"
        ) from exc
    if isinstance(exc, LengthFinishReasonError):
        raise vf.InvalidModelResponseError(
            f"REDSearcher judge stopped due to length for {model}: {exc}"
        ) from exc
    if isinstance(
        exc,
        (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            UnprocessableEntityError,
            ContentFilterFinishReasonError,
            BadRequestError,
            APIStatusError,
        ),
    ):
        raise vf.ModelError(
            f"REDSearcher judge request failed for {model}: {exc}"
        ) from exc
    raise AssertionError(
        f"Unhandled REDSearcher judge exception type: {type(exc).__name__}"
    ) from exc


def _completion_text(state: vf.State) -> str:
    completion = state.get("completion")
    if isinstance(completion, str):
        return completion.strip()
    if not isinstance(completion, list):
        return ""
    parts: list[str] = []
    for message in completion:
        role = None
        content = None
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role == "assistant" and isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts).strip()


def _normalize_for_match(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _exact_answer_match(*, response: str, answer: str) -> bool:
    normalized_answer = _normalize_for_match(answer)
    normalized_response = _normalize_for_match(response)
    if not normalized_answer or not normalized_response:
        return False
    return normalized_answer == normalized_response


def _parse_judge_choice(content: str) -> float | None:
    text = content.strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip("`*_ \t")
    upper = first_line.upper()
    if re.match(r"^\[?INCORRECT\]?(?:[\s.):\]-]|$)", upper) or re.match(
        r"^B(?:[\s.):\]-]|$)", upper
    ):
        return 0.0
    if re.match(r"^\[?CORRECT\]?(?:[\s.):\]-]|$)", upper) or re.match(
        r"^A(?:[\s.):\]-]|$)", upper
    ):
        return 1.0
    return None


class RedSearcherTaskSet(SandboxTaskSet):
    """REDSearcher text RL deep-search taskset."""

    default_workdir = DEFAULT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        filter_fn: str | None = None,
        ds_keep_in_memory: bool | None = True,
        ds_num_proc: int | None = None,
        sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
        sandbox_cpu_cores: int = 2,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int | None = None,
        answer_file: str = DEFAULT_ANSWER_FILE,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
        judge_max_retries: int = 5,
        use_exact_match_shortcut: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.ds_keep_in_memory = ds_keep_in_memory
        self.ds_num_proc = ds_num_proc
        self.answer_file = answer_file
        self._sandbox_spec = SandboxSpec(
            image=sandbox_image,
            cpu_cores=sandbox_cpu_cores,
            memory_gb=sandbox_memory_gb,
            disk_size_gb=sandbox_disk_size_gb,
            timeout_minutes=sandbox_timeout_minutes,
        )
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._judge_api_key_var = judge_api_key_var
        self._judge_sampling_args = dict(judge_sampling_args or {})
        self._judge_max_retries = judge_max_retries
        self._use_exact_match_shortcut = use_exact_match_shortcut
        super().__init__(
            dataset=self._build_dataset,
            name="search/redsearcher",
            filter_fn=filter_fn,
        )

    def _build_dataset(self) -> Dataset:
        raw = load_dataset(
            self.dataset_name,
            split=self.split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(raw):
            difficulty = str(row.get("difficulty") or "")
            question = str(row.get("problem") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                continue
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "info": {
                        "question": question,
                        "problem": question,
                        "answer": answer,
                        "difficulty": difficulty,
                        "dataset_name": self.dataset_name,
                        "split": self.split,
                        "row_index": idx,
                        "answer_file": self.answer_file,
                    },
                }
            )
        return Dataset.from_list(rows)

    def get_instruction(self, info: dict) -> str:
        question = str(info.get("question") or "")
        return (
            f"{question}\n\n"
            "This is a REDSearcher long-horizon search task. Break the problem into search subgoals, "
            "cross-check the answer across sources, and synthesize a concise final response.\n\n"
            f"When you have the final response, write it to {self.answer_file} using a tool call, "
            "then stop. The task is incomplete unless that file exists. Include the requested answer "
            "and supporting URLs/citations in the file, but do not include scratch reasoning or tool traces."
        )

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return self._sandbox_spec

    def get_workdir(self, info: dict) -> str:
        return self.default_workdir

    def get_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        for key in ("SERPER_API_KEY",):
            value = os.environ.get(key)
            if value:
                env_vars[key] = value
        return env_vars

    async def setup(self, state: vf.State) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {self.default_workdir} /task", timeout=10
        )

    def get_rubric(self) -> vf.Rubric:
        return RedSearcherRubric(
            answer_file=self.answer_file,
            judge_model=self._judge_model,
            judge_base_url=self._judge_base_url,
            judge_api_key_var=self._judge_api_key_var,
            judge_sampling_args=self._judge_sampling_args,
            judge_max_retries=self._judge_max_retries,
            use_exact_match_shortcut=self._use_exact_match_shortcut,
        )


class RedSearcherRubric(vf.Rubric):
    """Scores REDSearcher answers against the released ground-truth label."""

    def __init__(
        self,
        *,
        answer_file: str = DEFAULT_ANSWER_FILE,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
        judge_max_retries: int = 5,
        use_exact_match_shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.answer_file = answer_file
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key_var = judge_api_key_var
        self.judge_sampling_args = dict(judge_sampling_args or {})
        self.judge_max_retries = judge_max_retries
        self.use_exact_match_shortcut = use_exact_match_shortcut
        self._client: AsyncOpenAI | None = None
        self.add_reward_func(self.answer_reward, weight=1.0)

    async def _answer_score_for_state(self, state: vf.State) -> float:
        existing_error = state.get("error")
        if existing_error is not None:
            state["redsearcher_agent_error"] = repr(existing_error)
            return 0.0
        try:
            return await self.answer_reward(state)
        except vf.Error as exc:
            state["error"] = exc
            return 0.0

    async def score_rollout(self, state: vf.State) -> None:
        """Score one rollout and preserve judge failures as ``vf.Error`` values."""
        score = await self._answer_score_for_state(state)
        state["reward"] = score
        state["metrics"] = {"answer_reward": score}

    async def score_group(self, states: list[vf.State]) -> None:
        """Score rollouts while preserving judge failures as ``vf.Error`` values."""
        if not states:
            logger.warning("No states to score")
            return
        scores = await asyncio.gather(
            *(self._answer_score_for_state(state) for state in states)
        )
        avg_score = sum(scores) / len(scores)
        for state, score in zip(states, scores):
            state["reward"] = score
            state["advantage"] = score - avg_score
            for turn in state.get("trajectory", []):
                if isinstance(turn, dict):
                    if turn.get("advantage") is None:
                        turn["advantage"] = state["advantage"]
                    if turn.get("reward") is None:
                        turn["reward"] = state["reward"]
            state["metrics"] = {"answer_reward": score}

    async def answer_reward(self, state: vf.State, **_: Any) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            raise vf.SandboxError("REDSearcher scoring requires a live sandbox")
        try:
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"cat {self.answer_file} 2>/dev/null || true",
                working_dir=None,
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to read REDSearcher answer file {self.answer_file}"
            ) from exc
        response = (result.stdout or "").strip()
        answer_source = "answer_file"
        if not response:
            response = _completion_text(state)
            answer_source = "completion_fallback" if response else "missing"
        state["redsearcher_answer"] = response
        state["redsearcher_answer_source"] = answer_source
        if not response:
            state["redsearcher_eval_error"] = "empty_answer"
            return 0.0
        info = state.get("info") or {}
        question = str(info.get("question") or info.get("problem") or "")
        answer = str(state.get("answer") or info.get("answer") or "").strip()
        if not answer:
            raise vf.InfraError(
                "REDSearcher task is missing ground-truth answer metadata"
            )
        state["redsearcher_ground_truth"] = answer
        if self.use_exact_match_shortcut and _exact_answer_match(
            response=response, answer=answer
        ):
            state["redsearcher_match_method"] = "exact_match"
            state["redsearcher_judge_result"] = {
                "correct": "yes",
                "accuracy": 1.0,
                "reasoning": "Exact normalized answer match.",
            }
            return 1.0
        score = await self._judge_answer(
            question=question,
            response=response,
            answer=answer,
            state=state,
        )
        state["redsearcher_match_method"] = "llm_judge"
        return score

    async def _judge_answer(
        self,
        *,
        question: str,
        response: str,
        answer: str,
        state: vf.State,
    ) -> float:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            response=response,
            correct_answer=answer,
        )
        client = self._get_client()
        request_kwargs = dict(self.judge_sampling_args)
        last_content = ""
        max_attempts = max(1, self.judge_max_retries)
        for attempt in range(max_attempts):
            try:
                judge_response = await client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    **request_kwargs,
                )
            except _REDSEARCHER_JUDGE_ERROR_TYPES as exc:
                if isinstance(exc, _REDSEARCHER_TRANSIENT_JUDGE_ERROR_TYPES) and (
                    attempt + 1 < max_attempts
                ):
                    logger.warning(
                        "REDSearcher judge transient request failed on attempt %s/%s: %r",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    continue
                _raise_redsearcher_judge_error(exc, model=self.judge_model)
            choices = getattr(judge_response, "choices", None)
            if choices is None or len(choices) != 1:
                last_content = (
                    f"invalid choice count: {0 if choices is None else len(choices)}"
                )
            else:
                content = choices[0].message.content
                last_content = content or ""
                parsed = _parse_judge_choice(last_content)
                if parsed is not None:
                    state["redsearcher_judge_response"] = last_content
                    state["redsearcher_judge_result"] = {
                        "correct": "yes" if parsed == 1.0 else "no",
                        "accuracy": parsed,
                    }
                    return parsed
            logger.warning(
                "Failed to parse REDSearcher judge response on attempt %s/%s: %r",
                attempt + 1,
                max_attempts,
                last_content[:200],
            )
        raise vf.InvalidModelResponseError(
            f"REDSearcher judge response was not parseable as A/B: {last_content!r}"
        )

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        api_base_url = self.judge_base_url or "https://api.openai.com/v1"
        self._client = setup_openai_client(
            ClientConfig(
                api_key_var=self.judge_api_key_var,
                api_base_url=api_base_url,
                timeout=1200.0,
            )
        )
        return self._client

    @vf.cleanup
    async def cleanup(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            await delete_sandbox_for_rollout(sandbox_client, sandbox_id)

    @vf.teardown
    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
