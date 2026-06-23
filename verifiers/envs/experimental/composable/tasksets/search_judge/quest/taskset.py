"""QUEST composable search taskset.

This ports QUEST objective tasks into the verifiers composable taskset
structure while preserving the generated QUEST eval scripts and objective
verification tree runtime.
"""

import asyncio
import ast
import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, NoReturn

import verifiers as vf
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
from pydantic import BaseModel
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import setup_openai_client

from .obj_task_eval.utils.cache_filesys import CacheFileSys
from .obj_task_eval.utils.load_eval_script import load_eval_script
from .open_ended import score_open_ended_answer

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "osunlp/QUEST-RL-Data"
DEFAULT_SPLIT = "train"
DEFAULT_CATEGORY = "objective"
DEFAULT_ANSWER_FILE = "/task/answer.txt"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_JUDGE_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_JUDGE_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-mini"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"

_EVAL_SCRIPTS_ROOT_CACHE: dict[str, Path] = {}


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


def _is_context_length_error(exc: BadRequestError) -> bool:
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", "") or ""
    error_text = f"{response_text}\n{exc}".lower()
    return any(phrase in error_text for phrase in _CONTEXT_LENGTH_ERROR_PHRASES)


_QUEST_JUDGE_ERROR_TYPES = (
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


def _single_choice(response: Any, *, context: str) -> Any:
    if response is None:
        raise vf.EmptyModelResponseError(f"QUEST judge returned no {context} response")
    choices = getattr(response, "choices", None)
    if choices is None:
        raise vf.EmptyModelResponseError(
            f"QUEST judge returned no {context} response choices"
        )
    if len(choices) != 1:
        raise vf.InvalidModelResponseError(
            f"QUEST judge returned {len(choices)} {context} choices, expected 1"
        )
    return choices[0]


def _raise_quest_judge_error(exc: Exception, *, model: str) -> NoReturn:
    if isinstance(exc, BadRequestError) and _is_context_length_error(exc):
        raise vf.OverlongPromptError(
            f"QUEST judge prompt exceeded model context for {model}: {exc}"
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
            f"QUEST judge transient request failed for {model}: {exc}"
        ) from exc
    if isinstance(exc, APIResponseValidationError):
        raise vf.InvalidModelResponseError(
            f"QUEST judge SDK response validation failed for {model}: {exc}"
        ) from exc
    if isinstance(exc, LengthFinishReasonError):
        raise vf.InvalidModelResponseError(
            f"QUEST judge stopped due to length for {model}: {exc}"
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
        raise vf.ModelError(f"QUEST judge request failed for {model}: {exc}") from exc
    raise AssertionError(
        f"Unhandled QUEST judge exception type: {type(exc).__name__}"
    ) from exc


class QuestOpenAIClient:
    """OpenAI-compatible client adapter for QUEST's ``async_response`` API."""

    provider = "openai"

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.sampling_args = dict(sampling_args or {})
        self._client = client

    async def async_response(self, *, count_token: bool = False, **kwargs: Any) -> Any:
        response_format = kwargs.pop("response_format", None)
        messages = kwargs.pop("messages")
        model = kwargs.pop("model", self.model) or self.model
        request_kwargs = dict(self.sampling_args)
        request_kwargs.update(kwargs)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            try:
                response = await self._client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    **request_kwargs,
                )
            except _QUEST_JUDGE_ERROR_TYPES as exc:
                _raise_quest_judge_error(exc, model=model)
            choice = _single_choice(response, context="structured")
            parsed = choice.message.parsed
            if parsed is None:
                raise vf.InvalidModelResponseError(
                    f"QUEST judge returned no parsed structured response for {model}"
                )
            usage = _usage_dict(response)
            return (parsed, usage) if count_token else parsed
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **request_kwargs,
            )
        except _QUEST_JUDGE_ERROR_TYPES as exc:
            _raise_quest_judge_error(exc, model=model)
        choice = _single_choice(response, context="text")
        content = choice.message.content
        if content is None:
            raise vf.EmptyModelResponseError(
                f"QUEST judge returned no text content for {model}"
            )
        usage = _usage_dict(response)
        return (content, usage) if count_token else content

    async def close(self) -> None:
        await self._client.close()


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _parse_ast_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _parse_ast_literal(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_parse_ast_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_parse_ast_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _parse_ast_literal(key): _parse_ast_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _parse_ast_literal(node.operand)
        if isinstance(operand, int | float):
            return -operand
    if isinstance(node, ast.Name) and node.id == "object":
        return object
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "array" and len(node.args) == 1:
            return _parse_ast_literal(node.args[0])
    raise ValueError(f"Unsupported QUEST literal syntax: {ast.dump(node)}")


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return _parse_ast_literal(ast.parse(value, mode="eval"))
        except Exception:
            return value


def _extract_question(prompt: Any, extra_info: Any) -> str:
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
    if isinstance(extra_info, dict) and isinstance(extra_info.get("question"), str):
        return extra_info["question"]
    return ""


def _extract_task_id(reward_model: Any, extra_info: Any) -> str | None:
    if isinstance(reward_model, dict):
        task_id = reward_model.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            task_id = ground_truth.get("task_id")
            if isinstance(task_id, str) and task_id:
                return task_id
    if isinstance(extra_info, dict):
        task_id = extra_info.get("original_task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
    return None


def _safe_module_component(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")[:80]
    return f"{stem}_{digest}" if stem else digest


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


def _load_eval_script(script_path: Path) -> Any:
    return load_eval_script(str(script_path))


def _normalize_eval_scripts_root(path: Path) -> Path:
    root = path.expanduser()
    if root.name == "eval_scripts" and root.is_dir():
        return root.parent
    if (root / "eval_scripts").is_dir():
        return root
    raise ValueError(
        f"QUEST eval scripts directory must contain eval_scripts/*.py: {root}"
    )


def _resolve_eval_scripts_root(
    *,
    dataset_name: str,
    eval_scripts_dir: str | None,
) -> Path:
    if eval_scripts_dir is not None:
        return _normalize_eval_scripts_root(Path(eval_scripts_dir))

    cached = _EVAL_SCRIPTS_ROOT_CACHE.get(dataset_name)
    if cached is not None:
        return cached

    try:
        from huggingface_hub import snapshot_download

        try:
            root = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=["eval_scripts/*.py"],
                local_files_only=True,
            )
        except Exception:
            root = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=["eval_scripts/*.py"],
            )
    except Exception as exc:
        raise vf.InfraError(
            f"Failed to resolve QUEST eval scripts from {dataset_name}"
        ) from exc

    scripts_root = _normalize_eval_scripts_root(Path(root))
    _EVAL_SCRIPTS_ROOT_CACHE[dataset_name] = scripts_root
    return scripts_root


class QuestTaskSet(SandboxTaskSet):
    """QUEST search/research taskset."""

    default_workdir = DEFAULT_WORKDIR

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        category: str = DEFAULT_CATEGORY,
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
        quest_cache_dir: str | None = None,
        quest_eval_scripts_dir: str | None = None,
        quest_eval_concurrency: int = 8,
    ) -> None:
        if category not in {"objective", "open-ended", "all"}:
            raise ValueError(
                "category must be one of 'objective', 'open-ended', or 'all'"
            )
        self.dataset_name = dataset_name
        self.split = split
        self.category = category
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
        self._quest_cache_dir = quest_cache_dir
        self._quest_eval_scripts_root = (
            None
            if category == "open-ended" and quest_eval_scripts_dir is None
            else _resolve_eval_scripts_root(
                dataset_name=dataset_name,
                eval_scripts_dir=quest_eval_scripts_dir,
            )
        )
        self._quest_eval_concurrency = quest_eval_concurrency
        super().__init__(
            dataset=self._build_dataset,
            name=f"search_judge/quest/{category}",
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
        for row_index, row in enumerate(raw):
            row_category = row.get("rl_task_category")
            if self.category != "all" and row_category != self.category:
                continue
            if row_category not in {"objective", "open-ended"}:
                raise ValueError(
                    f"Unsupported QUEST row category at dataset index {row_index}: "
                    f"{row_category!r}"
                )
            extra_info = _parse_literal(row.get("extra_info"))
            reward_model = _parse_literal(row.get("reward_model"))
            question = _extract_question(row.get("prompt"), extra_info)
            task_id = _extract_task_id(reward_model, extra_info)
            if not task_id:
                raise ValueError(
                    f"QUEST {row_category} row is missing task_id metadata "
                    f"at dataset index {row_index}"
                )
            if row_category == "open-ended" and not isinstance(reward_model, dict):
                raise ValueError(
                    "QUEST open-ended row has invalid reward_model metadata "
                    f"at dataset index {row_index}"
                )
            rows.append(
                {
                    "question": question,
                    "answer": "",
                    "info": {
                        "question": question,
                        "raw_prompt": row.get("prompt"),
                        "data_source": row.get("data_source"),
                        "rl_task_category": row.get("rl_task_category"),
                        "task_id": task_id,
                        "reward_model": reward_model,
                        "reward_model_raw": row.get("reward_model"),
                        "extra_info": extra_info,
                        "extra_info_raw": row.get("extra_info"),
                        "answer_file": self.answer_file,
                    },
                }
            )
        return Dataset.from_list(rows)

    def get_instruction(self, info: dict) -> str:
        question = str(info.get("question") or "")
        return (
            f"{question}\n\n"
            f"When you have the final response, write it to {self.answer_file} using a tool call, "
            "then stop. The task is incomplete unless that file exists. Include supporting URLs/citations "
            "in the file and do not include scratch reasoning or tool traces."
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
        return QuestRubric(
            answer_file=self.answer_file,
            dataset_name=self.dataset_name,
            eval_scripts_dir=(
                str(self._quest_eval_scripts_root)
                if self._quest_eval_scripts_root is not None
                else None
            ),
            cache_dir=self._quest_cache_dir,
            judge_model=self._judge_model,
            judge_base_url=self._judge_base_url,
            judge_api_key_var=self._judge_api_key_var,
            judge_sampling_args=self._judge_sampling_args,
            eval_concurrency=self._quest_eval_concurrency,
        )


class QuestRubric(vf.Rubric):
    """Scores QUEST objective and open-ended tasks."""

    def __init__(
        self,
        *,
        answer_file: str = DEFAULT_ANSWER_FILE,
        dataset_name: str = DEFAULT_DATASET_NAME,
        eval_scripts_dir: str | None = None,
        cache_dir: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL,
        judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR,
        judge_sampling_args: dict[str, Any] | None = None,
        eval_concurrency: int = 8,
    ) -> None:
        super().__init__()
        self.answer_file = answer_file
        self.dataset_name = dataset_name
        self.eval_scripts_dir = (
            Path(eval_scripts_dir).expanduser() if eval_scripts_dir else None
        )
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir
            else Path.home() / ".cache" / "verifiers" / "quest"
        )
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key_var = judge_api_key_var
        self.judge_sampling_args = dict(judge_sampling_args or {})
        self.eval_concurrency = eval_concurrency
        self._client: QuestOpenAIClient | None = None
        self._semaphore = asyncio.Semaphore(eval_concurrency)
        self._scripts_root: Path | None = None
        self.add_reward_func(self.quest_reward, weight=1.0)

    async def _quest_score_for_state(self, state: vf.State) -> float:
        if state.get("error") is not None:
            return 0.0
        try:
            return await self.quest_reward(state)
        except vf.Error as exc:
            state["error"] = exc
            return 0.0

    def _metric_name(self, state: vf.State) -> str:
        info = state.get("info") or {}
        if info.get("rl_task_category") == "open-ended":
            return "open_ended_reward"
        return "objective_reward"

    async def score_rollout(self, state: vf.State) -> None:
        """Score one rollout and preserve QUEST infrastructure failures as ``vf.Error`` values."""
        score = await self._quest_score_for_state(state)
        state["reward"] = score
        state["metrics"] = {"quest_reward": score, self._metric_name(state): score}

    async def score_group(self, states: list[vf.State]) -> None:
        """Score rollouts while preserving QUEST infrastructure failures as ``vf.Error`` values."""
        if not states:
            logger.warning("No states to score")
            return
        scores = await asyncio.gather(
            *(self._quest_score_for_state(state) for state in states)
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
            state["metrics"] = {"quest_reward": score, self._metric_name(state): score}

    async def quest_reward(self, state: vf.State, **_: Any) -> float:
        info = state.get("info") or {}
        if info.get("rl_task_category") == "open-ended":
            return await self.open_ended_reward(state)
        return await self.objective_reward(state)

    async def _read_answer(self, state: vf.State) -> tuple[str, str]:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            raise vf.SandboxError("QUEST scoring requires a live sandbox")
        try:
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"cat {self.answer_file} 2>/dev/null || true",
                working_dir=None,
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to read QUEST answer file {self.answer_file}"
            ) from exc
        answer = (result.stdout or "").strip()
        answer_source = "answer_file"
        if not answer:
            answer = _completion_text(state)
            answer_source = "completion_fallback" if answer else "missing"
        state["quest_answer"] = answer
        state["quest_answer_source"] = answer_source
        return answer, answer_source

    async def objective_reward(self, state: vf.State, **_: Any) -> float:
        answer, _ = await self._read_answer(state)
        if not answer:
            state["quest_eval_error"] = "empty_answer"
            return 0.0
        info = state.get("info") or {}
        task_id = info.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("QUEST objective task is missing task_id metadata")
        state["quest_task_id"] = task_id
        script_path = self._script_path(task_id)
        cache = CacheFileSys(str(self.cache_dir / _safe_module_component(task_id)))
        evaluate_answer = _load_eval_script(script_path)
        client = self._get_client()
        summary = await evaluate_answer(
            client=client,
            answer=answer,
            agent_name="rlm",
            answer_name=str(state.get("trajectory_id") or task_id),
            cache=cache,
            semaphore=self._semaphore,
            logger=logger,
            model=self.judge_model,
        )
        state["quest_eval_summary"] = summary
        final_score = float(summary.get("final_score", 0.0) or 0.0)
        if not math.isfinite(final_score):
            final_score = 0.0
        final_score = max(0.0, min(1.0, final_score))
        state["quest_final_score"] = final_score
        return final_score

    async def open_ended_reward(self, state: vf.State, **_: Any) -> float:
        answer, _ = await self._read_answer(state)
        if not answer:
            state["quest_eval_error"] = "empty_answer"
            return 0.0
        info = state.get("info") or {}
        task_id = info.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("QUEST open-ended task is missing task_id metadata")
        reward_model = info.get("reward_model")
        if not isinstance(reward_model, dict):
            raise ValueError("QUEST open-ended task is missing reward_model metadata")
        question = str(info.get("question") or "")
        state["quest_task_id"] = task_id
        client = self._get_client()
        summary = await score_open_ended_answer(
            client=client,
            model=self.judge_model,
            semaphore=self._semaphore,
            answer=answer,
            question=question,
            reward_model=reward_model,
        )
        state["quest_eval_summary"] = summary
        final_score = float(summary.get("final_score", 0.0) or 0.0)
        if not math.isfinite(final_score):
            final_score = 0.0
        final_score = max(0.0, min(1.0, final_score))
        state["quest_final_score"] = final_score
        state["quest_upstream_pairwise_score"] = summary.get("upstream_pairwise_score")
        return final_score

    def _get_client(self) -> QuestOpenAIClient:
        if self._client is not None:
            return self._client
        api_base_url = self.judge_base_url or "https://api.openai.com/v1"
        openai_client = setup_openai_client(
            ClientConfig(
                api_key_var=self.judge_api_key_var,
                api_base_url=api_base_url,
                timeout=1200.0,
            )
        )
        self._client = QuestOpenAIClient(
            client=openai_client,
            model=self.judge_model,
            sampling_args=self.judge_sampling_args,
        )
        return self._client

    def _script_path(self, task_id: str) -> Path:
        scripts_root = self._ensure_eval_scripts_dir()
        script_path = scripts_root / "eval_scripts" / f"{task_id}.py"
        if not script_path.is_file():
            raise FileNotFoundError(
                f"QUEST eval script not found for task_id={task_id!r}: {script_path}"
            )
        return script_path

    def _ensure_eval_scripts_dir(self) -> Path:
        if self._scripts_root is not None:
            return self._scripts_root
        if self.eval_scripts_dir is None:
            raise RuntimeError(
                "QUEST eval scripts root was not resolved before scoring"
            )
        self._scripts_root = _normalize_eval_scripts_root(self.eval_scripts_dir)
        return self._scripts_root

    @vf.cleanup
    async def cleanup(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass

    @vf.teardown
    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
