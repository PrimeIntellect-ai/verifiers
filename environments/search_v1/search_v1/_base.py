"""Core of the v1 search/research taskset.

The composable ``search`` taskset family (QUEST / OpenSeeker / REDSearcher) ported
to one concrete ``vf.Taskset`` whose ``backend`` config field selects the dataset
and scoring strategy. (v1 env discovery wants exactly one exported ``Taskset``
subclass per env, so the three backends are strategies inside this class rather
than separate subclasses.)

Contract shared by every backend:

- the agent runs in a sandbox with web-search tooling (provided by the *harness*,
  e.g. ``rlm`` + the ``websearch`` / ``open_webpage`` skills — this taskset ships
  no tools and is harness-agnostic);
- the agent writes its final response to ``/task/answer.txt``;
- scoring reads that file out of the still-live runtime and grades it.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import verifiers.v1 as vf
from openai import AsyncOpenAI
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import setup_openai_client

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_FILE = "/task/answer.txt"
DEFAULT_WORKDIR = "/workspace"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"
DEFAULT_JUDGE_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_JUDGE_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-mini"

Backend = Literal["quest", "openseeker", "redsearcher"]


class SearchTask(vf.Task):
    """A single search/research task. Fields are a union over the backends;
    each backend's ``load_tasks`` populates the ones it uses."""

    question: str = ""
    """The natural-language research question shown to the agent."""
    answer: str = ""
    """Reference (gold) answer, when the backend has one. QUEST objective tasks
    grade against generated eval scripts instead and leave this empty."""
    # redsearcher
    difficulty: str = ""
    # openseeker
    number_of_tool_calls: int | None = None
    trajectory_correctness: Any = None
    # quest
    rl_task_category: str = ""
    task_id: str = ""
    reward_model: dict[str, Any] | None = None


class SearchConfig(vf.TasksetConfig):
    """Config for every search backend; ``backend`` selects which dataset/scoring
    strategy runs. Mirrors the kwargs the composable ``make_search_taskset(backend=...)``
    accepted, so the v1 port stays at parity with the v0 environment surface."""

    backend: Backend = "quest"

    # dataset
    dataset_name: str | None = None
    """HF dataset id; ``None`` uses the backend's default."""
    split: str | None = None
    filter_fn: str | None = None
    ds_keep_in_memory: bool | None = None
    ds_num_proc: int | None = None

    # answer / runtime layout
    answer_file: str = DEFAULT_ANSWER_FILE
    workdir: str = DEFAULT_WORKDIR

    # sandbox spec applied to every task
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE
    sandbox_cpu_cores: int = 2
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5

    # judge / scoring backend
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_base_url: str | None = DEFAULT_JUDGE_BASE_URL
    judge_api_key_var: str = DEFAULT_JUDGE_API_KEY_VAR
    judge_sampling_args: dict[str, Any] | None = None

    # openseeker
    include_trajectory: bool = False

    # redsearcher
    difficulty: str | None = None
    redsearcher_judge_max_retries: int = 5
    redsearcher_exact_match_shortcut: bool = True

    # quest
    category: str | None = None
    quest_cache_dir: str | None = None
    quest_eval_scripts_dir: str | None = None
    quest_eval_concurrency: int = 8


def completion_text_from_trace(trace: vf.Trace) -> str:
    """Concatenate the agent's assistant-message text — fallback answer when the
    agent stopped without writing the answer file."""
    parts: list[str] = []
    for message in trace.assistant_messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts).strip()


class SearchTaskset(vf.Taskset[SearchTask, SearchConfig]):
    """QUEST / OpenSeeker / REDSearcher research tasks, selected by ``config.backend``.

    Sandbox-backed (``NEEDS_CONTAINER``); the agent writes ``/task/answer.txt`` and
    the single ``@vf.reward`` reads it out of the live runtime and grades it via the
    selected backend strategy.
    """

    NEEDS_CONTAINER = True

    # shared judge client, created lazily and reused across rollouts
    _judge_client: AsyncOpenAI | None = None

    # --- task building ---

    def _task_defaults(self) -> dict[str, Any]:
        """Sandbox spec applied to every task built by a backend's ``load_tasks``."""
        return dict(
            image=self.config.sandbox_image,
            workdir=self.config.workdir,
            resources=vf.TaskResources(
                cpu=self.config.sandbox_cpu_cores,
                memory=self.config.sandbox_memory_gb,
                disk=self.config.sandbox_disk_size_gb,
            ),
        )

    def get_instruction(self, question: str) -> str:
        """Default user-prompt framing + the answer-file contract. Backends may
        override the framing via their own ``instruction`` helper."""
        return (
            f"{question}\n\n"
            f"When you have the final response, write it to {self.config.answer_file} "
            "using a tool call, then stop. The task is incomplete unless that file "
            "exists. Provide the requested answer directly, include supporting "
            "URLs/citations when useful, and do not include scratch reasoning or "
            "tool traces."
        )

    def load_tasks(self) -> list[SearchTask]:
        return self._backend().load_tasks(self)

    # --- runtime lifecycle ---

    async def setup(self, task: SearchTask, runtime: vf.Runtime) -> None:
        await runtime.run(["sh", "-c", f"mkdir -p {self.config.workdir} /task"], {})

    async def read_answer(self, runtime: vf.Runtime, trace: vf.Trace) -> tuple[str, str]:
        """Read the agent's answer from the answer file, falling back to the
        assistant transcript. Returns ``(answer, source)``."""
        result = await runtime.run(
            ["sh", "-c", f"cat {self.config.answer_file} 2>/dev/null || true"], {}
        )
        answer = (result.stdout or "").strip()
        if answer:
            return answer, "answer_file"
        fallback = completion_text_from_trace(trace)
        return fallback, ("completion_fallback" if fallback else "missing")

    @vf.reward(weight=1.0)
    async def answer_reward(
        self, task: SearchTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        answer, source = await self.read_answer(runtime, trace)
        trace.info["answer"] = answer
        trace.info["answer_source"] = source
        if not answer:
            trace.info["eval_error"] = "empty_answer"
            return 0.0
        # The composable rubrics swallowed scoring/judge failures to 0.0 so a
        # transient judge error never failed the rollout; preserve that.
        try:
            return await self._backend().score(self, task, answer, trace)
        except Exception as exc:  # noqa: BLE001 - mirror v0 resilience
            logger.warning("search-v1 (%s) scoring failed: %r", self.config.backend, exc)
            trace.info["eval_error"] = f"scoring_failed: {exc!r}"
            return 0.0

    # --- backend strategy dispatch ---

    def _backend(self):
        backend = self.config.backend
        if backend == "openseeker":
            from search_v1 import openseeker

            return openseeker
        if backend == "redsearcher":
            from search_v1 import redsearcher

            return redsearcher
        if backend == "quest":
            from search_v1.quest import quest

            return quest
        raise ValueError(
            f"Unknown search backend: {backend!r}. "
            "Expected one of: quest, openseeker, redsearcher."
        )

    # --- shared judge client (OpenAI-compatible, routed to judge_base_url) ---

    def judge_client(self) -> AsyncOpenAI:
        if self._judge_client is None:
            self._judge_client = setup_openai_client(
                ClientConfig(
                    api_key_var=self.config.judge_api_key_var,
                    api_base_url=self.config.judge_base_url or "https://api.openai.com/v1",
                    timeout=1200.0,
                )
            )
        return self._judge_client
