"""Native model calls for a RewardKit-authored rubric."""

import asyncio
import base64
import html
import io
import json
import mimetypes
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from openai import APITimeoutError

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.grading import scoring_network
from verifiers.v1.judge import Judge, JudgeView
from verifiers.v1.task import TaskData
from verifiers.v1.types import (
    ImageUrlContentPart,
    ImageUrlSource,
    SystemMessage,
    TextContentPart,
    UserMessage,
)
from verifiers.v1.utils.artifacts import ARTIFACTS_DIR
from verifiers.v1.utils.workspace import WorkspaceSnapshot

MAX_BYTES = 1024 * 1024
MAX_INPUT_BYTES = 32 * MAX_BYTES
MAX_INPUT_FILES = 256
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".msg",
    ".epub",
    ".html",
    ".htm",
}


class HarborJudgeConfig(JudgeConfig):
    view: JudgeView = "last_reply"
    """Native evidence selection, independent of the authored RewardKit rubric."""
    include_hidden_reasoning: bool = False
    """Include reasoning in full_trace; opaque provider state is always excluded."""


async def read_inputs(runtime, paths, *, cache=None, budget=None):
    if not paths:
        return []
    cache = {} if cache is None else cache
    budget = [MAX_INPUT_BYTES] if budget is None else budget
    roots = [str(PurePosixPath(runtime.config.workdir or "/app") / p) for p in paths]
    # Missing outputs are evidence for the judge, not infrastructure failures.
    # Bound the listing too, before any solver-controlled output reaches the host.
    result = await runtime.run(
        [
            "sh",
            "-c",
            f"find -L {shlex.join(roots)} -type f -print0 2>/dev/null | head -c {MAX_BYTES}",
        ],
        {},
    )
    if result.exit_code:
        raise TaskError(f"Cannot list judge inputs {paths!r}: {result.stderr[-500:]}")
    if len(result.stdout.encode()) >= MAX_BYTES:
        raise TaskError("Judge input listing exceeds the byte limit")
    parts = [TextContentPart(text=f"Requested files: {paths!r}")]
    if not result.stdout:
        parts.append(TextContentPart(text="No readable files found."))
    for path in sorted(set(result.stdout.split("\0")[:-1])):
        if path in cache:
            parts.extend(cache[path])
            continue
        if len(cache) >= MAX_INPUT_FILES or budget[0] <= 0:
            raise TaskError("Judge inputs exceed the aggregate file or byte limit")
        entry = [TextContentPart(text=f"File {path}:")]
        try:
            data = await runtime.read(path, max_bytes=min(MAX_BYTES, budget[0]))
        except SandboxError as exc:
            if "byte limit" not in str(exc):
                raise
            entry.append(
                TextContentPart(text="File omitted: exceeds input byte limit.")
            )
            cache[path] = entry
            parts.extend(entry)
            continue
        budget[0] -= len(data)
        mime = mimetypes.guess_type(path)[0] or ""
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            entry.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(
                        url=f"data:{mime};base64,{base64.b64encode(data).decode()}"
                    )
                )
            )
        else:
            if suffix in DOCUMENT_EXTENSIONS:
                try:
                    from markitdown import MarkItDown
                except ImportError as exc:
                    raise TaskError(
                        "Install the matching MarkItDown document extras in the evaluator project"
                    ) from exc
                converted = await asyncio.to_thread(
                    MarkItDown().convert_stream, io.BytesIO(data), file_extension=suffix
                )
                text = converted.text_content
            else:
                if mime.startswith(("image/", "audio/", "video/")) or b"\0" in data:
                    text = "File omitted: unsupported input format."
                else:
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = "File omitted: invalid UTF-8."
            if len(text.encode()) > MAX_BYTES:
                text = "File omitted: converted content exceeds input byte limit."
            budget[0] -= max(0, len(text.encode()) - len(data))
            if budget[0] < 0:
                raise TaskError("Judge inputs exceed the aggregate byte limit")
            entry.append(TextContentPart(text=text))
        cache[path] = entry
        parts.extend(entry)
    return parts


@dataclass
class PreparedJudge:
    reward: Any
    trace: Any
    path: str
    prefix: str
    authored: dict

    async def run(self) -> float:
        await self.reward.arun()
        value = self.reward.score
        self.trace.record_metrics(
            {f"{self.prefix}/{s.name}": s.value for s in self.reward.scores}
        )
        diagnostics = {f"criteria/{s.name}": s for s in self.reward.scores}
        guard = self.reward.guard_score
        if guard is not None:
            diagnostics["guard"] = guard
            if guard.error is None:
                self.trace.record_metric(
                    f"{self.prefix}/reliability/guard/flagged", guard.value > 0
                )
        for name, score in diagnostics.items():
            prefix = f"{self.prefix}/reliability/{name}"
            # RewardKit leaves sample diagnostics unset for a single judge sample.
            self.trace.record_metrics(
                {
                    f"{prefix}/{field}": getattr(score, field)
                    for field in ("agreement", "valid_samples")
                    if getattr(score, field) is not None
                }
            )
            self.trace.record_metric(f"{prefix}/error", score.error is not None)
        self.trace.info.setdefault("harbor_judge", {})[self.path] = {
            **self.reward.to_detail_dict(value),
            "authored": self.authored,
        }
        return value


async def prepare_judge(
    task,
    runtime,
    trace,
    path="/tests/judge.toml",
    *,
    raw=None,
    reward=None,
    prefix="harbor",
):
    from rewardkit.agents import AgentAttempt
    from rewardkit.models import AgentJudge, JudgeTomlConfig, LLMJudge

    from .rewardkit import NativeReward, load_rubric

    if raw is None:
        raw = tomllib.loads((await runtime.read(path, max_bytes=MAX_BYTES)).decode())
    config = JudgeTomlConfig.model_validate(raw)
    trace.info["authored_judging"] = True
    authored = config.judge
    unsupported = {
        "version",
        "cwd",
        "atif_trajectory",
        "mcp_servers",
    } & authored.model_fields_set
    if unsupported:
        raise TaskError(f"Unsupported native judge options: {sorted(unsupported)}")
    template = None
    if reward is None and authored.prompt_template is not None:
        source = PurePosixPath(path).parent / authored.prompt_template
        if source.suffix not in {".txt", ".md"}:
            raise TaskError("judge prompt_template must be .txt or .md")
        template = (await runtime.read(str(source), max_bytes=MAX_BYTES)).decode()
    if reward is None:
        reward = load_rubric(raw, path, runtime.config.workdir or "/app", template)
    agentic = isinstance(reward.judge, AgentJudge)
    if not agentic and not isinstance(reward.judge, LLMJudge):
        raise TaskError("JEV judging is not supported by the native judge backend")
    if agentic:
        if task.judge_executor is None:
            raise TaskError("Agent judge TOMLs require the agentic-judge environment")
        if task.config.judge is not None:
            raise TaskError("Configure agent judge overrides on env.judge")
    elif "model" in authored.model_fields_set:
        raise TaskError("judge.model is only supported for agent judges")
    if not agentic and "isolated" in authored.model_fields_set:
        raise TaskError("judge.isolated is only supported for agent judges")
    if authored.timeout <= 0:
        raise TaskError("judge.timeout must be positive")
    selections = {
        criterion.name: source.files
        if "files" in source.model_fields_set
        else authored.files
        for criterion, source in zip(reward.criteria, config.criterion, strict=True)
    }
    groups = {tuple(files) for files in selections.values()}
    combined = tuple(dict.fromkeys(f for files in selections.values() for f in files))
    if authored.mode != "individual":
        groups = {combined}
    elif authored.guard != "off":
        groups.add(combined)
    cache, budget = {}, [MAX_INPUT_BYTES]
    inputs = {
        paths: await read_inputs(runtime, paths, cache=cache, budget=budget)
        for paths in sorted(groups)
    }
    reference = (
        await read_inputs(runtime, [authored.reference], cache=cache, budget=budget)
        if authored.reference
        else []
    )
    if reference:
        reference.insert(
            0, TextContentPart(text="Reference solution (for comparison):")
        )
    workspace = None
    network = None
    if agentic:
        if any(not m.read_only for m in getattr(runtime.config, "mounts", {}).values()):
            raise TaskError("Independent judge workspaces require read-only mounts")
        # Capture after verifier staging/programmatic checks, before any judge runs.
        # Only these trees travel: the workdir, tests, verifier logs, published
        # artifacts and explicit inputs/reference. Other paths come from the image;
        # live processes and external services are not filesystem artifacts.
        workdir = PurePosixPath(runtime.config.workdir or "/app")
        roots = {
            workdir / path
            for path in (
                str(workdir),
                "/tests",
                "/logs/verifier",
                ARTIFACTS_DIR,
                *trace.state.artifacts,
                *(a.source for a in task.data.artifacts),
                *(p for paths in groups for p in paths),
                *([authored.reference] if authored.reference else []),
            )
        }
        if any(p == PurePosixPath("/") or ".." in p.parts for p in roots):
            raise TaskError("Judge workspace roots must be below '/' without '..'")
        workspace = await WorkspaceSnapshot.capture(runtime, roots)
        # Shared Harbor scoring can reopen the solver's network without changing
        # runtime.config. Fresh judges need that effective verifier policy.
        if task.scoring.mode == "shared":
            network = (
                scoring_network(trace.agent.config.runtime, task.scoring.network)
                if task.scoring.network is not None
                else NetworkPolicyConfig()
            )
    settings = (
        task.config.judge.model_dump(exclude_unset=True) if task.config.judge else {}
    )
    if reward.judge.model is not None:
        settings.setdefault("model", reward.judge.model)
    settings.setdefault("name", prefix)
    view = settings.pop("view", "last_reply")
    include_reasoning = settings.pop("include_hidden_reasoning", False)
    settings["sampling"] = {
        "reasoning_effort": reward.judge.reasoning_effort
    } | settings.get("sampling", {})
    judge = Judge(JudgeConfig(**settings))
    response = trace.last_reply
    if view == "full_trace":
        exclude = {"provider_state"}
        if not include_reasoning:
            exclude.add("reasoning_content")
        response = json.dumps(
            [
                m.model_dump(mode="json", exclude=exclude)
                for m in (trace.branches[-1].messages if trace.branches else [])
            ],
            ensure_ascii=False,
        )

    async def call(prompt, schema, files, criteria, weights):
        response_format = getattr(reward.judge, "response_format", "none")
        sampling = {}
        if response_format == "json-schema":
            sampling["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "schema": schema,
                    "strict": True,
                },
            }
        elif response_format == "json-object":
            sampling["response_format"] = {"type": "json_object"}
        content = [
            TextContentPart(
                text=f"Task:\n{task.data.prompt_text}\n\n<submission>\nResponse:"
            ),
            *[
                part.model_copy(update={"text": html.escape(part.text, quote=False)})
                if isinstance(part, TextContentPart)
                else part
                for part in [TextContentPart(text=response), *inputs[tuple(files)]]
            ],
            TextContentPart(text="</submission>"),
            *reference,
        ]
        try:
            async with asyncio.timeout(reward.judge.timeout):
                if agentic:
                    from rewardkit.judges import parse_judge_response

                    from verifiers.v1.configs.sampling import SamplingConfig
                    from verifiers.v1.envs.agentic_judge.env import (
                        TRACE_FILE,
                        VERDICT_FILE,
                        JudgeTask,
                    )
                    from verifiers.v1.judges.agent import AgentJudgeRequest

                    content.append(
                        TextContentPart(
                            text=f"Write the JSON answer to {VERDICT_FILE}. Required schema:\n{json.dumps(schema)}"
                        )
                    )
                    judge_task = JudgeTask(
                        TaskData(
                            idx=trace.task.data.idx,
                            prompt=[
                                SystemMessage(content=prompt),
                                UserMessage(content=content),
                            ],
                        ),
                        files={
                            TRACE_FILE: trace.to_json(
                                exclude={
                                    "nodes": {
                                        "__all__": {
                                            "message": {
                                                "reasoning_content",
                                                "provider_state",
                                            }
                                        }
                                    }
                                }
                            )
                        },
                        parse_verdict=lambda text: [
                            s.model_dump()
                            for s in parse_judge_response(text, criteria, weights)
                        ],
                        workspace=workspace,
                        env=dict(runtime.env),
                    )
                    verdict = await task.judge_executor(
                        AgentJudgeRequest(
                            judge_task,
                            model=reward.judge.model,
                            sampling=SamplingConfig(
                                reasoning_effort=reward.judge.reasoning_effort
                            )
                            if reward.judge.reasoning_effort is not None
                            else None,
                            timeout=authored.timeout,
                            isolated=True,
                            network=network,
                        ),
                        runtime,
                    )
                    if not verdict.ok:
                        return AgentAttempt(
                            error=RuntimeError(
                                "; ".join(e.message for e in verdict.errors)
                            )
                        )
                    return AgentAttempt(output=verdict.info["verdict_raw"])
                result = await judge.complete(
                    [SystemMessage(content=prompt), UserMessage(content=content)],
                    trace=trace,
                    **sampling,
                )
        except (TimeoutError, APITimeoutError):
            return AgentAttempt(timed_out=True)
        return AgentAttempt(output=result.text)

    reward = NativeReward.bind(
        reward, call, files=authored.files, selections=selections
    )
    return PreparedJudge(
        reward,
        trace,
        path,
        judge.reward_name,
        config.model_dump(mode="json", by_alias=True),
    )
