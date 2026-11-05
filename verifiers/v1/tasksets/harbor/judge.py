"""Native model calls for a RewardKit-authored rubric."""

import asyncio
import base64
import mimetypes
import tomllib
from pathlib import PurePosixPath

from openai import APITimeoutError

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.errors import TaskError
from verifiers.v1.judge import Judge
from verifiers.v1.types import (
    ImageUrlContentPart,
    ImageUrlSource,
    TextContentPart,
    UserMessage,
)

MAX_BYTES = 1024 * 1024


async def read_inputs(runtime, paths):
    if not paths:
        return []
    result = await runtime.run(
        [
            "find",
            "-L",
            *(p if p.startswith("/") else f"./{p}" for p in paths),
            "-maxdepth",
            "1",
            "-type",
            "f",
            "-print0",
        ],
        {},
    )
    if result.exit_code or not result.stdout:
        raise TaskError(f"Cannot list judge inputs {paths!r}: {result.stderr[-500:]}")
    parts = []
    for path in sorted(set(result.stdout.split("\0")[:-1])):
        data = await runtime.read(path, max_bytes=MAX_BYTES)
        mime = mimetypes.guess_type(path)[0] or ""
        parts.append(TextContentPart(text=f"File {path}:"))
        if mime.startswith("image/"):
            parts.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(
                        url=f"data:{mime};base64,{base64.b64encode(data).decode()}"
                    )
                )
            )
        else:
            parts.append(TextContentPart(text=data.decode("utf-8")))
    return parts


async def run_judge(task, runtime, trace, path="/tests/judge.toml"):
    from rewardkit.agents import AgentAttempt
    from rewardkit.models import JudgeTomlConfig, LLMJudge

    from .rewardkit import NativeReward, load_rubric

    raw = tomllib.loads((await runtime.read(path, max_bytes=MAX_BYTES)).decode())
    config = JudgeTomlConfig.model_validate(raw)
    authored = config.judge
    unsupported = {
        "model",
        "version",
        "isolated",
        "cwd",
        "reference",
        "atif_trajectory",
        "mcp_servers",
    } & authored.model_fields_set
    if unsupported:
        raise TaskError(f"Unsupported native judge options: {sorted(unsupported)}")
    template = None
    if authored.prompt_template is not None:
        source = PurePosixPath(path).parent / authored.prompt_template
        if source.suffix not in {".txt", ".md"}:
            raise TaskError("judge prompt_template must be .txt or .md")
        template = (await runtime.read(str(source), max_bytes=MAX_BYTES)).decode()
    reward = load_rubric(raw, path, runtime.config.workdir or "/app", template)
    if not isinstance(reward.judge, LLMJudge):
        raise TaskError("This judge requires native agentic execution")
    groups = {tuple(c.files or authored.files) for c in reward.criteria}
    groups.add(tuple(authored.files))
    groups.add(
        tuple(
            dict.fromkeys(
                f for c in reward.criteria for f in (c.files or authored.files)
            )
        )
    )
    inputs = {paths: await read_inputs(runtime, paths) for paths in groups}
    settings = (
        task.config.judge.model_dump(exclude_unset=True) if task.config.judge else {}
    )
    settings.setdefault("model", reward.judge.model)
    settings.setdefault("name", "harbor")
    settings["sampling"] = {
        "reasoning_effort": reward.judge.reasoning_effort
    } | settings.get("sampling", {})
    judge = Judge(JudgeConfig(**settings))

    async def call(prompt, schema, files):
        response_format = reward.judge.response_format
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
                text=prompt
                + f"\n\nTask:\n{task.data.prompt_text}\n\nFinal response:\n{trace.last_reply}"
            ),
            *inputs[tuple(files)],
        ]
        try:
            async with asyncio.timeout(reward.judge.timeout):
                result = await judge.complete(
                    [UserMessage(content=content)], trace=trace, **sampling
                )
        except (TimeoutError, APITimeoutError):
            return AgentAttempt(timed_out=True)
        return AgentAttempt(output=result.text)

    reward = NativeReward.bind(reward, call, files=authored.files)
    await reward.arun()
    trace.record_metrics(
        {f"{judge.reward_name}/{score.name}": score.value for score in reward.scores}
    )
    trace.info.setdefault("harbor_judge", {})[path] = reward.to_detail_dict(
        reward.score
    )
    return reward.score
