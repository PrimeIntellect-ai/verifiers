"""A reusable *agentic, group-level* LLM judge for v1 tasksets.

Where `Judge` (judge.py) is one chat call that scores one rollout from text in its prompt,
`AgenticGroupJudge` scores a whole *group* of a task's rollouts at once, by letting the judge
**investigate them with tools inside a sandbox** and emit one score per rollout. It is the
runtime behind a taskset's ``@group_reward``.

Subclass it like `Judge` — set the class fields you want and override `tests_tar` to supply the
task's pristine test suite — then call `score` from `@group_reward`. The shape, end to end:

  1. While each rollout's runtime is still live, the taskset's ``finalize`` snapshots what the
     judge will need onto the trace — the agent's diff and a transcript — since the runtime is
     torn down before group scoring runs (see episode.py). Use ``render_transcript``:

        async def finalize(self, task, trace, runtime):
            diff = await runtime.run(
                ["sh", "-c", f"cd {REPO} && git add -A && git diff --cached --binary"], {}
            )
            trace.info["judge"] = {"patch": diff.stdout, "transcript": render_transcript(trace)}

  2. ``@group_reward`` delegates to the judge, which provisions ONE fresh runtime from the
     task's image (``resolve_runtime_config`` injects ``task.image``), rebuilds each rollout as
     ``base + that rollout's patch`` under ``/judge/rollouts/<i>`` (shuffled, to defeat position
     bias), stages the *pristine* test suite alongside each, and runs the judge agent. The agent
     runs commands / reads files (e.g. runs the pristine tests against each candidate, inspects
     the diffs for tampering or regressions) and finishes by calling ``submit_scores``:

        class SweJudge(vf.AgenticGroupJudge):
            repo_path = "/repo"
            def tests_tar(self, task): return make_tar(task.tests_dir)

        class SweJudgedTaskset(vf.Taskset[SweTask, JudgedConfig]):
            @functools.cached_property
            def judge(self):                          # taskset wires the judge from its config
                return SweJudge(self.config.judge, self.config.judge_runtime)

            @vf.group_reward(weight=1.0)
            async def panel(self, traces, task) -> list[float]:
                return await self.judge.score(traces, task)

The scores returned by ``@group_reward`` are summed (× weight) into each ``trace.reward``; in
prime-rl those become the group's rewards and GRPO baselines them. Any failure — the judge
times out, returns the wrong count, a snapshot is missing — propagates, which marks the group
errored and (in prime-rl) drops it. The judge model/endpoint is a ``JudgeConfig`` (its own
endpoint, not the policy); the judge sandbox is a ``RuntimeConfig`` resolved per task so it
carries the task's toolchain (the candidates' tests actually run).
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from verifiers.v1.clients.config import build_async_openai
from verifiers.v1.judge import JudgeConfig
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime

if TYPE_CHECKING:
    from verifiers.v1.task import Task
    from verifiers.v1.trace import Trace


DEFAULT_JUDGE_PROMPT = """\
You are comparing {n} candidate solutions to the SAME task, laid out in the sandbox as
/judge/rollouts/0 .. /judge/rollouts/{last}. Each is the base repository with that candidate's
patch applied; the candidate could NOT see or edit the test suite, a pristine copy of which is
in each rollout's _tests/ directory. Each rollout's agent transcript is in _transcript.txt, and
_apply.log shows whether that candidate's patch applied cleanly (a non-zero exit means its
changes did NOT land — it is the untouched base).

Investigate with the tools: run the pristine tests against each candidate (e.g.
`cd /judge/rollouts/<i> && <how this repo runs its tests>`), read the diffs and transcripts,
and watch for regressions, for solutions that don't address the root cause, and for any attempt
to tamper with or hard-code around the tests. The _-prefixed files above are added by this
harness and are git-excluded, so `git -C /judge/rollouts/<i> diff` shows only that candidate's
own change.

Then score the candidates RELATIVE TO EACH OTHER — spread your scores and avoid ties; only the
within-group ordering and spacing matter. Finish by calling submit_scores with one score and one
short rationale per rollout, in displayed order (index 0..{last}).
"""

_OUTPUT_LIMIT = 8_000

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the judge sandbox and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the judge sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_scores",
            "description": "Submit one comparative score and rationale per rollout, in displayed order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scores": {"type": "array", "items": {"type": "number"}},
                    "rationale": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["scores", "rationale"],
            },
        },
    },
]


def truncate_output(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    """Clip tool output before it goes back into the judge's context."""
    return text if len(text) <= limit else text[:limit] + "\n...<truncated>"


def render_transcript(trace: "Trace") -> str:
    """Render a rollout's conversation to plain text for the judge to read. Walks the trace's
    branches; assistant tool calls are included so the judge sees what the agent *did*."""
    lines: list[str] = []
    for branch in trace.branches:
        for message in branch.messages:
            lines.append(f"## {getattr(message, 'role', '?')}")
            content = getattr(message, "content", None)
            if isinstance(content, str):
                lines.append(content)
            elif content is not None:
                lines.append(json.dumps(content, default=str))
            for call in getattr(message, "tool_calls", None) or []:
                lines.append(f"[tool_call] {call.function.name}({call.function.arguments})")
    return "\n".join(lines)


class AgenticGroupJudge:
    """Scores a group of a task's rollouts by letting an LLM judge investigate them with tools
    in a sandbox. Subclass to override `tests_tar` (and any class field), construct from a
    `JudgeConfig` + `RuntimeConfig` on the owning taskset, and call :meth:`score` from
    ``@group_reward``.
    """

    system_prompt: str = DEFAULT_JUDGE_PROMPT
    """Comparative judging prompt; ``{n}``/``{last}`` are formatted in per call."""
    repo_path: str = "/testbed"
    """Where the base repository lives inside the task image (image-dependent, e.g. ``/testbed``
    for SWE-bench-style images); each candidate is ``git clone``d from here."""
    max_turns: int = 25
    """Hard cap on judge tool-use turns before the group is failed."""

    def __init__(self, config: JudgeConfig, runtime: RuntimeConfig) -> None:
        self.config = config
        self.runtime_config = runtime
        self.client: AsyncOpenAI = build_async_openai(config)

    def tests_tar(self, task: "Task") -> bytes:
        """Override: the task's pristine test suite as a tar archive, staged into each
        candidate's ``_tests/`` for the judge to run."""
        raise NotImplementedError(f"{type(self).__name__} must implement tests_tar(task)")

    async def score(self, traces: list["Trace"], task: "Task") -> list[float]:
        """Provision the judge sandbox from ``task.image``, rebuild every rollout as base + its
        patch (shuffled) with the pristine tests staged, run the judge agent, and return one
        score per trace in the original order. Raises on any failure (→ the group is dropped)."""
        from verifiers.v1.env import resolve_runtime_config

        n = len(traces)
        name = f"group-judge-{uuid.uuid4().hex}"  # unique: groups are scored concurrently
        runtime = make_runtime(resolve_runtime_config(self.runtime_config, task), name=name)
        await runtime.start()
        try:
            order = list(range(n))
            random.Random(task.idx).shuffle(order)  # deterministic per task, hides identity
            tests = self.tests_tar(task)
            await runtime.run(["sh", "-c", "mkdir -p /judge/rollouts"], {})
            await asyncio.gather(
                *(self._stage_rollout(runtime, displayed, traces[real], tests) for displayed, real in enumerate(order))
            )
            displayed_scores = await self._run_agent(runtime, n)
            scores = [0.0] * n
            for displayed, real in enumerate(order):
                scores[real] = displayed_scores[displayed]
            return scores
        finally:
            await runtime.stop()

    async def _stage_rollout(self, runtime: Runtime, idx: int, trace: "Trace", tests: bytes) -> None:
        info = trace.info.get("judge")
        if not info or "patch" not in info:
            raise RuntimeError(
                f"rollout {trace.idx} has no judge snapshot in trace.info['judge']; "
                "does the taskset's finalize() capture it?"
            )
        d = f"/judge/rollouts/{idx}"
        clone = await runtime.run(["sh", "-c", f"rm -rf {d} && git clone -q {self.repo_path} {d}"], {})
        if clone.exit_code != 0:
            raise RuntimeError(f"judge clone of {self.repo_path} failed: {truncate_output(clone.stderr)}")
        # Keep the candidate's git state pristine: our scaffolding must not show up when the judge
        # inspects `git diff`/`git status` for the candidate's real change (or for test tampering).
        await runtime.run(
            ["sh", "-c", f"printf '%s\\n' _cand.patch _apply.log _transcript.txt _tests >> {d}/.git/info/exclude"], {}
        )
        await runtime.write(f"{d}/_cand.patch", info["patch"].encode())
        apply = await runtime.run(["sh", "-c", f"cd {d} && git apply --binary _cand.patch"], {})
        # Don't fail the group if a patch is messy — record the outcome so the judge sees whether
        # this candidate's changes actually landed (vs. being scored as the untouched base).
        await runtime.write(f"{d}/_apply.log", f"exit={apply.exit_code}\n{apply.stderr}".encode())
        await runtime.write(f"{d}/_transcript.txt", info.get("transcript", "").encode())
        await runtime.write(f"{d}/_tests.tar", tests)
        await runtime.run(
            ["sh", "-c", f"cd {d} && mkdir -p _tests && tar xf _tests.tar -C _tests && rm _tests.tar"], {}
        )

    async def _run_agent(self, runtime: Runtime, n: int) -> list[float]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt.format(n=n, last=n - 1)}]
        sampling = self.config.sampling.model_dump(exclude_none=True)
        for _ in range(self.max_turns):
            completion = await self.client.chat.completions.create(
                model=self.config.model, messages=messages, tools=_TOOLS, **sampling
            )
            message = completion.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                messages.append({"role": "user", "content": "Use the tools to investigate, then call submit_scores."})
                continue
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if call.function.name == "submit_scores":
                    scores = args.get("scores")
                    if not isinstance(scores, list) or len(scores) != n:
                        raise RuntimeError(f"submit_scores expected {n} scores, got {scores!r}")
                    return [float(s) for s in scores]
                result = await self._dispatch(runtime, call.function.name, args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        raise RuntimeError(f"judge did not submit scores within {self.max_turns} turns")

    async def _dispatch(self, runtime: Runtime, name: str, args: dict[str, Any]) -> str:
        if name == "run_command":
            res = await runtime.run(["sh", "-c", str(args.get("command", ""))], {})
            return truncate_output(f"exit={res.exit_code}\n{res.stdout}{res.stderr}")
        if name == "read_file":
            return truncate_output((await runtime.read(str(args.get("path", "")))).decode(errors="replace"))
        raise RuntimeError(f"unknown judge tool {name!r}")
