"""A plural taskset for a proposer -> solver workflow.

`load()` creates only proposer seeds. After a proposer rollout finishes, the orchestrator
calls `solved_task(trace)` to create the second concrete task type. The taskset owns both
types and their explicit configs even though orchestration happens outside the taskset.
Both types require a container runtime because their rewards execute proposed code.
"""

import asyncio
import math
import re
from typing import Self, cast

import verifiers.v1 as vf

from proposer_solver_v1.servers.submit import (
    SubmissionState,
    SubmitToolset,
    SubmitToolsetConfig,
)

PROPOSE_PROMPT = """You are inventing a compact reasoning puzzle.

Submit exactly one self-contained problem with the `propose_submit_question` tool. The
problem must be solvable both by careful reasoning and by a complete stdlib-only Python
script that reads one argument from `sys.argv[1]` and prints exactly one answer. Use small
integers or short strings; avoid randomness, external facts, placeholders, and ambiguity.
Do not reveal the code or answer in the natural-language question."""

SOLVE_PROMPT = """Solve this puzzle. Reason it out carefully.

{question}

Show your reasoning, then put the final answer alone on the last line as
`{answer_prefix} <value>`."""

UV_SCRIPT_HEADER = "# /// script\n# dependencies = []\n# ///\n"
GROUND_TRUTH_TIMEOUT = 10
UNTRUSTED_CODE_RESOURCES = vf.TaskResources(cpu=1, memory=1)


def parse_number(text: str) -> str | None:
    cleaned = text.strip().strip("`").lstrip("$").replace(",", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return str(int(value)) if value == int(value) else str(value)


def answer_tokens(text: str) -> list[str]:
    parts = [part for part in re.split(r"[,\s]+", text.strip().strip("[]()")) if part]
    return [parse_number(part) or part.lower() for part in parts]


def answers_match(expected: str, got: str) -> bool:
    expected_number, got_number = parse_number(expected), parse_number(got)
    if expected_number is not None and got_number is not None:
        return expected_number == got_number
    return answer_tokens(expected) == answer_tokens(got)


async def run_ground_truth(runtime: vf.Runtime, code: str, input: str) -> str | None:
    try:
        result = await asyncio.wait_for(
            runtime.run_uv_script(UV_SCRIPT_HEADER + code, args=[input]),
            GROUND_TRUTH_TIMEOUT,
        )
    except TimeoutError:
        return None
    if result.exit_code != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[-1].strip()


class ProposerData(vf.TaskData):
    pass


class SolvedData(vf.TaskData):
    question: str
    code: str
    input: str


class ProposerConfig(vf.TaskConfig):
    submit: SubmitToolsetConfig = SubmitToolsetConfig()


class SolvedConfig(vf.TaskConfig):
    answer_prefix: str = "ANSWER:"


class ProposerTask(vf.Task[ProposerData, SubmissionState, ProposerConfig]):
    NEEDS_CONTAINER = True
    tools = (SubmitToolset,)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        state = cast(SubmissionState, trace.state)
        if state.submitted:
            trace.info["submission"] = {
                "code": state.code,
                "input": state.input,
                "question": state.question,
            }

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return cast(SubmissionState, trace.state).submitted

    @vf.reward(weight=0.1)
    async def well_formed(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        submission = trace.info.get("submission")
        if not submission or not submission.get("code"):
            return 0.0
        expected = await run_ground_truth(
            runtime, submission["code"], submission.get("input", "")
        )
        return float(expected is not None)


class SolvedTask(vf.Task[SolvedData, vf.State, SolvedConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def from_trace(cls, trace: vf.Trace, *, config: SolvedConfig | None = None) -> Self:
        submission = trace.info["submission"]
        config = config or SolvedConfig()
        return cls(
            SolvedData(
                idx=trace.task.idx,
                prompt=SOLVE_PROMPT.format(
                    question=submission["question"],
                    answer_prefix=config.answer_prefix,
                ),
                resources=UNTRUSTED_CODE_RESOURCES,
                question=submission["question"],
                code=submission["code"],
                input=submission["input"],
            ),
            config,
        )

    async def validate(self, runtime: vf.Runtime) -> bool:
        return (
            await run_ground_truth(runtime, self.data.code, self.data.input) is not None
        )

    @vf.reward
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        config = cast(SolvedConfig, self.config)
        got = None
        for line in reversed(trace.last_reply.splitlines()):
            text = line.strip().strip("*`").strip()
            if text.upper().startswith(config.answer_prefix.upper()):
                got = text[len(config.answer_prefix) :].strip().strip("*`")
                break
        if got is None:
            return 0.0
        expected = await run_ground_truth(runtime, self.data.code, self.data.input)
        return float(expected is not None and answers_match(expected, got))


class ProposerSolverConfig(vf.TasksetConfig):
    num_tasks: int = 1
    proposer: ProposerConfig = ProposerConfig()
    solved: SolvedConfig = SolvedConfig()


class ProposerSolverTaskset(
    vf.Taskset[ProposerTask | SolvedTask, ProposerSolverConfig]
):
    def load(self) -> list[ProposerTask | SolvedTask]:
        return [
            ProposerTask(
                ProposerData(
                    idx=idx,
                    prompt=PROPOSE_PROMPT,
                    resources=UNTRUSTED_CODE_RESOURCES,
                ),
                self.config.proposer,
            )
            for idx in range(self.config.num_tasks)
        ]

    def solved_task(self, trace: vf.Trace) -> SolvedTask:
        return SolvedTask.from_trace(trace, config=self.config.solved)
