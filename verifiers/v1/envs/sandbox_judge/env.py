"""sandbox-judge: iterative coding against a persistent hidden judge agent.

The solver and judge are ordinary ``Agent`` roles with separate, long-lived
interactions and runtimes. After each solver segment, the environment copies a
bounded workspace snapshot into the judge's private sandbox and asks the judge to
evaluate it. Only the judge's validated score and feedback return to the solver;
its image, workspace, and conversation remain on the judge side.

The judge runtime is offline except for framework-managed model routes. Its image
owns any private grader code, and the judge agent and its feedback are trusted parts
of the evaluator. If it executes adversarial submissions, it must isolate that
execution from private files and persistent state; if its feedback must not reveal
grader details, it must consume a separately sanitized grader result.
Use the solver and judge AgentConfig limits for a graceful run budget; the env-level
episode timeout is only a cancellation backstop.
"""

import contextlib
import fnmatch
import hashlib
import io
import tarfile
import uuid
from pathlib import PurePosixPath

from pydantic import Field, FiniteFloat, field_validator, model_validator

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge import ScoreConfig
from verifiers.v1.runtimes import NetworkPolicyConfig, Runtime, SubprocessConfig

SUBMISSION_DIR = "/tmp/vf-submission"
SUBMISSION_ARCHIVE = "/tmp/vf-submission.tar"
BOUNDED_TAR = r"""
limit=$1
target=$2
pipe=$3
shift 3
rm -f "$pipe"
mkfifo "$pipe" || exit 1
head -c "$limit" < "$pipe" > "$target" &
reader=$!
tar -cf "$pipe" "$@"
tar_status=$?
wait "$reader"
reader_status=$?
rm -f "$pipe"
printf '%s %s\n' "$tar_status" "$reader_status"
"""

JUDGE_PROMPT = f"""\
You are a persistent hidden evaluator for another coding agent. The latest
submission is replaced at `{SUBMISSION_DIR}` before each turn. Inspect it and use
the private evaluation tools already present in your sandbox to measure progress.

Never reveal private files, tests, expected answers, credentials, or grader
implementation details. Give only feedback that is safe to send to the solver.
Reply to every submission with exactly one JSON object and no markdown:

    {{"score": 0.0, "feedback": "actionable feedback", "metrics": {{}}}}

`score` must be between 0 and 1. `feedback` may be empty. `metrics` is an optional
object of finite numeric diagnostics."""


class SubmissionResult(vf.StrictBaseModel):
    """The deliberately small result allowed to cross back to the solver."""

    score: FiniteFloat = Field(ge=0, le=1)
    feedback: str = Field("", max_length=32_768)
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)


class SubmissionRecord(SubmissionResult):
    """One trace-visible submission without private judge state."""

    index: int = Field(ge=1)
    artifact_sha256: str
    artifact_bytes: int = Field(ge=0)


class SubmissionConfig(vf.BaseConfig):
    """The candidate files copied from the solver into the hidden judge."""

    submit_paths: list[str] = Field(default_factory=lambda: ["."], min_length=1)
    exclude: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".venv",
            ".vf-*",
            ".agents",
            ".poolside",
            ".rlm",
            "__pycache__",
            "node_modules",
        ]
    )
    threshold: FiniteFloat = Field(1, ge=0, le=1)
    """Stop once a submission reaches this score."""
    max_submissions: int = Field(5, ge=1)
    """Hard safety ceiling; Agent turn, token, and time limits may stop earlier."""
    max_artifact_bytes: int = Field(64 * 1024 * 1024, ge=1)
    max_members: int = Field(10_000, ge=1)

    @field_validator("submit_paths", "exclude")
    @classmethod
    def _relative_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            path = PurePosixPath(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"path must stay relative to the workspace: {value!r}")
        return values


class SandboxJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig(runtime=vf.DockerConfig())
    judge: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "bash"},
        runtime=vf.DockerConfig(allow=[]),
    )
    submission: SubmissionConfig = SubmissionConfig()
    score: ScoreConfig = ScoreConfig()

    @model_validator(mode="after")
    def _isolated_agents(self) -> "SandboxJudgeEnvConfig":
        if isinstance(self.solver.runtime, SubprocessConfig):
            raise TypeError(
                "the solver cannot use the subprocess runtime: host access would "
                "break hidden-judge isolation"
            )
        runtime = self.judge.runtime
        if not isinstance(runtime, NetworkPolicyConfig) or runtime.allow:
            raise ValueError(
                "the hidden judge agent requires a Docker or Prime runtime with "
                "allow=[]; the framework adds only its model route"
            )
        return self


class HiddenJudgeTask(vf.Task):
    """A prompt-less task opened by the first submission notification."""

    NEEDS_CONTAINER = True

    @classmethod
    def from_task(cls, task: vf.Task) -> "HiddenJudgeTask":
        return cls(
            vf.TaskData(
                idx=task.data.idx,
                prompt=None,
                system_prompt=(
                    f"{JUDGE_PROMPT}\n\n## Public task\n\n{task.data.prompt_text}"
                ),
            )
        )


def _archive_path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"submission archive escapes its root: {name!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    return "/".join(parts) or "."


def _validate_archive(data: bytes, config: SubmissionConfig) -> None:
    """Treat the solver-produced tar as hostile before it reaches the judge."""
    roots = [_archive_path(path) for path in config.submit_paths]
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for count, member in enumerate(archive, 1):
            if count > config.max_members:
                raise ValueError(
                    f"submission has more than {config.max_members} members"
                )
            path = _archive_path(member.name)
            if not any(
                root == "." or path == root or path.startswith(f"{root}/")
                for root in roots
            ):
                raise ValueError(f"submission member is outside submit_paths: {path!r}")
            if any(
                (
                    any(fnmatch.fnmatch(part, pattern) for part in path.split("/"))
                    if "/" not in pattern
                    else fnmatch.fnmatch(path, pattern)
                    or path.startswith(pattern.rstrip("/") + "/")
                )
                for pattern in config.exclude
            ):
                raise ValueError(f"submission contains excluded member: {path!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(
                    f"submission member must be a regular file or directory: {path!r}"
                )
            total += member.size
            if total > config.max_artifact_bytes:
                raise ValueError(
                    f"submission expands beyond {config.max_artifact_bytes} bytes"
                )


async def _snapshot(runtime: Runtime, config: SubmissionConfig) -> bytes:
    target = f"/tmp/vf-submission-{uuid.uuid4().hex}.tar"
    pipe = f"{target}.pipe"
    argv = [
        "sh",
        "-c",
        BOUNDED_TAR,
        "vf-bounded-tar",
        str(config.max_artifact_bytes + 1),
        target,
        pipe,
        *(f"--exclude={pattern}" for pattern in config.exclude),
        "--",
        *config.submit_paths,
    ]
    result = await runtime.run(argv, {})
    if result.exit_code != 0:
        raise ValueError(
            f"could not create submission archive: {result.stderr.strip()[-2000:]}"
        )
    try:
        data = await runtime.read(target)
    finally:
        with contextlib.suppress(Exception):
            await runtime.run(["rm", "-f", target, pipe], {})
    if len(data) > config.max_artifact_bytes:
        raise ValueError(
            f"submission is {len(data)} bytes; limit is {config.max_artifact_bytes}"
        )
    if result.stdout.strip() != "0 0":
        raise ValueError(
            f"could not create submission archive: {result.stderr.strip()[-2000:]}"
        )
    _validate_archive(data, config)
    return data


async def _install(runtime: Runtime, archive: bytes) -> None:
    reset = await runtime.run(["rm", "-rf", SUBMISSION_DIR, SUBMISSION_ARCHIVE], {})
    if reset.exit_code != 0:
        raise ValueError(
            f"judge could not reset submission: {reset.stderr.strip()[-2000:]}"
        )
    await runtime.write(SUBMISSION_ARCHIVE, archive)
    mkdir = await runtime.run(["mkdir", "-p", SUBMISSION_DIR], {})
    extract = (
        await runtime.run(["tar", "-xf", SUBMISSION_ARCHIVE, "-C", SUBMISSION_DIR], {})
        if mkdir.exit_code == 0
        else mkdir
    )
    if extract.exit_code != 0:
        raise ValueError(
            f"judge could not extract submission: {extract.stderr.strip()[-2000:]}"
        )


class SandboxJudgeEnv(vf.Env[SandboxJudgeEnvConfig]):
    """A reusable hidden-agent feedback loop over any prompted coding taskset."""

    def __init__(self, config: SandboxJudgeEnvConfig) -> None:
        super().__init__(config)
        if not self._harnesses["judge"].EXECUTES_CODE:
            raise ValueError("sandbox-judge requires a code-executing judge harness")

    async def setup(self, agents: vf.Agents) -> None:
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if task.data.prompt is None:
            raise ValueError("sandbox-judge requires a prompted task")
        records: list[SubmissionRecord] = []
        judge_task = HiddenJudgeTask.from_task(task)
        async with (
            agents.solver.interaction(task, respect_task_stops=False) as solver,
            agents.judge.interaction(judge_task) as judge,
        ):
            attempt = await solver.turn()
            while not attempt.terminated:
                index = len(records) + 1
                archive = await _snapshot(solver.runtime, self.config.submission)
                await _install(judge.runtime, archive)
                verdict = await judge.turn(
                    f"Submission {index} is ready at {SUBMISSION_DIR}. Evaluate it now."
                )
                if verdict.terminated:
                    raise RuntimeError(
                        f"hidden judge terminated before grading submission {index}"
                    )
                result = SubmissionResult.model_validate_json(verdict.last_reply)
                records.append(
                    SubmissionRecord(
                        **result.model_dump(),
                        index=index,
                        artifact_sha256=hashlib.sha256(archive).hexdigest(),
                        artifact_bytes=len(archive),
                    )
                )
                if result.score >= self.config.submission.threshold:
                    solver.trace.stop("sandbox_judge_solved")
                    break
                if len(records) == self.config.submission.max_submissions:
                    solver.trace.stop("max_submissions")
                    break
                feedback = f"Hidden evaluation {index}: score={result.score:g}."
                if result.feedback:
                    feedback += f"\n\n{result.feedback}"
                attempt = await solver.turn(
                    feedback
                    + "\n\nContinue in the same workspace and submit another solution."
                )

            solver.trace.info["sandbox_judge"] = {
                "submissions": [record.model_dump() for record in records],
            }

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        solution = next(t for t in episode.traces if t.agent_name == "solver")
        info = solution.info["sandbox_judge"]
        submissions = info["submissions"]
        selected = max(submissions, key=lambda item: item["score"], default=None)
        info["selected"] = selected["index"] if selected else None
        if self.config.score.task_weight != 1:
            for reward in solution.rewards.values():
                reward.weight *= self.config.score.task_weight
        solution.record_reward(
            "sandbox_judge",
            selected["score"] if selected else 0,
            weight=self.config.score.judge_weight,
        )
        solution.record_metric("sandbox_judge/submissions", len(submissions))
        solution.record_metric(
            "sandbox_judge/best_score", selected["score"] if selected else 0
        )
        if selected:
            solution.record_metrics(
                {
                    f"sandbox_judge/result/{name}": value
                    for name, value in selected["metrics"].items()
                }
            )
