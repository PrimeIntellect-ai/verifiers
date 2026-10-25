"""best-of-n-judge: n attempts at the task, ranked side by side by agentic judges.

The "agent" role attempts the task `--env.n` times, as in best-of-n. A judge then
gets a fresh sandbox from the task's image holding every attempt's trace record
and collected artifacts under shuffled labels, verifies them empirically, and
writes a best-to-worst ranking. `--env.judges` runs that many independent judges
(an ensemble, each with its own shuffle); an attempt's `judge` reward is its rank
score averaged across them.
"""

import asyncio
import json
import random
import shlex
import statistics
import uuid

from pydantic import BaseModel, ConfigDict, Field

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge.env import (
    VERDICT_FILE,
    JudgeTaskConfig,
    ScoreConfig,
    TextSource,
)
from verifiers.v1.envs.best_of_n import BestOfNEnv, BestOfNEnvConfig
from verifiers.v1.runtimes import backend_cls
from verifiers.v1.utils.artifacts import _validate_restore

ATTEMPTS_DIR = "/tmp/attempts"
# Shuffles stay independent of the global `random`, which tasksets may seed.
_SHUFFLE = random.SystemRandom()

RANK_PROMPT = """\
You are comparing several attempts at the same task and ranking them from best
to worst. Verify the work EMPIRICALLY: reconstruct what each attempt did from its
trace and artifacts, and test its output with real execution in your sandbox —
never take a trace's word for an outcome you can check.

## The task the attempts were given

{prompt}"""

HINT_SECTION = """\
## Hints

{hint}"""

WORKSPACE_NOTE = f"""\
## Your workspace

This is a fresh sandbox from the task's image; no attempt ran in it. Each attempt
has its own directory, `{ATTEMPTS_DIR}/<label>/`, labelled in random order:

- `trace.json`: the attempt's observable trace record (JSON: messages, tool
  calls, and its `info` artifacts), written by the harness — not the attempt. It
  can be very large — never dump it whole; list its keys, then slice out specific
  fields with python or jq. Hidden reasoning and opaque provider state are not
  included. It may also carry the task's own
  scores and reference material: context, not your standard. Your ranking is what
  YOU verified by execution.
- `artifacts/`: the files the attempt published, each at its original absolute
  path (`/app/answer.txt` is at `artifacts/app/answer.txt`). Absent when the
  attempt published none.

To test an attempt's output, copy it to where the task expects it, and restore
that state before testing the next attempt."""


def _verdict_section(labels: list[str]) -> str:
    listing = ", ".join(labels)
    return f"""\
## Your verdict

The attempts are {listing}. When you are done verifying, write your ranking as
JSON to `{VERDICT_FILE}`:

    {{"ranking": [["<best label>"], ["<label>", "<label tied with it>"], ...],
      "reasons": {{"<label>": "<one sentence citing what you verified>", ...}}}}

`ranking` lists tiers from best to worst; attempts of equal quality share a tier.
Name every attempt exactly once, leave no tier empty, and give every attempt a
reason."""


class Ranking(BaseModel):
    """A judge's verdict: best-to-worst tiers of attempt labels, ties sharing one."""

    model_config = ConfigDict(extra="forbid")

    ranking: list[list[str]]
    reasons: dict[str, str]

    def check(self, labels: list[str]) -> None:
        ranked = [label for tier in self.ranking for label in tier]
        if any(not tier for tier in self.ranking) or sorted(ranked) != sorted(labels):
            raise ValueError(
                f"the ranking must name each of {labels} exactly once in nonempty "
                f"tiers, got {self.ranking}"
            )
        if sorted(self.reasons) != sorted(labels):
            raise ValueError(
                f"the reasons must cover exactly {labels}, got {sorted(self.reasons)}"
            )

    def scores(self) -> dict[str, float]:
        """Each attempt's share of the others it beats, a tie counting half: a sole
        best scores 1, a sole worst 0, and an all-tied ranking 0.5 each."""
        count = sum(len(tier) for tier in self.ranking)
        below = count
        scores = {}
        for tier in self.ranking:
            below -= len(tier)
            score = (below + (len(tier) - 1) / 2) / (count - 1)
            scores.update(dict.fromkeys(tier, score))
        return scores


class RankTask(vf.Task):
    """One judge's ranking task: every attempt's record and artifacts uploaded
    under its label before the judge starts, the ranking scraped off the box after
    it exits. `NEEDS_CONTAINER`: the judge executes code, never on the host."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, labels: dict[str, vf.Trace]) -> None:
        super().__init__(data)
        self.labels = labels

    @staticmethod
    def record(attempt: vf.Trace) -> bytes:
        record = attempt.to_record()
        # This snapshot is evidence for verification, not its final outcome.
        record.pop("ok", None)
        for node in record["nodes"]:
            node["message"].pop("reasoning_content", None)
            node["message"].pop("provider_state", None)
        return json.dumps(record).encode()

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # Archive bytes are the attempts': confine every member to its declared
        # root before anything lands in the judge's box.
        for attempt in self.labels.values():
            for source, archive in attempt.state.artifacts.items():
                _validate_restore(source, archive)
        await runtime.run(["rm", "-rf", ATTEMPTS_DIR, VERDICT_FILE], env={})
        for label, attempt in self.labels.items():
            root = f"{ATTEMPTS_DIR}/{label}"
            await runtime.write(f"{root}/trace.json", self.record(attempt))
            for source, archive in attempt.state.artifacts.items():
                if archive is None:
                    continue
                upload = f"/tmp/vf-attempt-{uuid.uuid4().hex}.tar"
                await runtime.write(upload, archive)
                target = shlex.quote(f"{root}/artifacts")
                unpack = (
                    f"mkdir -p {target} && tar -xf {upload} -C {target} "
                    f"&& rm -f {upload}"
                )
                result = await runtime.run(["sh", "-c", unpack], env={})
                if result.exit_code:
                    raise RuntimeError(
                        f"failed to unpack attempt {label}'s artifact {source!r}: "
                        f"{(result.stderr or result.stdout).strip()[-500:]}"
                    )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Scrape the ranking off the box while it's alive. A judge that wrote no
        valid ranking fails HERE — on its own trace, the retryable unit."""
        try:
            raw = await runtime.read(VERDICT_FILE)
        except Exception as e:
            raise ValueError(
                f"the judge wrote no ranking to {VERDICT_FILE}; its final act must be "
                'writing {"ranking": [[...], ...], "reasons": {...}} there'
            ) from e
        ranking = Ranking.model_validate_json(raw)
        ranking.check(list(self.labels))
        trace.info["ranking"] = {
            "attempts": {label: attempt.id for label, attempt in self.labels.items()},
            **ranking.model_dump(),
            "scores": {
                self.labels[label].id: score
                for label, score in ranking.scores().items()
            },
        }


class RankTaskConfig(vf.BaseConfig):
    """The judges' minted task: the ranking policy and what lands in their box."""

    prompt: TextSource | None = None
    """Ranking policy: a string is inline text; a `Path` in Python or
    `{ path = "policy.md" }` in config reads a file. Replaces only the policy
    body — the verdict contract and workspace note are always appended. May
    reference `{prompt}` (the attempts' task); if it doesn't, the task statement
    is appended after."""
    hint: TextSource | None = None
    """Optional hints, with the same inline/file forms as `prompt`, injected as
    their own section."""

    def build_policy(self) -> str:
        if self.prompt is None:
            return RANK_PROMPT
        return JudgeTaskConfig._resolve(self.prompt)

    def build_hint(self) -> str | None:
        return JudgeTaskConfig._resolve(self.hint) if self.hint is not None else None

    def build_prompt(self, task: vf.TaskData, labels: list[str]) -> str:
        statement = "\n\n".join(
            text for text in (task.system_prompt, task.prompt_text) if text
        )
        template = self.build_policy()
        body = template.replace("{prompt}", statement)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n## The task the attempts were given\n\n" + statement
        sections = [body, _verdict_section(labels), WORKSPACE_NOTE]
        if (hint := self.build_hint()) is not None:
            sections.insert(1, HINT_SECTION.replace("{hint}", hint))
        return "\n\n".join(sections)


class BestOfNJudgeEnvConfig(BestOfNEnvConfig):
    n: int = Field(4, ge=2)
    """Independent attempts per episode, ranked against each other."""
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge program; each judge gets a fresh container from the task's image."""
    judges: int = Field(1, ge=1)
    """Independent judges per episode (the ensemble); each ranks its own shuffle."""
    task: RankTaskConfig = RankTaskConfig()
    score: ScoreConfig = ScoreConfig()


class BestOfNJudgeEnv(BestOfNEnv, vf.Env[BestOfNJudgeEnvConfig]):
    config: BestOfNJudgeEnvConfig

    def __init__(self, config: BestOfNJudgeEnvConfig) -> None:
        super().__init__(config)
        # A missing policy file fails here, not mid-episode.
        config.task.build_policy()
        config.task.build_hint()
        # Refuse an impossible judge at construction, not after n attempts ran.
        judge = self._harnesses["judge"]
        if not judge.EXECUTES_CODE:
            raise ValueError(
                "best-of-n-judge requires a judge harness that can execute code, but "
                f"harness {judge.config.id!r} is a tool-less chat loop"
            )
        if not backend_cls(config.judge.runtime).SUPPORTS_TASK_IMAGES:
            raise ValueError(
                "best-of-n-judge requires a container judge runtime: "
                "--env.judge.runtime.type docker|prime"
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The judges grade the policy; their tokens are never training data.
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        # Artifacts need a runtime with its own filesystem; a subprocess attempt
        # is ranked on its trace alone.
        collect = backend_cls(self.config.agent.runtime).SUPPORTS_TASK_IMAGES
        async with asyncio.TaskGroup() as tg:
            runs = [
                tg.create_task(agents.agent.run(task, collect_artifacts=collect))
                for _ in range(self.config.n)
            ]
        attempts = [run.result() for run in runs]
        for attempt in attempts:
            attempt.record_metric("task_reward", attempt.reward)
        if not all(attempt.ok for attempt in attempts):
            # A failed attempt already fails the episode, and trainers skip failed
            # episodes whole: a ranking of the rest would be spent for nothing.
            return
        async with asyncio.TaskGroup() as tg:
            judged = [
                tg.create_task(self._rank(task, attempts, agents))
                for _ in range(self.config.judges)
            ]
        rankings = [run.result() for run in judged]
        for attempt in attempts:
            score = statistics.fmean(ranking[attempt.id] for ranking in rankings)
            if self.config.score.task_weight != 1.0:
                for reward in attempt.rewards.values():
                    if reward is not None:
                        reward.weight *= self.config.score.task_weight
            attempt.record_reward("judge", score, weight=self.config.score.judge_weight)

    async def _rank(
        self, task: vf.Task, attempts: list[vf.Trace], agents: vf.Agents
    ) -> dict[str, float]:
        """One judge's rank scores, keyed by attempt trace id."""
        shuffled = _SHUFFLE.sample(attempts, len(attempts))
        labels = dict(zip(_labels(len(shuffled)), shuffled))
        verdict = await agents.judge.run(self.make_rank_task(task, labels))
        if not verdict.ok:
            raise RuntimeError(
                "best-of-n judge failed: "
                + "; ".join(error.message for error in verdict.errors)
            )
        return verdict.info["ranking"]["scores"]

    def make_rank_task(self, task: vf.Task, labels: dict[str, vf.Trace]) -> RankTask:
        """One judge's task over the labelled attempts. Override to change what
        the judge sees (e.g. native image parts), or to return a RankTask subclass
        whose `setup` also rebuilds the task's world: a fresh box from the task's
        image doesn't replay the task's own setup."""
        data = task.data
        return RankTask(
            vf.TaskData(
                idx=data.idx,
                prompt=self.config.task.build_prompt(data, list(labels)),
                image=data.image,
                workdir=data.workdir,
                resources=data.resources,
                network=data.network,
            ),
            labels,
        )

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """best-of-n's sibling metrics over the attempts alone: `best` marks the
        attempt with the top final (judged) reward, `pass_at_n` whether any
        attempt's own task reward reached the threshold."""
        attempts = [t for t in episode.traces if t.agent.name == "agent"]
        top = max(t.reward for t in attempts)
        solved = float(
            max(t.metrics["task_reward"] or 0.0 for t in attempts)
            >= self.config.threshold
        )
        for trace in attempts:
            trace.record_metric("best", float(trace.reward == top))
            trace.record_metric("pass_at_n", solved)


def _labels(count: int) -> list[str]:
    """Distinct random hex labels, three digits unless more attempts need more,
    so a label hints at neither an attempt's index nor its listing position."""
    width = 3
    while 16**width < count:
        width += 1
    return [f"{i:0{width}x}" for i in _SHUFFLE.sample(range(16**width), count)]
