"""writer-editors-v1 — a writer drafts, n editors critique, the writer revises (example topology).

The rounds + fan-in shape, one self-contained package:

  - **Rounds are a loop in `go`**: each round, the current draft goes out to the editors
    and comes back revised — `num_rounds` cycles, plain Python.
  - **Fan-out**: `asyncio.gather(*(run.agent("editor").run(critique, parents=[draft])
    for _ in range(num_editors)))` — every editor critiques the same draft, concurrently.
  - **Fan-in**: the revision task is built from *all* the editors' traces at once, and the
    revised trace is linked under every one of them (`parents=[draft, *edits]`).
  - **One verdict, every trace**: a single `vf.Judge` call per instance compares the first
    draft to the final draft (memoized per graph), and the same `improvement` reward lands
    on writer and editor traces alike — the whole team is rewarded for how much the piece
    improved, which is only measurable after the instance completes.
"""

import asyncio
import logging

import verifiers.v1 as vf
from verifiers.v1.topologies.llm_judge import parse_score

logger = logging.getLogger(__name__)

BRIEFS = [
    "Write the opening paragraph (120-180 words) of a short story about a lighthouse "
    "keeper who receives a letter addressed to someone who died forty years ago.",
    "Write a product announcement (120-180 words) for a kitchen knife that never needs "
    "sharpening, aimed at serious home cooks who distrust gimmicks.",
    "Write a persuasive paragraph (120-180 words) convincing a city council to convert "
    "one downtown parking lot into a pocket park.",
    "Write a plain-language explanation (120-180 words) of why the sky is blue, for a "
    "curious ten-year-old.",
    "Write a cover-letter opening paragraph (120-180 words) for a career switcher moving "
    "from restaurant management into software support.",
    "Write a tense scene (120-180 words) in which two hikers realize the trail markers "
    "they have been following are not official ones.",
]

DRAFT_PROMPT = """{brief}

Reply with the piece alone — no preamble, no commentary."""

CRITIQUE_PROMPT = """You are an editor. A writer produced the draft below for this brief.

<brief>
{brief}
</brief>

<draft>
{draft}
</draft>

Give your single most important piece of feedback: what specifically to change and why it \
would improve the piece. Be concrete — point at words, sentences, structure. A few \
sentences, no rewrite."""

REVISE_PROMPT = """You wrote the draft below for this brief. Your editors have weighed in.

<brief>
{brief}
</brief>

<draft>
{draft}
</draft>

<feedback>
{feedback}
</feedback>

Rewrite the draft, taking the feedback that improves the piece and ignoring what doesn't. \
Stay within the brief. Reply with the revised piece alone — no preamble, no commentary."""

JUDGE_PROMPT = """An editing team revised a piece of writing. Compare the first draft to \
the final draft, for this brief.

<brief>
{brief}
</brief>

<first_draft>
{first}
</first_draft>

<final_draft>
{final}
</final_draft>

Judge how much the final draft improved on the first: quality of prose, fit to the brief, \
and whether the changes are real improvements rather than churn. Think it through, then \
end your reply with a final line of exactly `SCORE: <n>`, where <n> is an integer from 0 \
(worse or unchanged) to 10 (a dramatic, unambiguous improvement)."""


class ImprovementJudge(vf.Judge[float]):
    """One call comparing first draft to final draft — the cheap `vf.Judge` utility tier
    (the judge-as-agent tier is the `llm-judge`/`agentic-judge` topologies)."""

    prompt = JUDGE_PROMPT

    def parse(self, response: vf.JudgeResponse[float]) -> float | None:
        return parse_score(response.text)


class DraftTask(vf.Task):
    """A writing brief (the seed). Carries the brief as a typed field so `go` can re-render
    it into critique and revision tasks without re-parsing the prompt."""

    brief: str


class CritiqueTask(vf.Task):
    """An editing assignment: one draft, one piece of feedback. Minted in `go` from the
    current draft (the forward arrow)."""


class ReviseTask(vf.Task):
    """A revision assignment: the brief, the current draft, and every editor's feedback —
    the fan-in, rendered into one task."""


class WriterEditorsConfig(vf.TopologyConfig):
    writer: vf.DirectAgentConfig = vf.DirectAgentConfig()
    editor: vf.DirectAgentConfig = vf.DirectAgentConfig()
    num_editors: int = 3
    """Editors critiquing each draft (the fan-out width)."""
    num_rounds: int = 1
    """Critique→revise cycles (each round: every editor reads the current draft, the
    writer revises off all their feedback)."""
    judge: vf.JudgeConfig = vf.JudgeConfig()
    """The improvement judge's endpoint/model/sampling (`--topology.judge.model ...`)."""


class WriterEditorsTopology(vf.Topology[WriterEditorsConfig]):
    def __init__(self, config: WriterEditorsConfig) -> None:
        super().__init__(config)
        self.judge = ImprovementJudge(config.judge)
        self._verdicts: dict[str, float] = {}
        """Instance verdicts by graph id: the judge compares first draft to final draft
        once per instance, and every trace's `improvement` reward reads the same value."""

    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: the briefs are baked in, so no `--topology.taskset.id` needed."""
        return [
            DraftTask(idx=i, brief=brief, prompt=DRAFT_PROMPT.format(brief=brief))
            for i, brief in enumerate(BRIEFS)
        ]

    async def go(self, task: DraftTask, run: vf.TopologyRun) -> None:
        """Draft, then `num_rounds` critique→revise cycles: editors fan out over the
        current draft, their feedback fans back in to one revision task."""
        writer = run.agent("writer")
        editor = run.agent("editor")
        draft = await writer.run(task)
        for _ in range(self.config.num_rounds):
            if not draft.last_reply:
                return  # nothing to edit (errored or empty draft)
            critique = CritiqueTask(
                idx=task.idx,
                prompt=CRITIQUE_PROMPT.format(brief=task.brief, draft=draft.last_reply),
            )
            edits = list(
                await asyncio.gather(
                    *(
                        editor.run(critique, parents=[draft])
                        for _ in range(self.config.num_editors)
                    )
                )
            )
            feedback = "\n\n".join(
                f"Editor {i + 1}:\n{edit.last_reply}"
                for i, edit in enumerate(edits)
                if not edit.has_error and edit.last_reply
            )
            if not feedback:
                return  # every editor failed — nothing to revise from
            draft = await writer.run(
                ReviseTask(
                    idx=task.idx,
                    prompt=REVISE_PROMPT.format(
                        brief=task.brief, draft=draft.last_reply, feedback=feedback
                    ),
                ),
                parents=[draft, *edits],
            )

    @vf.reward(agent="writer")
    async def writer_improvement(self, graph: vf.AgentGraph) -> dict[str, float]:
        return {"improvement": await self.verdict(graph)}

    @vf.reward(agent="editor")
    async def editor_improvement(self, graph: vf.AgentGraph) -> dict[str, float]:
        return {"improvement": await self.verdict(graph)}

    async def verdict(self, graph: vf.AgentGraph) -> float:
        """The instance's improvement score: one judge call comparing the writer's first
        draft to its final draft, memoized per graph — instance scoring runs sequentially
        per graph, so the first reward computes it and the rest read it. An instance that
        never produced a revision (or whose judge commits to no verdict) scores 0."""
        if graph.id not in self._verdicts:
            writers = graph.by_agent("writer")
            first, final = writers[0], writers[-1]
            score = None
            if first is not final and first.last_reply and final.last_reply:
                result = await self.judge.evaluate(
                    trace=final,  # the judge call's usage lands on the final draft's trace
                    brief=first.task.brief,
                    first=first.last_reply,
                    final=final.last_reply,
                )
                score = result.parsed
                if score is None:
                    logger.warning(
                        "improvement judge returned no verdict for instance %s",
                        graph.id,
                    )
            self._verdicts[graph.id] = score or 0.0
        return self._verdicts[graph.id]


__all__ = ["WriterEditorsTopology"]
