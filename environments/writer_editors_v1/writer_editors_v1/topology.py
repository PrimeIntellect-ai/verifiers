"""writer-editors-v1 — a writer drafts, n editors critique, the writer revises (example topology).

The rounds + fan-in shape, one self-contained package:

  - **Rounds are a loop in `run`**: each round, the current draft goes out to the editors
    and comes back revised — `num_rounds` cycles, plain Python.
  - **A list role**: `editors: list[DirectAgentConfig]` declares one role with several
    seats — `asyncio.gather(*(editor.run(critique, parents=[draft]) for editor in
    agents.editors))` fans the same draft out to every seat, concurrently (and
    per-seat config makes mixed-model editorial boards a TOML change).
  - **Fan-in**: the revision task is built from *all* the editors' traces at once, and the
    revised trace is linked under every one of them (`parents=[draft, *edits]`).
  - **One shared reward, every trace**: a deterministic comparison of the writer's first
    draft to its final draft (closer to the brief's 120–180 word band is better). The same
    `improvement` value lands on writer and editor traces alike — the whole team is
    rewarded for a measurable graph-level fact, with no judge agent and no Topology-side
    memoization (pure function of the finished graph; cheap to recompute).
"""

import asyncio

import verifiers.v1 as vf

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

# Briefs ask for 120–180 words; the toy reward scores how close a draft sits to that band.
_TARGET_LO, _TARGET_HI = 120, 180
_TARGET_MID = (_TARGET_LO + _TARGET_HI) / 2


def _band_score(text: str) -> float:
    """1.0 inside the brief's word band; linear falloff outside it."""
    n = len(text.split())
    if _TARGET_LO <= n <= _TARGET_HI:
        return 1.0
    if n < _TARGET_LO:
        return max(0.0, n / _TARGET_LO)
    return max(0.0, 1.0 - (n - _TARGET_HI) / _TARGET_MID)


def improvement(graph: vf.AgentGraph) -> float:
    """How much the final draft improved on the first toward the brief's word band.

    0 when there was no revision (or empty text). Pure graph read — safe to call from
    every agent-scoped reward without caching.
    """
    writers = graph.by_agent("writer")
    if len(writers) < 2:
        return 0.0
    first, final = writers[0], writers[-1]
    if not first.last_reply or not final.last_reply:
        return 0.0
    if first.last_reply.strip() == final.last_reply.strip():
        return 0.0
    return max(0.0, _band_score(final.last_reply) - _band_score(first.last_reply))


class DraftData(vf.TaskData):
    """A writing brief (the seed). Carries the brief as a typed field so `go` can re-render
    it into critique and revision tasks without re-parsing the prompt."""

    brief: str


class DraftTask(vf.Task[DraftData]):
    """The task wrapping a `DraftData` brief row."""


class CritiqueTask(vf.Task[vf.TaskData]):
    """An editing assignment: one draft, one piece of feedback. Minted in `go` from the
    current draft (the forward arrow)."""


class ReviseTask(vf.Task[vf.TaskData]):
    """A revision assignment: the brief, the current draft, and every editor's feedback —
    the fan-in, rendered into one task."""


class WriterEditorsConfig(vf.TopologyConfig):
    writer: vf.DirectAgentConfig = vf.DirectAgentConfig()
    editors: list[vf.DirectAgentConfig] = [vf.DirectAgentConfig() for _ in range(3)]
    """The editorial board: one role, one seat per entry (the fan-out width). Per-seat
    per-seat overrides via `[[topology.editors]]` array-of-tables in TOML."""
    num_rounds: int = 1
    """Critique→revise cycles (each round: every editor reads the current draft, the
    writer revises off all their feedback)."""


class WriterEditorsTopology(vf.Topology[WriterEditorsConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: the briefs are baked in, so no `--topology.taskset.id` needed."""
        return [
            DraftTask(
                DraftData(idx=i, brief=brief, prompt=DRAFT_PROMPT.format(brief=brief))
            )
            for i, brief in enumerate(BRIEFS)
        ]

    async def run(self, task: DraftTask, agents: vf.Agents) -> None:
        """Draft, then `num_rounds` critique→revise cycles: editors fan out over the
        current draft, their feedback fans back in to one revision task."""
        writer = agents.writer
        draft = await writer.run(task)
        for _ in range(self.config.num_rounds):
            if not draft.last_reply:
                return  # nothing to edit (errored or empty draft)
            critique = CritiqueTask(
                vf.TaskData(
                    idx=task.data.idx,
                    prompt=CRITIQUE_PROMPT.format(
                        brief=task.data.brief, draft=draft.last_reply
                    ),
                )
            )
            edits = list(
                await asyncio.gather(
                    *(
                        editor.run(critique, parents=[draft])
                        for editor in agents.editors
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
                    vf.TaskData(
                        idx=task.data.idx,
                        prompt=REVISE_PROMPT.format(
                            brief=task.data.brief,
                            draft=draft.last_reply,
                            feedback=feedback,
                        ),
                    )
                ),
                parents=[draft, *edits],
            )

    @vf.reward(agent="writer")
    async def writer_improvement(self, graph: vf.AgentGraph) -> dict[str, float]:
        return {"improvement": improvement(graph)}

    @vf.reward(agent="editors")
    async def editor_improvement(self, graph: vf.AgentGraph) -> dict[str, float]:
        return {"improvement": improvement(graph)}


__all__ = ["WriterEditorsTopology"]
