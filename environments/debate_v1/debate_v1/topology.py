"""debate-v1 — n debaters argue a motion through concurrent live sessions (example topology).

The N-ary session shape: every seat's episode stays open for the whole debate, and `go`
is the moderator — it broadcasts the room to each seat (`view` controls exactly what each
agent sees) and gathers whole rounds *concurrently* (`asyncio.gather` over suspended
sessions: opening statements, rebuttals, and the final vote each run N-wide at once).
Each debater's trace is one multi-turn trajectory: its own statements as sampled turns,
the rest of the room arriving as its user turns.

Judgement is peer voting: after the rounds, every seat votes for the most convincing
OTHER debater; a seat's reward is the share of votes it received (stamped by `go` as
data, read by the declared `support` reward at instance end). No judge model, no ground
truth — the interaction scores itself.

    uv run eval --topology.id debate-v1 -n 1 --max-turns 8 --timeout.rollout 600
"""

import asyncio
import contextlib
import re

import verifiers.v1 as vf

MOTIONS = [
    "Cities should remove minimum parking requirements for new housing.",
    "Schools should replace most homework with in-class practice.",
    "Open-source licenses should require attribution for AI training use.",
]

SEAT_SYSTEM = """You are debater {seat} of {n} in a structured debate. Be persuasive but \
honest; engage the strongest version of the other side. Keep every statement under 120 \
words."""

OPENING = """Motion: {motion}

You are debater {seat}. {stance} Give your opening statement."""

REBUTTAL = """The other debaters said:

{others}

Give your rebuttal: engage their strongest specific points, keep your position."""

VOTE = """The debate is over. Full record:

{record}

Vote for the single most convincing debater OTHER than yourself (you are debater \
{seat}). Reply with exactly one line: `VOTE: <number>`."""

_VOTE = re.compile(r"VOTE:\s*(\d+)")


def parse_vote(reply: str, seat: int, n: int) -> int | None:
    """The voted seat on the last `VOTE:` line — refused if it's out of range or self."""
    votes = _VOTE.findall(reply.upper())
    if not votes:
        return None
    vote = int(votes[-1])
    return vote if 0 <= vote < n and vote != seat else None


class DebaterTask(vf.Task):
    """One debater's episode: framing only (`prompt=None`; sessions open on the first
    `turn()`). The seat is data so the record and rewards stay attributable."""

    seat: int


class DebateConfig(vf.TopologyConfig):
    debater: vf.DirectAgentConfig = vf.DirectAgentConfig()
    """One config, N seats — every debater is an episode of this agent
    (`--topology.debater.model <id>` moves the whole panel)."""
    num_debaters: int = 4
    num_rounds: int = 1
    """Rebuttal rounds after the opening statements."""


class DebateTopology(vf.Topology[DebateConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: one instance per motion."""
        return [vf.Task(idx=i, prompt=motion) for i, motion in enumerate(MOTIONS)]

    async def go(self, task: vf.Task, run: vf.TopologyRun) -> None:
        n = self.config.num_debaters
        motion = task.prompt_text
        async with contextlib.AsyncExitStack() as stack:
            seats = [
                await stack.enter_async_context(
                    run.agent("debater").interact(
                        DebaterTask(
                            idx=task.idx,
                            seat=i,
                            prompt=None,
                            system_prompt=SEAT_SYSTEM.format(seat=i, n=n),
                        )
                    )
                )
                for i in range(n)
            ]
            # Alternate stances so the panel genuinely disagrees.
            stances = [
                "Argue FOR the motion." if i % 2 == 0 else "Argue AGAINST the motion."
                for i in range(n)
            ]
            # Opening statements — N suspended episodes generating concurrently.
            statements = list(
                await asyncio.gather(
                    *(
                        seat.turn(
                            OPENING.format(motion=motion, seat=i, stance=stances[i])
                        )
                        for i, seat in enumerate(seats)
                    )
                )
            )
            record = [f"Debater {i} (opening): {s}" for i, s in enumerate(statements)]
            for _ in range(self.config.num_rounds):
                # Each seat's view: everyone's latest statement but its own.
                statements = list(
                    await asyncio.gather(
                        *(
                            seat.turn(
                                REBUTTAL.format(
                                    others="\n\n".join(
                                        f"Debater {j}: {statements[j]}"
                                        for j in range(n)
                                        if j != i
                                    )
                                )
                            )
                            for i, seat in enumerate(seats)
                        )
                    )
                )
                record += [
                    f"Debater {i} (rebuttal): {s}" for i, s in enumerate(statements)
                ]
            # Peer vote — concurrent again; go tallies (the moderator owns the rules).
            ballots = await asyncio.gather(
                *(
                    seat.turn(VOTE.format(record="\n\n".join(record), seat=i))
                    for i, seat in enumerate(seats)
                )
            )
            votes = [
                parse_vote(ballot, seat=i, n=n) for i, ballot in enumerate(ballots)
            ]
            for i, seat in enumerate(seats):
                seat.trace.info["debate"] = {
                    "seat": i,
                    "stance": stances[i],
                    "votes_received": votes.count(i),
                    "voted_validly": votes[i] is not None,
                }

    @vf.reward(agent="debater")
    async def support(self, trace: vf.Trace) -> dict[str, float]:
        """Share of the peer vote this seat won — the interaction scoring itself."""
        n = self.config.num_debaters
        return {"support": trace.info["debate"]["votes_received"] / max(n - 1, 1)}

    @vf.metric(agent="debater")
    async def voted_validly(self, trace: vf.Trace) -> float:
        """Whether this seat cast a well-formed, non-self vote."""
        return float(trace.info["debate"]["voted_validly"])


__all__ = ["DebateTopology"]
