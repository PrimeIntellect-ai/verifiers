"""kuhn-poker-v1 — seeded Kuhn poker self-play: the turn-coupled proof env.

One env-rollout is one hand: the env deals from the task's seed, opens BOTH seats as
live chat sessions (`agents[...].chat()` — each seat's trace is one real rollout),
and referees the betting host-side — asking the acting seat for its move, validating
it, and paying out the zero-sum result as a `payoff` reward on each seat's trace
(+ANTE/±2 chips; an unparseable move forfeits after `--env.invalid_retries`).

The SPIRAL training shape: both seats default to the run's own model (late binding =
shared-policy self-play; pin `--env.player1.model` for asymmetric play), traces are
role-stamped `player0`/`player1`, and the terminal zero-sum reward plus the
`trace.info["kuhn"]` game facts are exactly what a consumer-side advantage baseline
(per game x role) needs. Seats ride the in-process `direct` harness — a hand costs
just its model calls.
"""

import random
import re
from itertools import count
from typing import Iterator

import verifiers.v1 as vf

RANKS = {"J": 0, "Q": 1, "K": 2}

RULES = """You are playing Kuhn poker as {seat}. Rules:
- Three cards exist: J, Q, K (K beats Q beats J). Each player antes 1 chip and is dealt one private card.
- Player 0 acts first: [check] or [bet] (1 chip).
- After a check, Player 1 may [check] (showdown for the pot) or [bet]; after a bet, the other player may [fold] (bettor takes the pot) or [call] (showdown for the bigger pot).

Your private card: {card}.

Each turn you will be told the betting so far and your legal actions. Reply with your
reasoning if you like, but include EXACTLY ONE action in square brackets, e.g. [check].
Only the bracketed action counts."""

# history (dash-joined actions) -> which seat acts; absent = terminal.
TO_ACT = {"": 0, "check": 1, "bet": 1, "check-bet": 0}
LEGAL = {"": ("check", "bet"), "check": ("check", "bet"), "bet": ("fold", "call"),
         "check-bet": ("fold", "call")}  # fmt: skip


def payoff(history: str, cards: list[str]) -> int:
    """Player 0's net chips at the terminal `history` (zero-sum; player 1 nets the
    negation): +/-1 for a fold or a checked-down showdown, +/-2 for a called bet."""
    showdown = 1 if RANKS[cards[0]] > RANKS[cards[1]] else -1
    return {
        "check-check": showdown,
        "bet-fold": 1,
        "check-bet-fold": -1,
        "bet-call": 2 * showdown,
        "check-bet-call": 2 * showdown,
    }[history]


def parse_action(reply: str, legal: tuple[str, ...]) -> str | None:
    """The one legal bracketed action in the reply, else None. A reply bracketing
    BOTH options ("I'll [bet]... though [check] would...") is ambiguous whichever
    end you read from, so it consumes an invalid-move retry ("reply with exactly
    one") instead of being silently played."""
    found = {a for a in re.findall(r"\[(\w+)\]", reply.lower()) if a in legal}
    return found.pop() if len(found) == 1 else None


class KuhnPokerData(vf.TaskData):
    info: dict
    """The hand's RNG seed — the deal is reproducible from it."""


class KuhnPokerTask(vf.Task[KuhnPokerData]):
    pass


class KuhnPokerConfig(vf.TasksetConfig):
    pass


class KuhnPokerParams(vf.EnvParams):
    player0: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="direct"))
    player1: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="direct"))
    invalid_retries: int = 1
    """Re-prompts (with feedback) before an unparseable move forfeits the hand."""


class KuhnPokerEnv(vf.Environment[KuhnPokerParams]):
    def roles(self):
        return {"player0": self.params.player0, "player1": self.params.player1}

    async def rollout(self, task, agents):
        rng = random.Random(task.data.info["seed"])
        cards = rng.sample(list(RANKS), 2)
        seat_tasks = [
            vf.Task(
                vf.TaskData(
                    idx=task.data.idx,
                    prompt=None,  # each seat converses through its chat session
                    system_prompt=RULES.replace("{seat}", f"Player {i}").replace(
                        "{card}", cards[i]
                    ),
                )
            )
            for i in (0, 1)
        ]
        history: list[str] = []
        forfeited: int | None = None  # the seat that failed to produce a legal move

        async def ask(session, seat: int) -> str | None:
            told = "nothing yet" if not history else ", ".join(
                f"Player {TO_ACT['-'.join(history[:j])]} chose [{a}]"
                for j, a in enumerate(history)
            )  # fmt: skip
            legal = LEGAL["-".join(history)]
            prompt = (
                f"Betting so far: {told}. You are Player {seat}. "
                f"Your legal actions: {' or '.join(f'[{a}]' for a in legal)}."
            )
            for _ in range(self.params.invalid_retries + 1):
                reply = await session.turn(prompt)
                if reply.stopped:
                    return None
                action = parse_action(reply.text, legal)
                if action is not None:
                    return action
                prompt = (
                    "That was not a legal move. Reply with exactly one of "
                    f"{' or '.join(f'[{a}]' for a in legal)} in square brackets."
                )
            return None

        async with (
            agents["player0"].chat(seat_tasks[0]) as s0,
            agents["player1"].chat(seat_tasks[1]) as s1,
        ):
            sessions = [s0, s1]
            while (key := "-".join(history)) in TO_ACT:
                seat = TO_ACT[key]
                action = await ask(sessions[seat], seat)
                if action is None:
                    forfeited = seat
                    break
                history.append(action)
        traces = [s0.trace, s1.trace]
        net0 = (
            payoff("-".join(history), cards)
            if forfeited is None
            else (1 if forfeited == 1 else -1)
        )
        for i, trace in enumerate(traces):
            trace.record_reward("payoff", float(net0 if i == 0 else -net0))
            trace.record_metric("forfeit", float(forfeited == i))
            trace.info["kuhn"] = {
                "seat": i,
                "card": cards[i],
                "history": "-".join(history),
                "seed": task.data.info["seed"],
                "forfeited": forfeited,
            }
        return traces


class KuhnPokerTaskset(vf.Taskset[KuhnPokerTask, KuhnPokerConfig]):
    INFINITE = True

    def load(self) -> Iterator[KuhnPokerTask]:
        for i in count():
            yield KuhnPokerTask(
                KuhnPokerData(idx=i, name=f"hand#{i}", prompt=None, info={"seed": i})
            )
