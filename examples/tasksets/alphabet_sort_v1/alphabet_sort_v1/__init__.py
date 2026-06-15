"""alphabet-sort-v1 — maintain an alphabetically sorted list of names across turns.

The v1 port of the `alphabet-sort` environment. Each task is a multi-turn episode: the model
sorts an initial list of names (by first or last name) into `<alphabetical_sorted>` tags, then
on each follow-up turn re-sorts the cumulative list — tagging the newly added names — into
`<combined_alphabetical_sorted>` tags. The reward is the per-turn sequence similarity to the
ground truth, power-scaled.

The follow-up turns are supplied by a `vf.User` simulator (`AlphabetSortUser` below): the
interception server drives the simulator after every assistant turn and injects the next
follow-up as a user message, so the whole episode is one rollout the harness only ever sees as
a single exchange. The simulator runs on the host (not colocated in the agent's runtime), so
the host-driven loop reaches it on every runtime. The episodes are pre-generated in
`load_tasks`; the simulator replays them.
"""

import difflib
import random
import re
from typing import Literal

from datasets import load_dataset
from pydantic import PrivateAttr

import verifiers.v1 as vf

DATASET = "kalomaze/alphabetic-arxiv-authors-it1"
SEED = 1337420


class AlphabetSortConfig(vf.TasksetConfig):
    min_turns: int = 1
    """Minimum number of turns (assistant sorts) per episode."""
    max_turns: int = 3
    """Maximum number of turns per episode; each draws a count in [min_turns, max_turns]."""
    min_names_per_turn: int = 1
    """Minimum number of names introduced on each turn."""
    max_names_per_turn: int = 5
    """Maximum number of names introduced on each turn."""
    similarity_power: int = 4
    """Exponent applied to each turn's sequence-similarity score (higher = sharper penalty)."""
    power_per_turn: bool = True
    """Power-scale each turn then average (True), or average raw similarities then power once (False)."""
    split: Literal["train"] = "train"
    """Split of the source author-names dataset to build the episodes from."""


class AlphabetSortTask(vf.Task):
    info: dict
    """The pre-generated episode: the `follow_ups` the user simulator reveals turn by turn, the
    per-turn `ground_truths` the reward grades against, and `num_turns`."""


class AlphabetSortUser(vf.User):
    """Replays the episode's pre-generated follow-up turns: one `respond` per assistant turn,
    injecting the next follow-up as a user message until all turns are done. This colocates
    v0's `MultiTurnEnv.env_response` with the agent — the framework drives it, never the model."""

    follow_ups: list[str]
    num_turns: int
    _turns: int = PrivateAttr(default=0)

    async def respond(self, message: str) -> tuple[vf.Messages, bool]:
        self._turns += 1
        if self._turns >= self.num_turns:
            return [], True
        return [{"role": "user", "content": self.follow_ups[self._turns - 1]}], False


class AlphabetSortTaskset(vf.Taskset[AlphabetSortTask, AlphabetSortConfig]):
    def load_tasks(self) -> list[AlphabetSortTask]:
        c = self.config
        assert 1 <= c.min_turns <= c.max_turns, "need 1 <= min_turns <= max_turns"
        assert 1 <= c.min_names_per_turn <= c.max_names_per_turn, (
            "need 1 <= min_names_per_turn <= max_names_per_turn"
        )
        rng = random.Random(SEED)
        tasks: list[AlphabetSortTask] = []
        for entry in load_dataset(DATASET, split=c.split):
            names = list(dict.fromkeys(n.replace(" ", "") for n in entry["names"]))
            counts = [
                rng.randint(c.min_names_per_turn, c.max_names_per_turn)
                for _ in range(rng.randint(c.min_turns, c.max_turns))
            ]
            if len(names) < sum(counts):
                continue
            by_first = rng.choice([True, False])
            label = "FIRST" if by_first else "LAST"

            def sort_key(s: str) -> str:
                # split at the first capital after index 0 -> first- vs last-name part
                cut = next((i for i in range(1, len(s)) if s[i].isupper()), len(s))
                return s[:cut] if by_first else s[cut:]

            turns, cumulative, ground_truths, i = [], [], [], 0
            for count in counts:
                turn = names[i : i + count]
                i += count
                turns.append(turn)
                cumulative += turn
                ranked = sorted(cumulative, key=sort_key)
                ground_truths.append(
                    ranked
                    if len(turns) == 1
                    else [f"{x} // new name!" if x in turn else x for x in ranked]
                )

            first = turns[0][:]
            rng.shuffle(first)
            shown = rng.randint(c.min_names_per_turn, c.max_names_per_turn)
            instruction = (
                f"Sort these names in alphabetical order by {label} name: {', '.join(first)}\n\n"
                "Use exactly this format:\n<alphabetical_sorted>\n"
                + "\n".join(f"Name{j}" for j in range(1, shown + 1))
                + "\n</alphabetical_sorted>"
            )

            follow_ups = []
            for t in range(1, len(turns)):
                shuffled = turns[t][:]
                rng.shuffle(shuffled)
                shown = rng.randint(
                    c.min_names_per_turn, sum(len(x) for x in turns[: t + 1])
                )
                threshold = rng.randint(0, shown - 1)
                prompt = (
                    f"Now sort ALL of these names alphabetically by {label} name: {', '.join(shuffled)}\n\n"
                    "These are in addition to the prior list. Mark any NEW names (that weren't "
                    "in the prior list) with `// new name!` at the end."
                )
                if t == 1:
                    prompt += (
                        "\n\nUse exactly this format:\n<combined_alphabetical_sorted>\n"
                        + "\n".join(
                            f"Name{j}" + (" // new name!" if j > threshold else "")
                            for j in range(1, shown + 1)
                        )
                        + "\n</combined_alphabetical_sorted>"
                    )
                else:
                    prompt += " Follow the same format as before."
                follow_ups.append(prompt)

            tasks.append(
                AlphabetSortTask(
                    idx=len(tasks),
                    instruction=instruction,
                    info={
                        "follow_ups": follow_ups,
                        "ground_truths": ground_truths,
                        "num_turns": len(turns),
                    },
                )
            )
        return tasks

    def user(self, task: AlphabetSortTask) -> vf.User:
        return AlphabetSortUser(
            name="user",
            follow_ups=task.info["follow_ups"],
            num_turns=task.info["num_turns"],
        )

    @vf.reward(weight=1.0)
    async def alphabet_sort(self, task: AlphabetSortTask, trace: vf.Trace) -> float:
        c = self.config
        ground_truths = task.info["ground_truths"]
        num_turns = task.info["num_turns"]
        responses = [m.content or "" for m in trace.assistant_messages]
        scores = []
        for t in range(num_turns):
            tag = "alphabetical_sorted" if t == 0 else "combined_alphabetical_sorted"
            response = responses[t] if t < len(responses) else ""
            expected = "\n".join(s.strip().lower() for s in ground_truths[t])
            # Multiple <tag> attempts only count if they strictly improve (else 0).
            attempts = []
            for content in re.findall(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL):
                pred = "\n".join(
                    ln.strip().lower() for ln in content.split("\n") if ln.strip()
                )
                sim = (
                    difflib.SequenceMatcher(None, pred, expected).ratio()
                    if pred and expected
                    else 0.0
                )
                attempts.append(sim**c.similarity_power if c.power_per_turn else sim)
            if not attempts:
                scores.append(0.0)
            elif len(attempts) == 1:
                scores.append(attempts[0])
            else:
                improved = all(b > a for a, b in zip(attempts, attempts[1:]))
                scores.append(attempts[-1] if improved else 0.0)
        avg = sum(scores) / num_turns if num_turns else 0.0
        return avg if c.power_per_turn else avg**c.similarity_power


def load_taskset(config: AlphabetSortConfig) -> AlphabetSortTaskset:
    return AlphabetSortTaskset(config)
