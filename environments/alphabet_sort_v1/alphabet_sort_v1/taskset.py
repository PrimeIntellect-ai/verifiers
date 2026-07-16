"""alphabet-sort-v1 — maintain an alphabetically sorted list of names across turns.

The v1 port of the `alphabet-sort` environment. Each task is a multi-turn episode: the model
sorts an initial list of names (by first or last name) into `<alphabetical_sorted>` tags, then
on each follow-up turn re-sorts the cumulative list — tagging the newly added names — into
`<combined_alphabetical_sorted>` tags. The reward is the per-turn sequence similarity to the
ground truth, power-scaled.

The whole conversation is driven by a scripted user: the env's `rollout()` replays the
episode's pre-generated `user_turns` through `user=` — a plain closure. The task carries no
prompt (`prompt=None`), so the closure also opens the conversation with the initial sort
prompt — before the model is ever called — and then supplies each follow-up between
segments (the assistant yields, the closure answers, the exchange resumes); when the turns
run out it returns no messages, which ends the exchange. The taskset is
`INFINITE`: `load` generates episodes on demand, forever — each pass over the source name
lists draws fresh turn splits, so the stream never repeats; runs bound it with `-n`.
"""

import difflib
import random
import re
from collections.abc import Iterator
from typing import Literal

from datasets import load_dataset

import verifiers.v1 as vf

DATASET = "kalomaze/alphabetic-arxiv-authors-it1"
SEED = 1337420


class AlphabetSortTaskConfig(vf.TaskConfig):
    similarity_power: int = 4
    """Exponent applied to each turn's sequence-similarity score (higher = sharper penalty)."""
    power_per_turn: bool = True
    """Power-scale each turn then average (True), or average raw similarities then power once (False)."""


class AlphabetSortConfig(vf.TasksetConfig):
    min_turns: int = 1
    """Minimum number of turns (assistant sorts) per episode."""
    max_turns: int = 3
    """Maximum number of turns per episode; each draws a count in [min_turns, max_turns]."""
    min_names_per_turn: int = 1
    """Minimum number of names introduced on each turn."""
    max_names_per_turn: int = 5
    """Maximum number of names introduced on each turn."""
    split: Literal["train"] = "train"
    """Split of the source author-names dataset to build the episodes from."""
    task: AlphabetSortTaskConfig = AlphabetSortTaskConfig()


class AlphabetSortTaskData(vf.TaskData):
    info: dict
    """The pre-generated episode: the `user_turns` the env's scripted user reveals one by one
    (the opening sort prompt, then the follow-ups), the per-turn `ground_truths` the reward
    grades against, and `num_turns`. The row itself carries no prompt — the scripted user
    opens the conversation."""


class AlphabetSortTask(vf.Task[AlphabetSortTaskData, vf.State, AlphabetSortTaskConfig]):
    @vf.reward(weight=1.0)
    async def alphabet_sort(self, trace: vf.Trace) -> float:
        ground_truths = self.data.info["ground_truths"]
        num_turns = self.data.info["num_turns"]
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
                attempts.append(
                    sim**self.config.similarity_power
                    if self.config.power_per_turn
                    else sim
                )
            if not attempts:
                scores.append(0.0)
            elif len(attempts) == 1:
                scores.append(attempts[0])
            else:
                improved = all(b > a for a, b in zip(attempts, attempts[1:]))
                scores.append(attempts[-1] if improved else 0.0)
        avg = sum(scores) / num_turns if num_turns else 0.0
        return avg if self.config.power_per_turn else avg**self.config.similarity_power


class AlphabetSortEnv(vf.Environment):
    """Replays each episode's pre-generated user turns as the run's user."""

    async def rollout(self, task, agents):
        # A chat session replaying the pre-generated episode: the task carries no
        # prompt, so the first turn opens the conversation with the initial sort.
        async with agents["solver"].chat(task) as session:
            for prompt in task.data.info["user_turns"]:
                if (await session.turn(prompt)).stopped:
                    break
        return [session.trace]


class AlphabetSortTaskset(vf.Taskset[AlphabetSortTask, AlphabetSortConfig]):
    INFINITE = True

    def load(self) -> Iterator[AlphabetSortTask]:
        c = self.config
        assert 1 <= c.min_turns <= c.max_turns, "need 1 <= min_turns <= max_turns"
        assert 1 <= c.min_names_per_turn <= c.max_names_per_turn, (
            "need 1 <= min_names_per_turn <= max_names_per_turn"
        )
        rng = random.Random(SEED)
        entries = load_dataset(DATASET, split=c.split)
        idx = 0
        # Cycle the source name lists forever; the rng advances across passes, so every
        # pass draws fresh turn splits and the episode stream never repeats.
        while True:
            pass_start = idx
            for entry in entries:
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
                initial_prompt = (
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

                yield AlphabetSortTask(
                    AlphabetSortTaskData(
                        idx=idx,
                        # No prompt on the row: the scripted user opens with the sort prompt,
                        # then the follow-ups — one user turn per `user_turns` entry.
                        prompt=None,
                        info={
                            "user_turns": [initial_prompt, *follow_ups],
                            "ground_truths": ground_truths,
                            "num_turns": len(turns),
                        },
                    ),
                    c.task,
                )
                idx += 1
            if idx == pass_start:
                raise ValueError(
                    "no source name list is long enough for the configured turns - "
                    "lower min/max_turns or min/max_names_per_turn"
                )
