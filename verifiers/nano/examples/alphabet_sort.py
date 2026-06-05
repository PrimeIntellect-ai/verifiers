"""alphabet-sort: a multi-turn environment with a user simulator.

The model sorts names alphabetically; a `User` then feeds follow-up rounds with
more names to fold in. Each turn's <sorted> block is scored against that turn's
ground truth, and the reward is the average over turns. `follow_ups` and
`ground_truths` are typed fields on the Task subclass — they flow into both the
user simulator and the reward without any dict access.
"""

import re
from difflib import SequenceMatcher

import verifiers.nano as vf

TAG = re.compile(r"<sorted>(.*?)</sorted>", re.DOTALL)
FORMAT = "Put your answer inside <sorted> tags, one name per line."
INSTRUCTION = "Sort the given names alphabetically."

# (id, initial names, follow-up name sets, per-turn ground truths)
RAW_TASKS = [
    (
        "0",
        "Carol, Alice, Bob",
        ["Now fold these in and re-sort the full list: Dave, Eve"],
        [["Alice", "Bob", "Carol"], ["Alice", "Bob", "Carol", "Dave", "Eve"]],
    ),
    ("1", "Zara, Mia", [], [["Mia", "Zara"]]),
    (
        "2",
        "Tom, Ann, Sue",
        ["Now fold these in and re-sort the full list: Ben"],
        [["Ann", "Sue", "Tom"], ["Ann", "Ben", "Sue", "Tom"]],
    ),
]


class AlphabetSortTask(vf.Task):
    follow_ups: list[str]
    """Subsequent user turns; each adds names to fold into the sort."""
    ground_truths: list[list[str]]
    """The correct sorted list after each turn (turn 1 first)."""


class AlphabetUser(vf.User):
    async def get_response(
        self, task: AlphabetSortTask, transcript: vf.Transcript, messages: vf.Messages
    ) -> list[vf.UserMessage]:
        assistant_turns = sum(1 for m in messages if m.role == "assistant")
        if 0 < assistant_turns <= len(task.follow_ups):
            return [vf.UserMessage(content=task.follow_ups[assistant_turns - 1])]
        return []


class AlphabetSortTaskset(vf.Taskset[AlphabetSortTask, vf.TasksetConfig]):
    def __init__(self, config: vf.TasksetConfig) -> None:
        super().__init__(config)
        self.user = AlphabetUser(vf.UserConfig())

    def load_tasks(self) -> list[AlphabetSortTask]:
        return [
            AlphabetSortTask(
                id=task_id,
                instruction=f"{INSTRUCTION} {FORMAT}\n\nNames: {names}",
                follow_ups=[f"{follow_up} {FORMAT}" for follow_up in follow_ups],
                ground_truths=ground_truths,
            )
            for task_id, names, follow_ups, ground_truths in RAW_TASKS
        ]

    @vf.reward(weight=1.0)
    async def per_turn_match(
        self, task: AlphabetSortTask, transcript: vf.Transcript
    ) -> float:
        answers = [
            m.content or "" for m in transcript.messages if m.role == "assistant"
        ]
        scores: list[float] = []
        for turn, expected in enumerate(task.ground_truths):
            if turn >= len(answers):
                break
            match = TAG.search(answers[turn])
            block = match.group(1) if match else ""
            predicted = [line.strip() for line in block.splitlines() if line.strip()]
            scores.append(
                SequenceMatcher(
                    None,
                    "\n".join(p.lower() for p in predicted),
                    "\n".join(e.lower() for e in expected),
                ).ratio()
            )
        return sum(scores) / len(task.ground_truths) if task.ground_truths else 0.0


class EnvConfig(vf.EnvConfig):
    harness: vf.HarnessConfig = vf.HarnessConfig(max_turns=4)


def load_taskset(config: vf.TasksetConfig | None = None) -> AlphabetSortTaskset:
    return AlphabetSortTaskset(config or vf.TasksetConfig())


def load_harness(config: vf.HarnessConfig | None = None) -> vf.Harness:
    return vf.Harness(config or vf.HarnessConfig(max_turns=4))


def load_environment(config: EnvConfig | None = None) -> vf.Environment:
    config = config or EnvConfig()
    return vf.Environment(
        taskset=load_taskset(config.taskset),
        harness=load_harness(config.harness),
    )
