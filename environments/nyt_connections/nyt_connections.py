import json
import random
import re
from functools import lru_cache
from urllib.request import urlopen

import verifiers as vf

NYT_CONNECTIONS_URL = (
    "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/refs/heads/main/"
    "connections.json"
)

NYT_CONNECTIONS_SYSTEM_PROMPT = """You are playing NYT Connections.

Rules:
- The board has 16 words split into 4 hidden groups of 4.
- Each hidden group has a shared theme.
- You have 4 lives; incorrect or invalid guesses cost 1 life.
- Each turn, reason briefly and submit exactly 4 comma-separated words.
- Put your final guess for the turn inside <guess>...</guess> tags.

Example format:
<think>These four words are all wet weather events.</think>
<guess>HAIL, RAIN, SLEET, SNOW</guess>"""


def normalize_word(word: str) -> str:
    return " ".join(str(word).upper().split())


def parse_guess(content: str) -> list[str] | None:
    matches = re.findall(r"<guess>\s*(.*?)\s*</guess>", content, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    words = [normalize_word(word) for word in matches[-1].split(",") if word.strip()]
    if len(words) != 4 or len(set(words)) != 4:
        return None
    return words


def normalize_group(group: dict) -> set[str]:
    return {normalize_word(str(word)) for word in group["members"]}


def format_group(group: dict) -> str:
    return f"{group['group']}: {', '.join(normalize_word(str(word)) for word in group['members'])}"


def format_board(game: dict) -> str:
    lines: list[str] = []
    if game["found_groups"]:
        lines.append("SOLVED GROUPS:")
        lines.extend(format_group(group) for group in game["found_groups"])
        lines.append("")
        lines.append("REMAINING WORDS:")
    else:
        lines.append("WORDS ON THE BOARD:")
    lines.append(", ".join(game["remaining_words"]))
    lines.append("")
    lines.append(f"Lives remaining: {game['lives']}")
    return "\n".join(lines)


def make_game(task: vf.Task) -> dict:
    groups = [dict(group) for group in task["groups"]]
    remaining_words = [word for group in groups for word in normalize_group(group)]
    random.Random(int(task["seed"])).shuffle(remaining_words)
    return {
        "groups": groups,
        "remaining_words": remaining_words,
        "found_groups": [],
        "lives": 4,
        "turns": 0,
        "invalid_guesses": 0,
        "incorrect_guesses": 0,
    }


@lru_cache(maxsize=4)
def fetch_connections_rows(dataset_url: str) -> list[dict]:
    with urlopen(dataset_url, timeout=30) as response:
        rows = json.loads(response.read().decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("NYT Connections dataset must be a list of puzzles.")
    return rows


def build_tasks(
    dataset_url: str,
    seed: int,
    max_turns: int,
    num_train_examples: int,
    num_eval_examples: int,
) -> tuple[list[vf.ConfigData], list[vf.ConfigData]]:
    rows = fetch_connections_rows(dataset_url)
    tasks: list[vf.ConfigData] = []
    for row in rows:
        groups = row.get("answers")
        if not isinstance(groups, list) or len(groups) != 4:
            continue
        if any(not isinstance(group, dict) or len(group.get("members", [])) != 4 for group in groups):
            continue

        puzzle_id = int(row["id"])
        puzzle_seed = seed + puzzle_id
        words = [word for group in groups for word in normalize_group(group)]
        random.Random(puzzle_seed).shuffle(words)
        question = (
            "Find the four NYT Connections groups. Submit each turn as "
            "<guess>WORD1, WORD2, WORD3, WORD4</guess>.\n\n"
            "WORDS ON THE BOARD:\n"
            f"{', '.join(words)}\n\nLives remaining: 4"
        )
        tasks.append(
            {
                "example_id": puzzle_id,
                "prompt": [{"role": "user", "content": question}],
                "answer": json.dumps(groups),
                "groups": groups,
                "date": str(row.get("date", "")),
                "seed": puzzle_seed,
                "max_turns": max_turns,
                "info": {
                    "date": str(row.get("date", "")),
                    "dataset_url": dataset_url,
                    "groups": groups,
                },
            }
        )

    rng = random.Random(seed)
    rng.shuffle(tasks)
    train = tasks[:num_train_examples]
    eval_start = num_train_examples
    eval_end = eval_start + num_eval_examples
    eval_tasks = tasks[eval_start:eval_end]
    return train, eval_tasks


async def connections_user(
    task: vf.Task, state: vf.State, messages: list[dict]
) -> list[dict[str, str]]:
    game = state.setdefault("connections", make_game(task))
    assistant_messages = vf.get_messages(messages, role="assistant")
    if not assistant_messages:
        return []

    game["turns"] += 1
    content = str(assistant_messages[-1].content or "")
    guess = parse_guess(content)
    if guess is None:
        game["lives"] -= 1
        game["invalid_guesses"] += 1
        response = (
            "Invalid guess. Use exactly four comma-separated remaining board words "
            "inside <guess>...</guess>."
        )
    elif not set(guess).issubset(set(game["remaining_words"])):
        game["lives"] -= 1
        game["invalid_guesses"] += 1
        response = "Invalid guess. Every guessed word must still be on the board."
    else:
        guess_set = set(guess)
        matched_group = next(
            (
                group
                for group in game["groups"]
                if group not in game["found_groups"] and guess_set == normalize_group(group)
            ),
            None,
        )
        if matched_group is None:
            game["lives"] -= 1
            game["incorrect_guesses"] += 1
            response = "Incorrect guess."
        else:
            game["found_groups"].append(matched_group)
            found_words = normalize_group(matched_group)
            game["remaining_words"] = [
                word for word in game["remaining_words"] if word not in found_words
            ]
            response = f"Correct! You found {format_group(matched_group)}."

    if len(game["found_groups"]) == 4:
        state.stop("solved")
        return [
            {
                "role": "user",
                "content": response + "\n\nCongratulations, you solved the puzzle!",
            }
        ]

    if game["lives"] <= 0:
        state.stop("out_of_lives")
        answers = "\n".join(format_group(group) for group in game["groups"])
        return [
            {
                "role": "user",
                "content": response + "\n\nGame over. Correct groups:\n" + answers,
            }
        ]

    return [{"role": "user", "content": response + "\n\n" + format_board(game)}]


@vf.reward(weight=1.0)
async def success_reward(task: vf.Task, state: vf.State) -> float:
    _ = task
    game = state.get("connections") or {}
    return 1.0 if len(game.get("found_groups", [])) == 4 else 0.0


@vf.reward(weight=0.3)
async def progress_reward(task: vf.Task, state: vf.State) -> float:
    _ = task
    game = state.get("connections") or {}
    return len(game.get("found_groups", [])) / 4


@vf.reward(weight=0.2)
async def efficiency_reward(task: vf.Task, state: vf.State) -> float:
    _ = task
    game = state.get("connections") or {}
    if len(game.get("found_groups", [])) != 4:
        return 0.0
    return max(float(game.get("lives", 0)), 0.0) / 4


@vf.reward(weight=0.1)
async def format_reward(task: vf.Task, state: vf.State) -> float:
    _ = task
    completion = state.get("completion") or []
    assistant_messages = vf.get_messages(completion, role="assistant")
    if not assistant_messages:
        return 0.0
    scores: list[float] = []
    for message in assistant_messages:
        content = str(message.content or "")
        has_think = bool(re.search(r"<think>.*?</think>", content, re.DOTALL | re.IGNORECASE))
        has_guess = parse_guess(content) is not None
        scores.append((0.3 if has_think else 0.0) + (0.7 if has_guess else 0.0))
    return sum(scores) / len(scores)


class NYTConnectionsTasksetConfig(vf.TasksetConfig):
    dataset_url: str = NYT_CONNECTIONS_URL
    num_train_examples: int = 900
    num_eval_examples: int = 100
    seed: int = 42
    max_turns: int = 8
    system_prompt: str | None = NYT_CONNECTIONS_SYSTEM_PROMPT
    user: str | None = "connections_user"
    rewards: list[str] = [
        "success_reward",
        "progress_reward",
        "efficiency_reward",
        "format_reward",
    ]


class NYTConnectionsEnvConfig(vf.EnvConfig):
    taskset: NYTConnectionsTasksetConfig = NYTConnectionsTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


class NYTConnectionsTaskset(vf.Taskset[NYTConnectionsTasksetConfig]):
    def _tasks(self) -> tuple[list[vf.ConfigData], list[vf.ConfigData]]:
        return build_tasks(
            dataset_url=self.config.dataset_url,
            seed=self.config.seed,
            max_turns=self.config.max_turns,
            num_train_examples=self.config.num_train_examples,
            num_eval_examples=self.config.num_eval_examples,
        )

    def load_tasks(self) -> vf.Tasks:
        train_tasks, _ = self._tasks()
        return train_tasks

    def load_eval_tasks(self) -> vf.Tasks:
        _, eval_tasks = self._tasks()
        return eval_tasks


def load_taskset(config: NYTConnectionsTasksetConfig) -> NYTConnectionsTaskset:
    return NYTConnectionsTaskset(config=config)


def load_environment(config: NYTConnectionsEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=NYTConnectionsTaskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
