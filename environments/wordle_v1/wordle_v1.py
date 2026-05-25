import re
from pathlib import Path

import verifiers as vf
from verifiers.v1.packages.tasksets.textarena import (
    TextArenaTaskset,
    TextArenaTasksetConfig,
)

WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


class WordleTasksetConfig(TextArenaTasksetConfig):
    tasks: str | None = "verifiers.v1.packages.tasksets.textarena:load_tasks"
    eval_tasks: str | None = "verifiers.v1.packages.tasksets.textarena:load_eval_tasks"
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    system_prompt: str | None = WORDLE_SYSTEM_PROMPT
    path_to_system_prompt: str = ""


class WordleTaskset(TextArenaTaskset):
    guess_pattern = r"<guess>(.*?)</guess>"
    config: WordleTasksetConfig

    def __init__(self, config: WordleTasksetConfig):
        assert isinstance(config, WordleTasksetConfig)
        super().__init__(config=config)

    def guesses(self, content: str) -> list[str]:
        return re.findall(self.guess_pattern, content, re.DOTALL)

    def format_observation(self, observation: str) -> str:
        latest_observation = observation.split("[GAME]")[-1].strip()
        if "Feedback:" in latest_observation:
            return latest_observation.split("Feedback:")[-1]
        return latest_observation

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = task["answer"]
        assert isinstance(answer, str)
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        for message in reversed(vf.get_messages(completion)):
            if not isinstance(message, vf.AssistantMessage):
                continue
            content = message.content
            assert isinstance(content, str)
            matches = self.guesses(content)
            if matches:
                return 1.0 if matches[-1].strip() == f"[{answer}]" else 0.0
        return 0.0

    @vf.reward(weight=1.0)
    async def length_bonus(self, task: vf.Task, state: vf.State) -> float:
        answer = task["answer"]
        assert isinstance(answer, str)
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        guess = ""
        num_guesses = 0
        for message in vf.get_messages(completion):
            if not isinstance(message, vf.AssistantMessage):
                continue
            content = message.content
            assert isinstance(content, str)
            if re.search(self.guess_pattern, content, re.DOTALL):
                num_guesses += 1
                matches = self.guesses(content)
                if matches:
                    guess = matches[-1].strip()
        is_correct = 1.0 if guess == f"[{answer}]" else 0.0
        assert num_guesses > 0 or is_correct == 0.0
        return is_correct / (num_guesses or 1)

    @vf.reward(weight=1.0)
    async def partial_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = task["answer"]
        assert isinstance(answer, str)
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        for message in reversed(vf.get_messages(completion)):
            if not isinstance(message, vf.AssistantMessage):
                continue
            content = message.content
            assert isinstance(content, str)
            matches = self.guesses(content)
            if matches:
                if matches[-1].strip() == f"[{answer}]":
                    return 0.0
                break
        for message in reversed(vf.get_messages(completion)):
            if not isinstance(message, vf.UserMessage):
                continue
            content = message.content
            assert isinstance(content, str)
            parts = content.strip().split("\n")
            if len(parts) == 3:
                scoring = parts[1].strip()
                return 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
        return 0.0

    @vf.reward(weight=0.2)
    async def format_reward(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        found = False
        for message in vf.get_messages(completion):
            if not isinstance(message, vf.AssistantMessage):
                continue
            found = True
            content = message.content
            assert isinstance(content, str)
            if len(self.guesses(content)) != 1:
                return 0.0
        return 1.0 if found else 0.0


def load_taskset(config: WordleTasksetConfig) -> WordleTaskset:
    assert isinstance(config, WordleTasksetConfig)
    if config.path_to_system_prompt:
        system_prompt = (
            Path(config.path_to_system_prompt).expanduser().read_text(encoding="utf-8")
        )
        if not system_prompt:
            raise ValueError("Wordle system prompt file must not be empty.")
        config = config.model_copy(update={"system_prompt": system_prompt})
    return WordleTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset_config = config.taskset
    assert isinstance(taskset_config, WordleTasksetConfig)
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=vf.Harness(config=config.harness),
    )
