import re

import verifiers.v1 as vf
from tasksets.textarena import (
    TextArenaTask,
    TextArenaTaskset,
    TextArenaTasksetConfig,
)

WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    system_prompt: vf.SystemPrompt = WORDLE_SYSTEM_PROMPT


class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    guess_pattern = r"<guess>(.*?)</guess>"
    config: WordleTasksetConfig

    def guesses(self, content: str) -> list[str]:
        return re.findall(self.guess_pattern, content, re.DOTALL)

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: TextArenaTask, state: vf.State) -> float:
        answer = task.answer
        completion = state.completion
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
    async def length_bonus(self, task: TextArenaTask, state: vf.State) -> float:
        answer = task.answer
        completion = state.completion
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
    async def partial_answer(self, task: TextArenaTask, state: vf.State) -> float:
        answer = task.answer
        completion = state.completion
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
        completion = state.completion
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
    return WordleTaskset(config=config)
