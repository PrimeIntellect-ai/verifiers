import re

import verifiers as vf
from tasksets.textarena import (
    TextArenaTaskset,
    TextArenaTasksetConfig,
    TextArenaUser,
    TextArenaUserConfig,
)

WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


class WordleUserConfig(TextArenaUserConfig):
    pass


class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    user: WordleUserConfig | None = WordleUserConfig()
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = WORDLE_SYSTEM_PROMPT


class WordleUser(TextArenaUser):
    config: WordleUserConfig

    async def get_response(
        self, task: vf.Task, state: vf.State, messages: list[vf.Message]
    ) -> list[vf.UserMessage]:
        response = await super().get_response(task, state, messages)
        if state.get("done") is True:
            return response
        assert len(response) == 1
        content = response[0].content
        assert isinstance(content, str)
        latest_feedback = content.split("[GAME]")[-1].strip()
        if "Feedback:" in latest_feedback:
            latest_feedback = latest_feedback.split("Feedback:")[-1]
        return [vf.UserMessage(content=latest_feedback)]


class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    guess_pattern = r"<guess>(.*?)</guess>"
    config: WordleTasksetConfig

    def guesses(self, content: str) -> list[str]:
        return re.findall(self.guess_pattern, content, re.DOTALL)

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
    return WordleTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset_config = config.taskset
    harness_config = config.harness
    assert isinstance(taskset_config, WordleTasksetConfig)
    assert isinstance(harness_config, vf.HarnessConfig)
    return vf.Env(
        taskset=load_taskset(taskset_config),
        harness=vf.Harness(config=harness_config),
    )
