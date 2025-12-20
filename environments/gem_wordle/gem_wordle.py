import logging

import nltk  # type: ignore

_original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: _original_nltk_download(  # type: ignore[method-assign]
    *args, **{**kwargs, "quiet": True}
)

# prevent GEM from hijacking logging levels
_root_level = logging.root.level
from gem.envs.game_env.wordle import WordleEnv  # noqa: E402

logging.root.setLevel(_root_level)

import verifiers as vf  # noqa: E402
from verifiers.envs.experimental.gym_env import EpisodicSumRubric  # noqa: E402

GEM_WORDLE_SYSTEM_PROMPT = """You are a competitive Wordle player.
Your goal is to guess the secret 5-letter word within 20 turns.

In each turn:
1. Think step-by-step about the feedback
   (G=Green/Correct, Y=Yellow/Wrong Pos, X=Gray/Wrong).
2. Output your final guess inside \\boxed{YOUR_GUESS}.
"""


def load_environment(
    num_train_episodes: int = 1000,
    num_eval_episodes: int = 20,
    word_length: int = 5,
    max_turns: int = 20,
    only_real_words: bool = True,
):
    def win_rate(state: vf.State) -> float:
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return 0.0
        prompt = trajectory[-1].get("prompt", "")
        content = (
            prompt[-1].get("content", "")
            if isinstance(prompt, list) and prompt
            else prompt
        )
        return 1.0 if "Congratulations!" in str(content) else 0.0

    rubric = EpisodicSumRubric(weight=1.0)
    rubric.add_reward_func(win_rate, weight=0.0)

    return vf.GymEnv(
        env_cls=WordleEnv,
        env_kwargs={
            "word_length": word_length,
            "max_turns": max_turns,
            "only_real_words": only_real_words,
        },
        rubric=rubric,
        num_train_episodes=num_train_episodes,
        num_eval_episodes=num_eval_episodes,
        system_prompt=GEM_WORDLE_SYSTEM_PROMPT,
        max_episode_steps=20,
    )
