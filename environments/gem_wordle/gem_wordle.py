import logging
from typing import Any

import verifiers as vf
from verifiers.envs.gym_env import GymEnv


def _patch_nltk_download_quiet() -> None:
    """
    GEM (or its dependencies) may call `nltk.download(...)` at import time and print
    noisy `[nltk_data] ...` logs. Force `quiet=True` if NLTK is installed.
    """
    try:
        import nltk
    except ImportError:
        return

    original_download = getattr(nltk, "_vf_original_download", None)
    if original_download is None:
        original_download = nltk.download
        setattr(nltk, "_vf_original_download", original_download)

    def download_quiet(*args: Any, **kwargs: Any) -> Any:
        return original_download(*args, **{**kwargs, "quiet": True})

    setattr(nltk, "download", download_quiet)


def _silence_openai_http_logs() -> None:
    # GEM (or its deps) may configure root logging to INFO, which makes OpenAI/httpx
    # per-request logs noisy during evals.
    for name in ("openai", "openai._base_client", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

### Prompt
# GEM relies on regex matching for \boxed{}, so we instruct the model accordingly.
GEM_WORDLE_SYSTEM_PROMPT = """You are a competitive Wordle player.
Your goal is to guess the secret 5-letter word within 20 turns.

In each turn:
1. Think step-by-step about the feedback (G=Green/Correct, Y=Yellow/Wrong Pos, X=Gray/Wrong).
2. Output your final guess inside \\boxed{YOUR_GUESS}.

Example:
<think>
I need to test vowels. "ADIEU" is a good start.
</think>
\\boxed{ADIEU}
"""

### Reward Functions
def gem_success_bonus(*, state: vf.State, **kwargs) -> float:
    """
    Reward for winning the game, based on GEM's success message in the final observation.

    The success message appears in the prompt of the last trajectory step,
    which contains the environment's response to the previous action.
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return 0.0

    # Check the last step's prompt for the success message
    # (the prompt contains the env observation from the previous turn)
    last_step = trajectory[-1]
    prompt = last_step.get("prompt", [])

    # Extract content from the last user message in the prompt
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "Congratulations!" in str(content):
                    return 1.0
                break
    elif isinstance(prompt, str) and "Congratulations!" in prompt:
        return 1.0

    return 0.0


def win_rate(*, state: vf.State, **kwargs) -> float:
    return gem_success_bonus(state=state, **kwargs)

### Environment Loader
def load_environment(
    num_train_episodes: int = 1000,
    num_eval_episodes: int = 20,
):
    """
    Load the GEM Wordle environment.

    Args:
        num_train_episodes: Number of training episodes (each with a random word).
        num_eval_episodes: Number of evaluation episodes.
    """
    from datasets import Dataset

    from verifiers.envs.gym_env import EpisodicSumRubric

    _patch_nltk_download_quiet()
    _silence_openai_http_logs()
    from gem.envs.game_env.wordle import WordleEnv

    # Rubric: sum of step rewards (dense) + success bonus (sparse)
    rubric = EpisodicSumRubric(weight=1.0)
    rubric.add_reward_func(win_rate, weight=0.0)

    # Build eval dataset with specified number of episodes
    eval_dataset = Dataset.from_dict({
        "example_id": list(range(num_eval_episodes)),
        "info": [{"seed": i} for i in range(num_eval_episodes)],
        "task": ["gem_wordle"] * num_eval_episodes,
        "prompt": [[]] * num_eval_episodes,
        "answer": [""] * num_eval_episodes,
    })

    vf_env = GymEnv(
        env_cls=WordleEnv,
        env_kwargs={
            "word_length": 5,
            "max_turns": 20,
            "only_real_words": True,
        },
        # GEM handles \boxed{} parsing internally and applies its own format/validity penalties
        # (e.g. missing boxed guess terminates with a small negative reward), so pass raw text.
        action_parser=lambda x: x,
        rubric=rubric,
        num_train_episodes=num_train_episodes,
        eval_dataset=eval_dataset,
        system_prompt=GEM_WORDLE_SYSTEM_PROMPT,
        message_type="chat",
        max_episode_steps=20,
    )

    return vf_env
