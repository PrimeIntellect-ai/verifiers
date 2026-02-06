"""Reward provider utilities for TextArena environments."""

from __future__ import annotations

import re
from typing import Dict, List, Protocol, Tuple

try:
    from textarena_env.models import TextArenaAction, TextArenaObservation
except ImportError:
    from models import TextArenaAction, TextArenaObservation


class RewardProvider(Protocol):
    """Interface for computing auxiliary reward signals."""

    def reset(self) -> None:
        """Clear any internal state before a new episode."""

    def compute(
        self, *, action: TextArenaAction, observation: TextArenaObservation
    ) -> Dict[str, float]:
        """Return a mapping of reward names to float values for the step."""


def build_reward_providers(env_id: str) -> List[RewardProvider]:
    """Instantiate reward providers appropriate for the given environment."""

    providers: List[RewardProvider] = []
    if env_id == "Wordle-v0":
        providers.append(_WordleRewardProvider())
    return providers


_WORDLE_GUESS_PATTERN = re.compile(r"\[[A-Za-z]{5}\]")


def extract_guess(text: str) -> str:
    """Normalize a Wordle guess string from arbitrary text."""

    match = _WORDLE_GUESS_PATTERN.search(text)
    if match:
        return match.group(0).lower()

    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[dunno]"


def extract_wordle_feedback(observation: TextArenaObservation) -> str:
    """Pull the latest feedback text from a Wordle observation."""

    for message in reversed(observation.messages):
        content = message.content.strip()
        if "Feedback:" in content:
            return content.split("Feedback:", 1)[-1].strip()
    return ""


def extract_feedback_counts(feedback: str) -> Tuple[int, int]:
    """Return counts of green (G) and yellow (Y) markers from feedback."""

    if not feedback:
        return (0, 0)

    lines = [line.strip() for line in feedback.split("\n") if line.strip()]
    if len(lines) < 2:
        return (0, 0)

    for line in reversed(lines):
        normalized = line.replace(" ", "")
        if normalized and all(c in "GYX" for c in normalized):
            green = normalized.count("G")
            yellow = normalized.count("Y")
            return (green, yellow)

    return (0, 0)


class _WordleRewardProvider:
    """Reward provider that mirrors the GRPO Wordle heuristics."""

    SIGNAL_MAP = {
        "greens": "wordle.greens",
        "yellows": "wordle.yellows",
        "repetitions": "wordle.repetitions",
        "correct": "wordle.correct",
    }

    def __init__(self) -> None:
        self._guess_history: Dict[str, int] = {}

    def reset(self) -> None:
        self._guess_history.clear()

    def compute(
        self, *, action: TextArenaAction, observation: TextArenaObservation
    ) -> Dict[str, float]:
        guess = extract_guess(action.message)
        feedback = extract_wordle_feedback(observation)

        normalized_guess = guess if guess and guess != "[dunno]" else ""
        previous_occurrences = (
            self._guess_history.get(normalized_guess, 0) if normalized_guess else 0
        )

        green_score = 0.0
        yellow_score = 0.0
        if feedback:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0

        repetition_score = 1.0 - previous_occurrences
        correct_score = float(observation.reward or 0.0)

        if normalized_guess:
            self._guess_history[normalized_guess] = previous_occurrences + 1

        return {
            self.SIGNAL_MAP["greens"]: float(green_score),
            self.SIGNAL_MAP["yellows"]: float(yellow_score),
            self.SIGNAL_MAP["repetitions"]: float(repetition_score),
            self.SIGNAL_MAP["correct"]: float(correct_score),
        }


__all__ = [
    "RewardProvider",
    "build_reward_providers",
    "extract_feedback_counts",
    "extract_guess",
    "extract_wordle_feedback",
]
