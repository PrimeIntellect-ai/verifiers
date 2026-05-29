import json
import re
import string
import urllib.request
from functools import lru_cache
from pathlib import Path

from datasets import Dataset

import verifiers as vf

DICTIONARY_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
DEFAULT_STARTING_WORDS = [
    "hat",
    "mine",
    "lung",
    "layer",
    "pattern",
    "camping",
    "avoid",
    "traveller",
    "origin",
    "abysmal",
]

# Small fallback used when the full dictionary cannot be downloaded in offline
# smoke tests. The online path uses dwyl/english-words, matching LisanBench.
FALLBACK_WORDS = {
    "a",
    "at",
    "ate",
    "bat",
    "bit",
    "bin",
    "cat",
    "cot",
    "cut",
    "hat",
    "hit",
    "hot",
    "hut",
    "mat",
    "mine",
    "mint",
    "mind",
    "minder",
    "line",
    "fine",
    "lung",
    "long",
    "log",
    "lag",
    "layer",
    "payer",
    "player",
    "slayer",
    "pattern",
    "patter",
    "batter",
    "better",
    "camping",
    "capping",
    "mapping",
    "avoid",
    "void",
    "voila",
    "traveller",
    "traveler",
    "travel",
    "origin",
    "origins",
    "abysmal",
    "abyssal",
    "sit",
    "wit",
    "win",
    "pin",
    "pit",
    "pat",
}

SYSTEM_PROMPT = """You are playing LisanBench, an open-ended word-chain benchmark.
Create the longest possible chain of valid English words.

Rules:
- Start with the given starting word.
- Each next word must have Levenshtein edit distance exactly 1 from the previous word.
- One step may add one letter, remove one letter, or substitute one letter.
- Every word must be a valid English word.
- Never repeat a word.
- Output only a comma-separated list of words, with no explanation."""


def create_prompt(starting_word: str) -> str:
    return f"""Starting word: {starting_word}

Return only the comma-separated word chain:
{starting_word}, next_word, next_word, ..."""


def build_dataset(starting_words: list[str] | None = None) -> Dataset:
    words = starting_words or DEFAULT_STARTING_WORDS
    return Dataset.from_list(
        [
            {
                "question": create_prompt(word),
                "answer": word,
                "info": {"starting_word": word},
            }
            for word in words
        ]
    )


def dictionary_cache_path() -> Path:
    return Path.home() / ".cache" / "verifiers" / "lisanbench" / "words_alpha.txt"


@lru_cache(maxsize=1)
def load_word_dictionary() -> set[str]:
    path = dictionary_cache_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DICTIONARY_URL, path)
        except Exception:
            return set(FALLBACK_WORDS)

    try:
        return {
            word.strip().lower()
            for word in path.read_text(encoding="utf-8").splitlines()
            if word.strip().isalpha()
        }
    except Exception:
        return set(FALLBACK_WORDS)


def edit_distance(word1: str, word2: str) -> int:
    word1 = word1.lower()
    word2 = word2.lower()
    if len(word1) == len(word2):
        return sum(c1 != c2 for c1, c2 in zip(word1, word2))

    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def is_valid_link(word1: str, word2: str) -> bool:
    return edit_distance(word1, word2) == 1


def extract_word_chain(completion: str) -> list[str]:
    # Accept comma lists, arrows, numbered lists, or newline lists while stripping
    # punctuation that often surrounds LLM answers.
    normalized = completion.lower().replace("→", ",").replace("->", ",")
    words: list[str] = []
    for line in normalized.splitlines():
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        line = line.translate(str.maketrans({ch: " " for ch in string.punctuation if ch not in {",", "'"}}))
        for token in re.split(r"[,\s]+", line):
            token = token.strip(" '").lower()
            if token.isalpha():
                words.append(token)
    return words


def validate_chain(word_chain: list[str], valid_words: set[str]) -> dict[str, int | float | bool]:
    if not word_chain:
        return {
            "valid_prefix_length": 0,
            "valid_links": 0,
            "invalid_links": 0,
            "duplicate_count": 0,
            "all_words_valid": False,
            "starts_correctly": False,
        }

    seen: set[str] = set()
    valid_prefix_length = 0
    valid_links = 0
    invalid_links = 0
    duplicate_count = 0

    for idx, word in enumerate(word_chain):
        if word in seen:
            duplicate_count += 1
            break
        seen.add(word)
        if word not in valid_words:
            break
        if idx == 0:
            valid_prefix_length = 1
            continue
        previous = word_chain[idx - 1]
        if is_valid_link(previous, word):
            valid_prefix_length = idx + 1
        else:
            break

    seen_links: set[str] = set()
    for word1, word2 in zip(word_chain, word_chain[1:]):
        duplicate = word1 in seen_links or word2 in seen_links
        if duplicate:
            invalid_links += 1
        elif word1 in valid_words and word2 in valid_words and is_valid_link(word1, word2):
            valid_links += 1
        else:
            invalid_links += 1
        seen_links.add(word1)

    return {
        "valid_prefix_length": valid_prefix_length,
        "valid_links": valid_links,
        "invalid_links": invalid_links,
        "duplicate_count": duplicate_count,
        "all_words_valid": all(word in valid_words for word in word_chain),
        "starts_correctly": bool(word_chain),
    }


def starting_word_reward(completion: str, answer: str, **kwargs) -> float:
    _ = kwargs
    chain = extract_word_chain(completion)
    return 1.0 if chain and chain[0] == answer.lower() else 0.0


def valid_chain_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    chain = extract_word_chain(completion)
    stats = validate_chain(chain, load_word_dictionary())
    total = stats["valid_links"] + stats["invalid_links"]
    if total == 0:
        return 0.0
    return stats["valid_links"] / total


def length_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    chain = extract_word_chain(completion)
    prefix_length = validate_chain(chain, load_word_dictionary())["valid_prefix_length"]
    # Reward longer valid prefixes, saturating at 25 words to keep reward bounded.
    return min(float(prefix_length) / 25.0, 1.0)


def no_duplicate_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    chain = extract_word_chain(completion)
    return 1.0 if len(chain) == len(set(chain)) and chain else 0.0


def format_reward(completion: str, answer: str, **kwargs) -> float:
    _ = answer, kwargs
    chain = extract_word_chain(completion)
    if len(chain) < 2:
        return 0.0
    has_explanation_markers = bool(re.search(r"\b(reason|because|explanation|steps?)\b", completion, re.I))
    comma_like = "," in completion or "->" in completion or "→" in completion
    return 1.0 if comma_like and not has_explanation_markers else 0.5 if comma_like else 0.0


def load_environment(starting_words: list[str] | None = None) -> vf.Environment:
    rubric = vf.Rubric(
        funcs=[
            starting_word_reward,
            valid_chain_reward,
            length_reward,
            no_duplicate_reward,
            format_reward,
        ],
        weights=[1.0, 2.0, 2.0, 1.0, 0.5],
    )
    return vf.SingleTurnEnv(
        dataset=build_dataset(starting_words),
        system_prompt=SYSTEM_PROMPT,
        parser=vf.Parser(),
        rubric=rubric,
    )


def main() -> None:
    _ = load_environment()
    print(json.dumps({"environment": "lisanbench", "num_tasks": len(build_dataset())}))


if __name__ == "__main__":
    main()
