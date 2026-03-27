"""
Context Dropping RLM Environment.

Tests a model's ability to manage its own context window by reading facts
from many files, dropping old turns when context fills up, and using
summaries to retain key information.

Each sample has N files in the context directory, each containing a
key-value fact padded with filler text. The question asks for a subset
of those facts. The total content is sized to exceed the context window,
forcing the model to use `remove_conversation_turns` to keep working.

Rewards:
- Partial credit for each fact found.
- Exact match bonus when all facts are found.
"""

import random
import string
from typing import Literal

import verifiers as vf
from datasets import Dataset
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.utils.data_utils import extract_boxed_answer


# =============================================================================
# Fact Generation
# =============================================================================

# Domains and templates for generating diverse key-value facts
_FACT_TEMPLATES = [
    ("The capital of {key} is {value}.",),
    ("The inventor of {key} is {value}.",),
    ("The color of {key} is {value}.",),
    ("The password for {key} is {value}.",),
    ("The price of {key} is {value}.",),
]

_KEY_POOLS = {
    "capital": [
        "Zarbia",
        "Plondia",
        "Querthos",
        "Vexmoor",
        "Brindal",
        "Omrath",
        "Telvian",
        "Korinth",
        "Yulstead",
        "Grenmarch",
        "Pellura",
        "Daxholm",
        "Ithwane",
        "Norbliss",
        "Caltreen",
        "Umbrix",
        "Faldren",
        "Jothica",
        "Wenphor",
        "Selvoria",
    ],
    "value": [
        "Mottenheim",
        "Clarifax",
        "Durnwall",
        "Skelvoss",
        "Plimmoth",
        "Trevayne",
        "Galdric",
        "Hestport",
        "Kinvara",
        "Lorcastle",
        "Ashforth",
        "Belmire",
        "Corvidal",
        "Dunwick",
        "Elstow",
        "Fenhollow",
        "Grimstead",
        "Hartmoor",
        "Ivyreach",
        "Junecliff",
    ],
}


def _generate_filler(num_words: int) -> str:
    """Generate plausible-looking filler text."""
    words = []
    for _ in range(num_words):
        length = random.randint(3, 10)
        words.append("".join(random.choices(string.ascii_lowercase, k=length)))
    # Break into sentences of 8-15 words
    sentences = []
    i = 0
    while i < len(words):
        sent_len = random.randint(8, 15)
        sent_words = words[i : i + sent_len]
        if sent_words:
            sent_words[0] = sent_words[0].capitalize()
            sentences.append(" ".join(sent_words) + ".")
        i += sent_len
    return " ".join(sentences)


def _generate_fact_files(
    num_files: int,
    num_target_facts: int,
    filler_words_per_file: int,
    seed: int,
) -> tuple[dict[str, str], list[tuple[str, str]], list[tuple[str, str]]]:
    """Generate files with facts embedded in filler text.

    Returns:
        (file_map, target_facts, all_facts)
        - file_map: {filename: content} for the context directory
        - target_facts: the facts the question will ask about
        - all_facts: all facts across all files
    """
    rng = random.Random(seed)

    keys = list(_KEY_POOLS["capital"])
    values = list(_KEY_POOLS["value"])
    rng.shuffle(keys)
    rng.shuffle(values)

    all_facts: list[tuple[str, str]] = []
    file_map: dict[str, str] = {}

    for i in range(num_files):
        key = keys[i % len(keys)]
        value = values[i % len(values)]
        template = rng.choice(_FACT_TEMPLATES)[0]
        fact_line = template.format(key=key, value=value)
        all_facts.append((key, value))

        # Build file content: filler before, fact, filler after
        before = _generate_filler(filler_words_per_file // 2)
        after = _generate_filler(filler_words_per_file // 2)
        content = f"{before}\n\n{fact_line}\n\n{after}"
        file_map[f"file_{i:03d}.txt"] = content

    # Select target facts (the ones the question asks about)
    target_indices = rng.sample(range(num_files), min(num_target_facts, num_files))
    target_facts = [all_facts[i] for i in target_indices]

    return file_map, target_facts, all_facts


# =============================================================================
# Environment
# =============================================================================


def load_environment(
    # Dataset options
    num_samples: int = 5,
    num_files: int = 15,
    num_target_facts: int = 5,
    filler_words_per_file: int = 2000,
    seed: int = 42,
    # RLM options
    max_turns: int = 30,
    max_output_length: int = 4096,
    code_execution_timeout: int = 120,
    repl_language: Literal["bash", "python"] = "python",
    root_prompt_verbosity: Literal["light", "medium", "heavy"] = "heavy",
    # Context dropping options
    allow_context_dropping: bool = True,
    min_turns_in_context: int = 3,
    expose_message_history: bool = True,
    # Sandbox options
    sandbox_docker_image: str = "python:3.11-slim",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_disk_size_gb: int = 5,
    sandbox_timeout_minutes: int = 60,
    **kwargs,
) -> vf.Environment:
    """
    Load the context dropping test environment.

    The model receives N files, each containing a fact buried in filler text.
    The question asks for a subset of those facts. The total content is large
    enough that the model must drop old context to keep working.

    Args:
        num_samples: Number of dataset samples.
        num_files: Number of files per sample.
        num_target_facts: Number of facts the question asks about.
        filler_words_per_file: Amount of filler text per file.
        seed: Random seed for reproducibility.
        max_turns: Maximum REPL turns.
        max_output_length: Max output length per REPL call.
        code_execution_timeout: Timeout for code execution.
        repl_language: REPL language ("python" or "bash").
        root_prompt_verbosity: Verbosity of root system prompt.
        allow_context_dropping: Enable context dropping tool.
        min_turns_in_context: Minimum turns to keep in context.
        expose_message_history: Write .messages file to sandbox.
        sandbox_docker_image: Docker image for sandbox.
        sandbox_cpu_cores: CPU cores for sandbox.
        sandbox_memory_gb: Memory in GB.
        sandbox_disk_size_gb: Disk in GB.
        sandbox_timeout_minutes: Sandbox lifetime.
        **kwargs: Additional arguments passed to RLMEnv.
    """

    def build_dataset():
        rows = []
        for i in range(num_samples):
            sample_seed = seed + i
            file_map, target_facts, _all_facts = _generate_fact_files(
                num_files=num_files,
                num_target_facts=num_target_facts,
                filler_words_per_file=filler_words_per_file,
                seed=sample_seed,
            )

            # Build the question
            target_keys = [k for k, _v in target_facts]
            target_lines = "\n".join(f"- {k}" for k in target_keys)
            question = (
                f"There are {num_files} text files in the working directory "
                f"(file_000.txt through file_{num_files - 1:03d}.txt). "
                f"Each file contains one hidden fact buried in filler text.\n\n"
                f"Find the values for these keys:\n{target_lines}\n\n"
                f"IMPORTANT INSTRUCTIONS:\n"
                f"- You MUST read each file individually using cat or open(), "
                f"printing its FULL content, to verify you haven't missed anything.\n"
                f"- Do NOT use grep, regex, or any shortcut to extract facts. "
                f"Read each file completely.\n"
                f"- After reading a few files, your context will fill up. "
                f"Use `remove_conversation_turns` to drop old turns and free space. "
                f"Write a good summary of what you found so far before dropping.\n"
                f"- After dropping context, consult .summaries to recall previous findings.\n\n"
                f"Return your answer as a comma-separated list of "
                f"'key=value' pairs."
            )

            # Answer string
            answer = ", ".join(f"{k}={v}" for k, v in target_facts)

            rows.append(
                {
                    "prompt": [{"role": "user", "content": question}],
                    "answer": answer,
                    "info": {"context": file_map},
                }
            )

        return Dataset.from_list(rows)

    # Reward functions
    def partial_match_reward(state: vf.State, completion, **_kwargs) -> float:
        """Fraction of target key=value pairs found."""
        final_answer = state.get("final_answer", "")
        if not final_answer:
            # Fallback: check last assistant message in completion
            for msg in reversed(completion):
                content = (
                    msg.get("content", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", "")
                )
                if content and isinstance(content, str):
                    final_answer = content
                    break
        boxed = extract_boxed_answer(final_answer)
        response = boxed if boxed != final_answer else final_answer

        expected = state.get("answer", "")
        expected_pairs = {p.strip().lower() for p in expected.split(",") if "=" in p}
        found_pairs = {p.strip().lower() for p in response.split(",") if "=" in p}
        if not expected_pairs:
            return 0.0
        return len(expected_pairs & found_pairs) / len(expected_pairs)

    def exact_match_reward(state: vf.State, completion, **_kwargs) -> float:
        """1.0 only if all target pairs are found."""
        return 1.0 if partial_match_reward(state, completion) == 1.0 else 0.0

    rubric = vf.Rubric(
        funcs=[partial_match_reward, exact_match_reward],
        weights=[1.0, 0.0],
    )

    sandbox_labels = kwargs.pop("sandbox_labels", ["context-dropping-rlm"])

    return RLMEnv(
        dataset=build_dataset,
        rubric=rubric,
        max_turns=max_turns,
        max_output_length=max_output_length,
        code_execution_timeout=code_execution_timeout,
        repl_language=repl_language,
        root_prompt_verbosity=root_prompt_verbosity,
        allow_context_dropping=allow_context_dropping,
        min_turns_in_context=min_turns_in_context,
        expose_message_history=expose_message_history,
        sandbox_docker_image=sandbox_docker_image,
        sandbox_cpu_cores=sandbox_cpu_cores,
        sandbox_memory_gb=sandbox_memory_gb,
        sandbox_disk_size_gb=sandbox_disk_size_gb,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        sandbox_labels=sandbox_labels,
        **kwargs,
    )
