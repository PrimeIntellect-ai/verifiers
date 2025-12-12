"""
Synthetic data generation for verbatim copy task.

Generates different types of text with varying difficulty levels:
- Structured data (JSON-like, CSV-like) using faker
- Random word sequences
- Alphanumeric codes using UUIDs
- Mixed content combining multiple types
"""

import json
import random
import uuid
from typing import Literal

from faker import Faker

# Difficulty levels
DifficultyLevel = Literal["easy", "medium", "hard", "mixed"]


def generate_structured_data(
    fake: Faker,
    num_records: int = 3,
    seed: int | None = None,
) -> str:
    """
    Generate structured data (JSON-like records) using faker.

    Args:
        fake: Faker instance to use
        num_records: Number of records to generate
        seed: Random seed for reproducibility

    Returns:
        JSON string with fake records
    """
    if seed is not None:
        fake.seed_instance(seed)

    records = []
    for _ in range(num_records):
        record = {
            "id": fake.random_int(min=10000, max=99999),
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.street_address(),
            "city": fake.city(),
            "country": fake.country(),
        }
        records.append(record)

    return json.dumps(records, indent=2)


def generate_word_sequence(
    num_words: int = 30,
    seed: int | None = None,
) -> str:
    """
    Generate a sequence of random common English words.

    Args:
        num_words: Number of words to generate
        seed: Random seed for reproducibility

    Returns:
        Space-separated word sequence
    """
    # Common English words that are unambiguous and varied
    word_list = [
        "telescope",
        "umbrella",
        "fourteen",
        "marble",
        "quantum",
        "village",
        "keyboard",
        "elephant",
        "mountain",
        "journal",
        "cabinet",
        "whisper",
        "library",
        "diamond",
        "blanket",
        "thunder",
        "penguin",
        "harvest",
        "factory",
        "dolphin",
        "chapter",
        "balloon",
        "mystery",
        "kitchen",
        "science",
        "chamber",
        "lantern",
        "century",
        "granite",
        "weather",
        "platform",
        "calendar",
        "triangle",
        "spectrum",
        "hospital",
        "argument",
        "criminal",
        "daughter",
        "evidence",
        "familiar",
        "generous",
        "handbook",
        "innocent",
        "judicial",
        "kilogram",
        "landmark",
        "magnetic",
        "national",
        "obituary",
        "parallel",
        "quantity",
        "rational",
        "sandwich",
        "tangible",
        "umbrella",
        "valuable",
        "warranty",
        "yearbook",
        "absolute",
        "boundary",
    ]

    if seed is not None:
        random.seed(seed)

    selected = random.choices(word_list, k=num_words)
    return " ".join(selected)


def generate_alphanumeric_codes(
    num_codes: int = 5,
    code_format: Literal["uuid", "short", "mixed"] = "mixed",
    seed: int | None = None,
) -> str:
    """
    Generate alphanumeric codes (UUIDs, short codes, etc.).

    Args:
        num_codes: Number of codes to generate
        code_format: Type of codes to generate
        seed: Random seed for reproducibility

    Returns:
        Newline-separated codes
    """
    if seed is not None:
        random.seed(seed)

    codes = []
    for i in range(num_codes):
        if code_format == "uuid":
            # Full UUID
            code = str(uuid.UUID(int=random.getrandbits(128)))
        elif code_format == "short":
            # Short alphanumeric code like A7X-K9M2-QP4L
            chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # No confusables
            segments = []
            for _ in range(3):
                segment = "".join(random.choices(chars, k=4))
                segments.append(segment)
            code = "-".join(segments)
        else:  # mixed
            if i % 2 == 0:
                code = str(uuid.UUID(int=random.getrandbits(128)))
            else:
                chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
                segments = ["".join(random.choices(chars, k=4)) for _ in range(3)]
                code = "-".join(segments)
        codes.append(code)

    return "\n".join(codes)


def generate_csv_data(
    fake: Faker,
    num_rows: int = 5,
    seed: int | None = None,
) -> str:
    """
    Generate CSV-formatted data.

    Args:
        fake: Faker instance to use
        num_rows: Number of data rows to generate
        seed: Random seed for reproducibility

    Returns:
        CSV string with header and data rows
    """
    if seed is not None:
        fake.seed_instance(seed)

    lines = ["id,product,price,quantity,date"]
    for _ in range(num_rows):
        row = [
            str(fake.random_int(min=1000, max=9999)),
            fake.word().capitalize() + " " + fake.word().capitalize(),
            f"{fake.random_int(min=10, max=999)}.{fake.random_int(min=0, max=99):02d}",
            str(fake.random_int(min=1, max=100)),
            fake.date(),
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def generate_sample(
    difficulty: DifficultyLevel = "medium",
    length_scale: float = 1.0,
    seed: int | None = None,
) -> dict:
    """
    Generate a single sample for the verbatim copy task.

    Args:
        difficulty: Difficulty level of the text to copy
        length_scale: Multiplier for output length (1.0 = default, 2.0 = double, etc.)
                      Allows arbitrary scaling for future-proofing as models improve.
        seed: Random seed for reproducibility

    Returns:
        Dict with 'text' (the text to copy), 'difficulty', and 'length_scale' metadata
    """
    # Base values (what length_scale=1.0 produces)
    BASE_WORDS = 25
    BASE_RECORDS = 2
    BASE_ROWS = 4
    BASE_CODES = 8
    # Mixed mode uses smaller base values
    BASE_MIXED_CODES = 3
    BASE_MIXED_WORDS = 10
    BASE_MIXED_ROWS = 3

    # Scale with minimum bounds to avoid degenerate cases
    num_words = max(5, int(BASE_WORDS * length_scale))
    num_records = max(1, int(BASE_RECORDS * length_scale))
    num_rows = max(1, int(BASE_ROWS * length_scale))
    num_codes = max(2, int(BASE_CODES * length_scale))
    mixed_codes = max(1, int(BASE_MIXED_CODES * length_scale))
    mixed_words = max(3, int(BASE_MIXED_WORDS * length_scale))
    mixed_rows = max(1, int(BASE_MIXED_ROWS * length_scale))

    fake = Faker()

    if seed is not None:
        random.seed(seed)
        fake.seed_instance(seed)

    if difficulty == "easy":
        # Word sequences - familiar patterns
        text = generate_word_sequence(num_words=num_words, seed=seed)
    elif difficulty == "medium":
        # Structured data - numbers and special chars
        choice = random.choice(["json", "csv"])
        if choice == "json":
            text = generate_structured_data(fake, num_records=num_records, seed=seed)
        else:
            text = generate_csv_data(fake, num_rows=num_rows, seed=seed)
    elif difficulty == "hard":
        # Alphanumeric codes - no semantic cues
        text = generate_alphanumeric_codes(
            num_codes=num_codes, code_format="mixed", seed=seed
        )
    else:  # mixed
        # Combine multiple types
        parts = [
            f"Reference codes:\n{generate_alphanumeric_codes(num_codes=mixed_codes, code_format='short', seed=seed)}",
            f"\nKeywords: {generate_word_sequence(num_words=mixed_words, seed=seed + 1 if seed else None)}",
            f"\nData:\n{generate_csv_data(fake, num_rows=mixed_rows, seed=seed + 2 if seed else None)}",
        ]
        text = "\n".join(parts)

    return {
        "text": text,
        "difficulty": difficulty,
        "length_scale": length_scale,
    }


def generate_dataset(
    num_samples: int = 100,
    difficulty_distribution: dict[DifficultyLevel, float] | None = None,
    length_scale: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a dataset of verbatim copy samples.

    Args:
        num_samples: Total number of samples to generate
        difficulty_distribution: Dict mapping difficulty to proportion (must sum to 1.0)
                                 Default: {"easy": 0.25, "medium": 0.35, "hard": 0.25, "mixed": 0.15}
        length_scale: Multiplier for output length (1.0 = default, 2.0 = double, etc.)
                      Allows arbitrary scaling for future-proofing as models improve.
        seed: Base random seed for reproducibility

    Returns:
        List of sample dicts with 'text', 'difficulty', and 'length_scale' keys
    """
    if difficulty_distribution is None:
        difficulty_distribution = {
            "easy": 0.25,
            "medium": 0.35,
            "hard": 0.25,
            "mixed": 0.15,
        }

    random.seed(seed)

    # Calculate counts for each difficulty
    difficulties: list[DifficultyLevel] = []
    for diff, proportion in difficulty_distribution.items():
        count = int(num_samples * proportion)
        difficulties.extend([diff] * count)

    # Fill remaining slots randomly
    while len(difficulties) < num_samples:
        difficulties.append(random.choice(list(difficulty_distribution.keys())))

    random.shuffle(difficulties)

    # Generate samples
    samples = []
    for i, difficulty in enumerate(difficulties):
        sample_seed = seed + i * 1000  # Ensure different seeds per sample
        sample = generate_sample(
            difficulty=difficulty, length_scale=length_scale, seed=sample_seed
        )
        sample["id"] = i
        samples.append(sample)

    return samples
