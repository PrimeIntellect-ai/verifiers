"""
Synthetic data generation for verbatim copy task.

Generates different types of text content:
- JSON formatted data using faker
- CSV tabular data using faker
- Random word sequences
- Alphanumeric codes using UUIDs
- Mixed content combining multiple types

Supports optional fragmentation to create tokenization-challenging sequences
by slicing and concatenating content from multiple sources.
"""

import json
import logging
import random
import uuid
from typing import Literal

from faker import Faker

logger = logging.getLogger(__name__)

# Content types (what kind of content is generated)
ContentType = Literal["words", "json", "csv", "codes", "mixed"]

# Default target lengths (in characters) for each content type
DEFAULT_TARGET_LENGTHS: dict[ContentType, int] = {
    "words": 200,
    "json": 500,
    "csv": 500,
    "codes": 300,
    "mixed": 600,
}


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


def _generate_raw_content(
    content_type: ContentType,
    target_length: int,
    seed: int | None,
    fake: Faker,
) -> str:
    """
    Generate raw content of at least target_length characters.

    Over-produces content to ensure we have enough material for slicing.
    """
    # Over-produce by 2x to ensure enough material
    overproduce_factor = 2
    needed_length = target_length * overproduce_factor

    content_parts: list[str] = []
    current_length = 0
    iteration = 0

    while current_length < needed_length:
        iter_seed = seed + iteration * 100 if seed is not None else None

        if content_type == "words":
            # Word sequences - familiar patterns
            chunk = generate_word_sequence(num_words=50, seed=iter_seed)
        elif content_type == "json":
            # JSON structured data
            chunk = generate_structured_data(fake, num_records=3, seed=iter_seed)
        elif content_type == "csv":
            # CSV tabular data
            chunk = generate_csv_data(fake, num_rows=6, seed=iter_seed)
        elif content_type == "codes":
            # Alphanumeric codes - UUIDs and short codes
            chunk = generate_alphanumeric_codes(
                num_codes=10, code_format="mixed", seed=iter_seed
            )
        else:  # mixed
            # Rotate through different types
            type_choice = iteration % 4
            if type_choice == 0:
                chunk = generate_alphanumeric_codes(
                    num_codes=5, code_format="short", seed=iter_seed
                )
            elif type_choice == 1:
                chunk = generate_word_sequence(num_words=20, seed=iter_seed)
            elif type_choice == 2:
                chunk = generate_structured_data(fake, num_records=2, seed=iter_seed)
            else:
                chunk = generate_csv_data(fake, num_rows=4, seed=iter_seed)

        content_parts.append(chunk)
        current_length += len(chunk)
        iteration += 1

    return "\n".join(content_parts)


def _apply_fragmentation(
    raw_content: str,
    target_length: int,
    mean_fragment_length: int,
    seed: int | None,
) -> str:
    """
    Apply fragmentation by taking random slices and concatenating them.

    This creates tokenization-challenging sequences by breaking natural
    token boundaries.
    """
    if seed is not None:
        random.seed(seed)

    result_parts: list[str] = []
    current_length = 0
    content_len = len(raw_content)

    while current_length < target_length:
        remaining = target_length - current_length

        # Vary fragment size: uniform in [0.5 * mean, 1.5 * mean], clamped to remaining
        min_frag = max(1, int(mean_fragment_length * 0.5))
        max_frag = min(int(mean_fragment_length * 1.5), remaining)

        # Ensure valid range (min_frag <= max_frag)
        min_frag = min(min_frag, max_frag)
        fragment_size = random.randint(min_frag, max_frag)

        # Pick a random start position in the raw content
        max_start = max(0, content_len - fragment_size)
        start_pos = random.randint(0, max_start) if max_start > 0 else 0

        # Extract the fragment
        fragment = raw_content[start_pos : start_pos + fragment_size]
        result_parts.append(fragment)
        current_length += len(fragment)

    return "".join(result_parts)


def generate_sample(
    content_type: ContentType = "json",
    target_length: int | None = None,
    mean_fragment_length: int | None = None,
    seed: int | None = None,
) -> dict:
    """
    Generate a single sample for the verbatim copy task.

    Args:
        content_type: Type of content to generate:
                      - "words": English word sequences
                      - "json": JSON formatted data
                      - "csv": CSV tabular data
                      - "codes": UUIDs and alphanumeric codes
                      - "mixed": combination of all types
        target_length: Target length in characters. If None, uses default for content type.
        mean_fragment_length: If set, enables fragmentation - content is sliced into
                              fragments of approximately this size (with random variation)
                              and concatenated. This creates tokenization-challenging
                              sequences. If None, no fragmentation is applied.
        seed: Random seed for reproducibility

    Returns:
        Dict with 'text', 'content_type', 'target_length', and 'mean_fragment_length'
    """
    # Resolve target_length
    if target_length is None:
        target_length = DEFAULT_TARGET_LENGTHS[content_type]

    # Validate mean_fragment_length
    if mean_fragment_length is not None:
        if mean_fragment_length <= 0:
            raise ValueError("mean_fragment_length must be positive")
        if mean_fragment_length > target_length:
            logger.warning(
                f"mean_fragment_length ({mean_fragment_length}) > target_length "
                f"({target_length}). Disabling fragmentation."
            )
            mean_fragment_length = None

    fake = Faker()
    if seed is not None:
        random.seed(seed)
        fake.seed_instance(seed)

    # Generate raw content (over-produced)
    raw_content = _generate_raw_content(content_type, target_length, seed, fake)

    # Apply fragmentation or simple truncation
    if mean_fragment_length is not None:
        text = _apply_fragmentation(
            raw_content, target_length, mean_fragment_length, seed
        )
    else:
        # No fragmentation: just truncate to target length
        text = raw_content[:target_length]

    return {
        "text": text,
        "content_type": content_type,
        "target_length": target_length,
        "mean_fragment_length": mean_fragment_length,
    }


def generate_dataset(
    num_samples: int = 100,
    content_type: ContentType | Literal["all"] = "all",
    target_length: int | None = None,
    mean_fragment_length: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate a dataset of verbatim copy samples.

    Args:
        num_samples: Total number of samples to generate
        content_type: Type of content for samples:
                      - "words": English word sequences
                      - "json": JSON formatted data
                      - "csv": CSV tabular data
                      - "codes": UUIDs and alphanumeric codes
                      - "mixed": combination of all types
                      - "all": balanced mix across all types
        target_length: Target length in characters. If None, uses default per content type.
        mean_fragment_length: If set, enables fragmentation for tokenization-challenging
                              sequences. If None, no fragmentation is applied.
        seed: Random seed for reproducibility. If None, uses system randomness.

    Returns:
        List of sample dicts with 'text', 'content_type', 'target_length', etc.
    """
    random.seed(seed)

    # Build list of content types for each sample
    if content_type == "all":
        # Balanced distribution across all content types
        distribution: dict[ContentType, float] = {
            "words": 0.20,
            "json": 0.20,
            "csv": 0.20,
            "codes": 0.25,
            "mixed": 0.15,
        }
        content_types: list[ContentType] = []
        for ct, proportion in distribution.items():
            count = int(num_samples * proportion)
            content_types.extend([ct] * count)
        # Fill remaining slots randomly
        while len(content_types) < num_samples:
            content_types.append(random.choice(list(distribution.keys())))
        random.shuffle(content_types)
    else:
        # Single content type for all samples
        content_types = [content_type] * num_samples

    # Generate samples
    samples = []
    for i, sample_content_type in enumerate(content_types):
        # Ensure different seeds per sample (if seed is provided)
        sample_seed = seed + i * 1000 if seed is not None else None
        sample = generate_sample(
            content_type=sample_content_type,
            target_length=target_length,
            mean_fragment_length=mean_fragment_length,
            seed=sample_seed,
        )
        sample["id"] = i
        samples.append(sample)

    return samples
