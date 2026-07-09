import random
import re
from typing import Literal

from datasets import Dataset

import verifiers as vf

Parity = Literal["e", "o"]
Irrep = tuple[int, Parity]

IRREP_RE = re.compile(r"^(\d+)([eo])$")
SYSTEM_PROMPT = (
    "Solve O(3) irreducible representation tensor product decompositions. "
    "Return only the final decomposition inside <answer> tags."
)


def _validate_config(
    min_l: int,
    max_l: int,
    include_even: bool,
    include_odd: bool,
    num_train_examples: int,
    num_eval_examples: int,
) -> None:
    if min_l < 0:
        raise ValueError("min_l must be non-negative")
    if max_l < min_l:
        raise ValueError("max_l must be greater than or equal to min_l")
    if not include_even and not include_odd:
        raise ValueError("At least one parity must be enabled")
    if num_train_examples < 0:
        raise ValueError("num_train_examples must be non-negative")
    if num_eval_examples < 0:
        raise ValueError("num_eval_examples must be non-negative")


def _multiply_parity(left: Parity, right: Parity) -> Parity:
    return "e" if left == right else "o"


def _format_irrep(irrep: Irrep) -> str:
    l_value, parity = irrep
    return f"{l_value}{parity}"


def _decompose(left: Irrep, right: Irrep) -> list[Irrep]:
    left_l, left_parity = left
    right_l, right_parity = right
    parity = _multiply_parity(left_parity, right_parity)
    return [
        (l_value, parity)
        for l_value in range(abs(left_l - right_l), left_l + right_l + 1)
    ]


def _format_decomposition(terms: list[Irrep]) -> str:
    return " + ".join(_format_irrep(term) for term in terms)


def _normalize_decomposition(answer: str | None) -> str | None:
    if answer is None:
        return None
    terms = [term.strip() for term in answer.strip().split("+")]
    if not terms or any(not term for term in terms):
        return None
    if any(IRREP_RE.fullmatch(term) is None for term in terms):
        return None
    return " + ".join(terms)


def _build_question(left: Irrep, right: Irrep) -> str:
    left_text = _format_irrep(left)
    right_text = _format_irrep(right)
    return (
        "Decompose the tensor product of two O(3) irreducible representations.\n\n"
        "Use e3nn-style notation: l followed by parity e or o. "
        "The tensor product l1p1 x l2p2 contains every l from |l1-l2| to l1+l2, "
        "and output parity is e when input parities match, otherwise o.\n\n"
        f"Problem: {left_text} x {right_text}\n\n"
        "Write the decomposition in ascending l order, separated by ` + `."
    )


def _make_example(
    idx: int,
    seed: int,
    min_l: int,
    max_l: int,
    include_even: bool,
    include_odd: bool,
) -> dict:
    rng = random.Random(seed + idx)
    parities: list[Parity] = []
    if include_even:
        parities.append("e")
    if include_odd:
        parities.append("o")

    left = (rng.randint(min_l, max_l), rng.choice(parities))
    right = (rng.randint(min_l, max_l), rng.choice(parities))
    terms = _decompose(left, right)
    answer = _format_decomposition(terms)
    return {
        "question": _build_question(left, right),
        "answer": answer,
        "info": {
            "source_dataset": "irreps_tensor_product",
            "source_index": idx,
            "operands": [_format_irrep(left), _format_irrep(right)],
            "left": {"l": left[0], "parity": left[1]},
            "right": {"l": right[0], "parity": right[1]},
            "terms": [_format_irrep(term) for term in terms],
            "min_l": min_l,
            "max_l": max_l,
        },
    }


def _build_dataset(
    count: int,
    seed: int,
    offset: int,
    min_l: int,
    max_l: int,
    include_even: bool,
    include_odd: bool,
) -> Dataset:
    return Dataset.from_list(
        [
            _make_example(
                idx=offset + idx,
                seed=seed,
                min_l=min_l,
                max_l=max_l,
                include_even=include_even,
                include_odd=include_odd,
            )
            for idx in range(count)
        ]
    )


def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 2000,
    seed: int = 0,
    min_l: int = 0,
    max_l: int = 4,
    include_even: bool = True,
    include_odd: bool = True,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> vf.Environment:
    """Load a standalone O(3) irreps tensor product Verifiers environment."""

    _validate_config(
        min_l=min_l,
        max_l=max_l,
        include_even=include_even,
        include_odd=include_odd,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
    )

    parser = vf.XMLParser(fields=["answer"])

    def exact_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion)
        return float(_normalize_decomposition(response) == answer)

    rubric = vf.Rubric(parser=parser, funcs=[exact_reward_func])
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.0)

    dataset = _build_dataset(
        count=num_train_examples,
        seed=seed,
        offset=0,
        min_l=min_l,
        max_l=max_l,
        include_even=include_even,
        include_odd=include_odd,
    )
    eval_dataset = _build_dataset(
        count=num_eval_examples,
        seed=seed,
        offset=num_train_examples,
        min_l=min_l,
        max_l=max_l,
        include_even=include_even,
        include_odd=include_odd,
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        message_type="chat",
    )
