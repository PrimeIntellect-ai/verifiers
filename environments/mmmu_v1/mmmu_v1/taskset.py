"""mmmu-v1 — MMMU multimodal multiple-choice benchmark.

Each row shows one or more images plus a multiple-choice question; the model
reasons and answers with the option letter inside \\boxed{}. The prompt carries
the question text (with its `<image N>` markers) plus every non-null
`image_1`..`image_7` as base64 PNG data-URL parts, in marker order. Options are
lettered A.. dynamically (MMMU rows range from 2 to 9 choices); open-ended rows
(no options) are skipped — this environment scores multiple choice only.
"""

import ast
import base64
import re
from collections.abc import Iterator
from io import BytesIO
from typing import Literal

import verifiers.v1 as vf

ALL_SUBSETS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

LETTERS = "ABCDEFGHI"


def image_data_url(pil_image) -> str:
    """The row's PIL image as a base64 PNG `data:` URL."""
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def question_text(question: str, options: list[str]) -> str:
    """Question, lettered options, and the boxed-answer instruction."""
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return (
        f"{question}\n\n"
        + "\n".join(lines)
        + "\n\nThink step-by-step and give the letter of your final answer inside \\boxed{}."
    )


def parse_letter(text: str) -> str:
    """The boxed answer, with an optional LaTeX `\\text{...}` wrapper stripped
    (models often box `\\text{C}` rather than the bare letter)."""
    boxed = vf.extract_boxed_answer(text, strict=True).strip()
    match = re.fullmatch(r"\\text\{(.*)\}", boxed)
    return (match.group(1) if match else boxed).strip()


class MMMUData(vf.TaskData):
    answer: str
    """The ground-truth option letter (A-I)."""


class MMMUTask(vf.Task[MMMUData]):
    @vf.reward(weight=1.0)
    async def correct_answer(self, trace: vf.Trace) -> float:
        return 1.0 if parse_letter(trace.last_reply) == self.data.answer else 0.0

    @vf.metric
    async def has_boxed_answer(self, trace: vf.Trace) -> float:
        return 1.0 if vf.extract_boxed_answer(trace.last_reply, strict=True) else 0.0


class MMMUConfig(vf.TasksetConfig):
    subset: str | None = None
    """MMMU subject subset; `None` loads all subjects."""
    split: Literal["dev", "validation", "test"] = "dev"


class MMMUTaskset(vf.Taskset[MMMUTask, MMMUConfig]):
    def load(self) -> Iterator[MMMUTask]:
        from datasets import load_dataset

        c = self.config
        if c.subset is not None and c.subset not in ALL_SUBSETS:
            raise ValueError(f"Invalid subset: {c.subset}")
        subsets = ALL_SUBSETS if c.subset is None else [c.subset]

        idx = 0
        for subset in subsets:
            for row in load_dataset("MMMU/MMMU", subset, split=c.split):
                options = ast.literal_eval(row["options"])
                if not options:  # open-ended row; only multiple choice is scored
                    continue
                segments = re.split(
                    r"<image ([1-7])>", question_text(row["question"], options)
                )
                parts: list[vf.ContentPart] = []
                for i, segment in enumerate(segments):
                    if i % 2:
                        image = row[f"image_{segment}"]
                        parts.append(
                            vf.ImageUrlContentPart(
                                image_url=vf.ImageUrlSource(url=image_data_url(image))
                            )
                        )
                    elif segment:
                        parts.append(vf.TextContentPart(text=segment))
                yield MMMUTask(
                    MMMUData(
                        idx=idx,
                        name=row["id"],
                        prompt=[vf.UserMessage(content=parts)],
                        answer=row["answer"],
                    ),
                    c.task,
                )
                idx += 1
