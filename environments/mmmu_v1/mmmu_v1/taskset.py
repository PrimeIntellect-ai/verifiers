"""mmmu-v1 — MMMU multimodal multiple-choice benchmark (the v1 port of `mmmu`).

Each row shows one image plus a four-option multiple-choice question; the model
reasons and answers with the option letter inside \\boxed{}. The prompt (question
text + options + the image as a base64 PNG data URL) and the boxed-letter
exact-match scoring mirror the v0 environment one-to-one.
"""

import ast
import base64
import re
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
    "Management",
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

LETTERS = "ABCD"


def image_data_url(pil_image) -> str:
    """The row's PIL image as a base64 PNG `data:` URL."""
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def question_text(question: str, options: list[str]) -> str:
    """The v0 prompt text verbatim: question, lettered options, boxed-answer instruction."""
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
    """The ground-truth option letter (A-D)."""


class MMMUTask(vf.Task[MMMUData]):
    @vf.reward(weight=1.0)
    async def correct_answer(self, trace: vf.Trace) -> float:
        return 1.0 if parse_letter(trace.last_reply) == self.data.answer else 0.0

    @vf.metric
    async def has_boxed_answer(self, trace: vf.Trace) -> float:
        return 1.0 if vf.extract_boxed_answer(trace.last_reply, strict=True) else 0.0


class MMMUConfig(vf.TasksetConfig):
    subset: str | None = "Art"
    """MMMU subject subset; `None` loads all 30 subjects."""
    split: Literal["dev", "validation", "test"] = "dev"


class MMMUTaskset(vf.Taskset[MMMUTask, MMMUConfig]):
    def load(self) -> list[MMMUTask]:
        from datasets import load_dataset

        c = self.config
        if c.subset is not None and c.subset not in ALL_SUBSETS:
            raise ValueError(f"Invalid subset: {c.subset}")
        subsets = ALL_SUBSETS if c.subset is None else [c.subset]

        tasks: list[MMMUTask] = []
        for subset in subsets:
            for row in load_dataset("MMMU/MMMU", subset, split=c.split):
                options = ast.literal_eval(row["options"])
                assert len(options) == 4  # v0 supports exactly A-D rows
                parts = [
                    vf.TextContentPart(text=question_text(row["question"], options)),
                    vf.ImageUrlContentPart(
                        image_url=vf.ImageUrlSource(url=image_data_url(row["image_1"]))
                    ),
                ]
                tasks.append(
                    MMMUTask(
                        MMMUData(
                            idx=len(tasks),
                            name=row["id"],
                            prompt=[vf.UserMessage(content=parts)],
                            answer=row["answer"],
                        ),
                        c.task,
                    )
                )
        return tasks
