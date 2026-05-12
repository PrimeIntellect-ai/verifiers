from __future__ import annotations

import base64
import random
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from datasets import Dataset
from PIL import Image, ImageDraw

from verifiers.messages import add_image

HopType = Literal["perception", "relational", "compose", "tool_hint"]


@dataclass
class HopSpec:
    type: HopType
    question: str
    ground_truth: str
    grounding_norm: tuple[float, float, float, float] | None = None


def _draw_scene(
    rng: random.Random,
    size: tuple[int, int] = (320, 240),
) -> tuple[Image.Image, dict[str, Any]]:
    w, h = size
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    n_red = rng.randint(1, 5)
    n_blue = rng.randint(1, 5)
    circles: list[dict[str, Any]] = []
    for i in range(n_red + n_blue):
        cx = rng.randint(30, w - 30)
        cy = rng.randint(30, h - 30)
        r = rng.randint(12, 22)
        color = (220, 60, 60) if i < n_red else (60, 100, 220)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(0, 0, 0))
        circles.append(
            {
                "cx": cx,
                "cy": cy,
                "r": r,
                "color": "red" if color == (220, 60, 60) else "blue",
            }
        )
    meta = {"n_red": n_red, "n_blue": n_blue, "circles": circles}
    return img, meta


def _bbox_for_color(circles: list[dict[str, Any]], color: str) -> tuple[float, float, float, float]:
    subset = [c for c in circles if c["color"] == color]
    if not subset:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [c["cx"] for c in subset]
    ys = [c["cy"] for c in subset]
    rs = [c["r"] for c in subset]
    min_x = min(x - r for x, r in zip(xs, rs))
    max_x = max(x + r for x, r in zip(xs, rs))
    min_y = min(y - r for y, r in zip(ys, rs))
    max_y = max(y + r for y, r in zip(ys, rs))
    return (float(min_x), float(min_y), float(max_x), float(max_y))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ba = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _build_hops(meta: dict[str, Any]) -> list[HopSpec]:
    n_red = meta["n_red"]
    n_blue = meta["n_blue"]
    bbox_red = _bbox_for_color(meta["circles"], "red")
    hops: list[HopSpec] = [
        HopSpec(
            type="perception",
            question=(
                "Hop 1 — perception: How many **red** circles are in the image? "
                "Reply with your reasoning, a grounding tag for the red region like "
                '<grounding bbox="x0,y0,x1,y1" desc="red circles"/>, '
                "and your numeric answer inside <hop_answer></hop_answer>."
            ),
            ground_truth=str(n_red),
            grounding_norm=bbox_red,
        ),
        HopSpec(
            type="perception",
            question=(
                "Hop 2 — perception: How many **blue** circles are in the image? "
                "Include <grounding bbox=\"x0,y0,x1,y1\" desc=\"blue circles\"/> "
                "and <hop_answer>your count</hop_answer>."
            ),
            ground_truth=str(n_blue),
            grounding_norm=_bbox_for_color(meta["circles"], "blue"),
        ),
        HopSpec(
            type="relational",
            question=(
                "Hop 3 — relational: What is the **sum** of the red count and blue count "
                "from the previous hops? Put only the integer in <hop_answer></hop_answer> "
                "and the final boxed answer \\boxed{n} on the last line."
            ),
            ground_truth=str(n_red + n_blue),
            grounding_norm=None,
        ),
    ]
    return hops


def _image_to_b64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _format_initial_user_text(hops: list[HopSpec], system_hint: str) -> str:
    return (
        f"{system_hint}\n\n"
        "You will answer a chain of visual questions about the **same** image. "
        "Each hop builds on prior answers; re-ground in the image when needed.\n\n"
        f"{hops[0].question}"
    )


def synthesize(
    num_samples: int = 64,
    image_source: Literal["procedural"] = "procedural",
    min_hops: int = 3,
    max_hops: int = 3,
    seed: int = 0,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
) -> Dataset:
    """
    Build a VeriHop dataset with procedural scenes and a fixed 3-hop chain (count red,
    count blue, sum). ``min_hops`` / ``max_hops`` / ``difficulty`` are reserved for
    future curriculum extensions; currently all samples use three hops.
    """
    if image_source != "procedural":
        raise ValueError("Only image_source='procedural' is implemented in this release.")
    if min_hops != 3 or max_hops != 3:
        pass

    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    hint = {
        "easy": "Use short reasoning.",
        "medium": "Think step-by-step briefly.",
        "hard": "Explain each step carefully before answering.",
    }[difficulty]

    for i in range(num_samples):
        img, meta = _draw_scene(rng)
        hops = _build_hops(meta)
        user_msg: dict[str, Any] = {"role": "user", "content": []}
        user_msg["content"] = _format_initial_user_text(hops, hint)
        add_image(user_msg, img)
        prompt = [user_msg]
        final = hops[-1].ground_truth
        info = {
            "verihop": {
                "hops": [
                    {
                        "type": h.type,
                        "question": h.question,
                        "ground_truth": h.ground_truth,
                        "grounding_norm": h.grounding_norm,
                    }
                    for h in hops
                ],
                "image_b64": _image_to_b64_png(img),
                "meta": meta,
            }
        }
        rows.append(
            {
                "prompt": prompt,
                "answer": final,
                "info": info,
                "task": "verihop",
            }
        )

    return Dataset.from_list(rows)


def parse_hop_answer(text: str) -> str | None:
    m = re.search(r"<hop_answer>\s*(.*?)\s*</hop_answer>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def parse_grounding_boxes(text: str) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for m in re.finditer(
        r'<grounding[^>]*bbox\s*=\s*["\']([^"\']+)["\']', text, re.IGNORECASE
    ):
        raw = m.group(1).strip()
        parts = re.split(r"[,\s]+", raw)
        if len(parts) != 4:
            continue
        try:
            nums = tuple(float(x) for x in parts)
            out.append((nums[0], nums[1], nums[2], nums[3]))
        except ValueError:
            continue
    return out
