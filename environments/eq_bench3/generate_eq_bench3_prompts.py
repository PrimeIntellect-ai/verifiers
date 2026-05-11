import argparse
import json
import random
from pathlib import Path

SCENARIOS = [
    (
        "Maya",
        "Noah",
        [
            "Noah admits he forgot to submit a shared application before the deadline.",
            "Maya had reminded him twice and had already rearranged her week around it.",
            "Noah says he was overwhelmed but did not want to disappoint her.",
        ],
    ),
    (
        "Priya",
        "Jon",
        [
            "Jon receives praise for an idea that Priya originally suggested in private.",
            "Priya notices Jon does not correct the group during the meeting.",
            "Afterward, Jon says he panicked and did not know how to speak up.",
        ],
    ),
    (
        "Eli",
        "Sam",
        [
            "Sam cancels a long-planned visit after getting an unexpected job interview.",
            "Eli says they understand, but had been counting down to the visit.",
            "Sam offers to reschedule and sounds worried about seeming selfish.",
        ],
    ),
    (
        "Rosa",
        "Theo",
        [
            "Theo sells an old guitar without realizing it belonged to Rosa's late brother.",
            "Rosa goes quiet when she sees the empty stand in the living room.",
            "Theo apologizes immediately and offers to contact the buyer.",
        ],
    ),
]

EMOTION_SETS = [
    ["hurt", "relieved", "angry", "hopeful"],
    ["embarrassed", "grateful", "defensive", "sad"],
    ["anxious", "resentful", "sympathetic", "dismissive"],
    ["guilty", "confused", "proud", "disappointed"],
]


def build_dialogue(speaker_a: str, speaker_b: str, beats: list[str], depth: int) -> str:
    lines = [
        f"{speaker_a}: I need to talk about what happened.",
        f"{speaker_b}: I know. I have been avoiding it because I feel awful.",
    ]
    for index, beat in enumerate(beats[:depth], start=1):
        if index % 2:
            lines.append(f"{speaker_a}: {beat}")
        else:
            lines.append(f"{speaker_b}: {beat}")
    return "\n".join(lines)


def score_emotions(
    target: str,
    speaker_a: str,
    speaker_b: str,
    emotions: list[str],
    depth: int,
) -> dict[str, int]:
    scores = {emotion: 0 for emotion in emotions}
    if target == speaker_a:
        scores[emotions[0]] = min(10, 3 + depth)
        scores[emotions[2]] = 2 + (depth // 2)
        scores[emotions[3]] = max(0, 5 - depth)
    else:
        scores[emotions[0]] = 2 + depth
        scores[emotions[1]] = 3
        scores[emotions[2]] = min(10, 4 + depth)
    return scores


def build_question(dialogue: str, target: str, emotions: list[str]) -> str:
    emotion_lines = "\n".join(f"- {emotion}" for emotion in emotions)
    return (
        "Your task is to predict the likely emotional reactions of a character "
        "in this dialogue.\n\n"
        f"{dialogue}\n"
        "[End of dialogue]\n\n"
        f"At the end of this dialogue, {target} would feel:\n"
        f"{emotion_lines}\n\n"
        "Assign each emotion an intensity from 0 to 10. Return only JSON with "
        "emotion names as keys and numeric scores as values."
    )


def generate_rows(num_examples: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, tuple[str, ...]]] = set()
    while len(rows) < num_examples:
        speaker_a, speaker_b, beats = rng.choice(SCENARIOS)
        target = rng.choice([speaker_a, speaker_b])
        emotions = rng.choice(EMOTION_SETS)
        depth = rng.randint(1, len(beats))
        key = (speaker_a, target, depth, tuple(emotions))
        if key in seen:
            continue
        seen.add(key)
        dialogue = build_dialogue(speaker_a, speaker_b, beats, depth)
        answer = score_emotions(target, speaker_a, speaker_b, emotions, depth)
        rows.append(
            {
                "question": build_question(dialogue, target, emotions),
                "answer": json.dumps(answer, sort_keys=True),
                "info": {
                    "benchmark": "eq-bench3",
                    "source": "synthetic-uncontaminated-v1",
                    "target": target,
                    "emotions": emotions,
                    "dialogue_depth": depth,
                },
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate uncontaminated EQ-Bench3-style training prompts."
    )
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "eq_bench3_train_sample.jsonl",
    )
    args = parser.parse_args()
    rows = generate_rows(num_examples=args.num_examples, seed=args.seed)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
