"""Evaluate a bias detector checkpoint on ALL real articles.

Usage:
    python3 eval_checkpoint.py [checkpoint_dir] [--base-only]

If no checkpoint_dir, evaluates the base model (no LoRA).
With --base-only, skips LoRA loading even if checkpoint is specified.

Requires a vLLM server running on localhost:8000.
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter

import httpx

ARTICLES_PATH = Path("/home/ubuntu/research/bias/articles.json")
BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

CRITERIA_DESC = """BIAS SCORING CRITERIA (each scored -3 to +3):

C1 — Frame (Cause & Remedy):
  Negative (-): structural/collective causes (systemic inequality, institutional failure)
  Zero (0): no clear causal frame or balanced
  Positive (+): agency/order/market causes (individual responsibility, market solutions)

C2 — Temperature (Rhetorical Heat):
  Negative (-): loaded/moralizing/alarmist language
  Zero (0): neutral newsroom baseline
  Positive (+): measured/clinical/careful hedging

C3 — Evidence Diversity:
  Negative (-): single-mode or one-sided sourcing
  Zero (0): uneven or mixed sourcing
  Positive (+): multiple evidence modes with counter-perspectives"""


def load_real_articles():
    with open(ARTICLES_PATH) as f:
        raw = json.load(f)
    usable = []
    for a in raw:
        try:
            c1 = int(a.get("c1_score_a", ""))
            c2 = int(a.get("c2_score_a", ""))
            c3 = int(a.get("c3_score_a", ""))
            text = a.get("article_text", "")
            if text and len(text) > 100:
                usable.append({
                    "id": a["id"], "topic": str(a.get("topic", "General")).strip(),
                    "headline": str(a.get("headline", "")),
                    "source": str(a.get("source", "")),
                    "c1": c1, "c2": c2, "c3": c3, "text": text,
                })
        except (ValueError, TypeError):
            pass
    return usable


def select_diverse_fewshot(articles, target_c1, target_c2, target_c3, n=5, exclude_id=-1):
    available = [a for a in articles if a["id"] != exclude_id]
    if len(available) <= n:
        return available
    scored = [(abs(a["c1"]-target_c1)+abs(a["c2"]-target_c2)+abs(a["c3"]-target_c3), a)
              for a in available]
    scored.sort(key=lambda x: x[0])
    selected = [scored[0][1], scored[1][1]]
    selected_ids = {s["id"] for s in selected}
    remaining = [a for a in available if a["id"] not in selected_ids]
    remaining.sort(key=lambda a: -(abs(a["c1"]-target_c1)+abs(a["c2"]-target_c2)+abs(a["c3"]-target_c3)))
    if remaining:
        selected.append(remaining[0])
        selected_ids.add(remaining[0]["id"])
    mid = [a for a in available if a["id"] not in selected_ids]
    mid.sort(key=lambda a: abs(abs(a["c1"]-target_c1)+abs(a["c2"]-target_c2)+abs(a["c3"]-target_c3) - 6))
    for a in mid[:n - len(selected)]:
        selected.append(a)
    random.shuffle(selected)
    return selected[:n]


def format_examples(articles, max_text_len=1500):
    parts = []
    for i, a in enumerate(articles, 1):
        text = a["text"][:max_text_len]
        if len(a["text"]) > max_text_len:
            text += "..."
        parts.append(
            f"--- Example {i} ---\n"
            f"Source: {a['source']} | Topic: {a['topic']}\n"
            f"Headline: {a['headline']}\n"
            f"Scores: C1={a['c1']}, C2={a['c2']}, C3={a['c3']}\n"
            f"Article (excerpt):\n{text}\n"
        )
    return "\n".join(parts)


def make_disc_prompt(article, topic, examples_text):
    return f"""You are a media bias analyst.

Your task: Read the article below and predict its bias scores on three criteria.

{CRITERIA_DESC}

IMPORTANT SCORING NOTES:
- C1 is about the CAUSAL FRAME the article uses, not the topic. An article about
  systemic problems can still use an individual-agency frame (C1 positive) if it
  emphasizes personal responsibility as the solution.
- C2 is about RHETORICAL STYLE. Negative means heated/emotional language.
  Positive means calm/clinical. This is independent of the topic.
- C3 is about SOURCE DIVERSITY. Count distinct types of evidence: statistics,
  expert quotes, government data, anecdotes, counter-arguments. More types = higher.

Study the example articles below carefully — they show what different score
combinations look like in practice.

{examples_text}

Topic: {topic}
ARTICLE TO ANALYZE:
---
{article[:6000]}
---

Analyze this article and predict integer scores for C1, C2, and C3 (each -3 to +3).
Submit your predictions in the handoff block."""


async def evaluate_article(client, article, all_articles):
    fewshot = select_diverse_fewshot(
        all_articles, article["c1"], article["c2"], article["c3"],
        n=5, exclude_id=article["id"]
    )
    examples_text = format_examples(fewshot)
    prompt = make_disc_prompt(article["text"], article["topic"], examples_text)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": 'Analyze the article and predict bias scores. Submit in a <handoff> block with format: {"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "brief explanation"}'},
        ],
        "max_tokens": 768,
        "temperature": 0.0,
    }
    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Remove think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Parse scores
    match = re.search(r'"c1_score"\s*:\s*(-?\d+)\s*,\s*"c2_score"\s*:\s*(-?\d+)\s*,\s*"c3_score"\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Fallback: find JSON
    brace = text.rfind("{")
    if brace >= 0:
        candidate = text[brace:]
        close = candidate.find("}")
        if close >= 0:
            data = json.loads(candidate[:close + 1])
            return int(data.get("c1_score", 0)), int(data.get("c2_score", 0)), int(data.get("c3_score", 0))

    raise ValueError(f"Could not parse: {text[:200]}")


async def main():
    articles = load_real_articles()
    print(f"Evaluating on {len(articles)} real articles")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check server is running
        resp = await client.get(f"{BASE_URL}/models")
        models = resp.json()
        print(f"Model: {models['data'][0]['id']}")

        results = []
        batch_size = 4
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            tasks = [evaluate_article(client, a, articles) for a in batch]
            preds = await asyncio.gather(*tasks, return_exceptions=True)

            for a, p in zip(batch, preds):
                if isinstance(p, Exception):
                    print(f"  Error for article {a['id']}: {p}")
                    continue
                pc1, pc2, pc3 = p
                c1e = abs(pc1 - a["c1"])
                c2e = abs(pc2 - a["c2"])
                c3e = abs(pc3 - a["c3"])
                results.append({
                    "id": a["id"], "topic": a["topic"],
                    "true": (a["c1"], a["c2"], a["c3"]),
                    "pred": (pc1, pc2, pc3),
                    "c1_err": c1e, "c2_err": c2e, "c3_err": c3e,
                })

            print(f"  Evaluated {min(i+batch_size, len(articles))}/{len(articles)}")

    # Summary stats
    n = len(results)
    c1_mae = sum(r["c1_err"] for r in results) / n
    c2_mae = sum(r["c2_err"] for r in results) / n
    c3_mae = sum(r["c3_err"] for r in results) / n
    avg_mae = (c1_mae + c2_mae + c3_mae) / 3
    perfect = sum(1 for r in results if r["c1_err"]+r["c2_err"]+r["c3_err"] == 0)

    print(f"\n{'='*60}")
    print(f"RESULTS: {n} articles evaluated")
    print(f"  Overall MAE: {avg_mae:.2f}")
    print(f"  C1 MAE: {c1_mae:.2f}")
    print(f"  C2 MAE: {c2_mae:.2f}")
    print(f"  C3 MAE: {c3_mae:.2f}")
    print(f"  Perfect (0 total error): {perfect}/{n} ({100*perfect/n:.0f}%)")
    print(f"  <=1 total error: {sum(1 for r in results if r['c1_err']+r['c2_err']+r['c3_err']<=1)}/{n}")
    print(f"  <=3 total error: {sum(1 for r in results if r['c1_err']+r['c2_err']+r['c3_err']<=3)}/{n}")

    # Per-criterion breakdown
    print(f"\nPer-criterion breakdown:")
    for crit, key in [("C1", "c1_err"), ("C2", "c2_err"), ("C3", "c3_err")]:
        errs = [r[key] for r in results]
        print(f"  {crit}: MAE={sum(errs)/n:.2f} | Perfect={sum(1 for e in errs if e==0)}/{n} | <=1={sum(1 for e in errs if e<=1)}/{n}")

    # By C1 sign
    print(f"\nBy C1 sign:")
    for label, cond in [("C1-neg (<=−1)", lambda r: r["true"][0] <= -1),
                         ("C1-zero (0)", lambda r: r["true"][0] == 0),
                         ("C1-pos (>=1)", lambda r: r["true"][0] >= 1)]:
        subset = [r for r in results if cond(r)]
        if subset:
            mae = sum(r["c1_err"]+r["c2_err"]+r["c3_err"] for r in subset) / (3*len(subset))
            c1m = sum(r["c1_err"] for r in subset) / len(subset)
            print(f"  {label}: n={len(subset)}, MAE={mae:.2f}, C1_MAE={c1m:.2f}")

    # Worst articles
    results.sort(key=lambda r: -(r["c1_err"]+r["c2_err"]+r["c3_err"]))
    print(f"\nWorst 10 articles:")
    for r in results[:10]:
        total_err = r["c1_err"]+r["c2_err"]+r["c3_err"]
        print(f"  id={r['id']:3d} [{r['topic'][:20]:20s}] true={r['true']} pred={r['pred']} err=({r['c1_err']},{r['c2_err']},{r['c3_err']}) total={total_err}")


if __name__ == "__main__":
    asyncio.run(main())
