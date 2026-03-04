"""Generate a synthetic article corpus for bias detector training.

1. Generate articles using the base model with few-shot prompts
2. Score each article with a separate judge call
3. Filter for faithfulness (judge scores match target ±1 per criterion)
4. Save as expanded training corpus alongside real articles
"""

import asyncio
import json
import random
import re
from pathlib import Path

import httpx

ARTICLES_PATH = Path("/home/ubuntu/research/bias/articles.json")
OUTPUT_PATH = Path(__file__).parent / "synthetic_corpus.json"

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

TOPICS = [
    "Immigration", "Economy/Taxes/Inflation", "Healthcare",
    "Climate / Energy", "Foreign Policy", "Crime/Policing",
    "Education", "Technology/AI", "Housing/Real Estate",
    "Gun Policy", "Social Security/Medicare",
]


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


def select_fewshot(articles, target_c1, target_c2, target_c3, n=3):
    scored = []
    for a in articles:
        dist = abs(a["c1"] - target_c1) + abs(a["c2"] - target_c2) + abs(a["c3"] - target_c3)
        scored.append((dist, a))
    scored.sort(key=lambda x: x[0])
    pool = scored[:n + 2]
    random.shuffle(pool)
    return [x[1] for x in pool[:n]]


def format_examples(articles, max_text_len=1200):
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


def make_gen_prompt(topic, c1, c2, c3, examples_text):
    return f"""You are a news article generator for bias research.

Your task: Given a TOPIC and TARGET BIAS SCORES, write a realistic news article
that embodies those specific bias characteristics.

{CRITERIA_DESC}

Study the example articles below carefully — they show what different score
combinations look like in practice.

{examples_text}

YOUR ASSIGNMENT:
Topic: {topic}
Target scores: C1={c1}, C2={c2}, C3={c3}

Write a realistic 300-600 word news article on this topic that matches these
target bias scores. Think about how the scores should manifest in your writing:
- C1 affects which causes/remedies you emphasize
- C2 affects your word choices and tone
- C3 affects how many/what kinds of sources you cite

Output ONLY the article text, no metadata or scores."""


def make_judge_prompt(article, topic):
    return f"""You are a media bias analyst. Read the article and predict its bias scores.

{CRITERIA_DESC}

Topic: {topic}

ARTICLE:
---
{article[:4000]}
---

Predict integer scores for C1, C2, and C3 (each -3 to +3).
Respond with ONLY a JSON object: {{"c1_score": N, "c2_score": N, "c3_score": N}}"""


async def generate_article(client, topic, c1, c2, c3, examples_text):
    prompt = make_gen_prompt(topic, c1, c2, c3, examples_text)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Write the article now."},
        ],
        "max_tokens": 1024,
        "temperature": 1.0,
    }
    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


async def judge_article(client, article, topic):
    prompt = make_judge_prompt(article, topic)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Remove think tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    match = re.search(r'\{[^{}]*"c1_score"[^{}]*\}', text)
    if match:
        return json.loads(match.group(0))

    # Fallback: find any JSON
    brace = text.rfind("{")
    if brace >= 0:
        candidate = text[brace:]
        close = candidate.find("}")
        if close >= 0:
            return json.loads(candidate[: close + 1])

    raise ValueError(f"Could not parse judge response: {text[:200]}")


async def generate_and_judge(client, real_articles, topic, c1, c2, c3, article_id):
    fewshot = select_fewshot(real_articles, c1, c2, c3, n=3)
    examples_text = format_examples(fewshot)

    article_text = await generate_article(client, topic, c1, c2, c3, examples_text)
    if len(article_text) < 100:
        return None

    scores = await judge_article(client, article_text, topic)
    jc1 = int(scores.get("c1_score", 0))
    jc2 = int(scores.get("c2_score", 0))
    jc3 = int(scores.get("c3_score", 0))

    faithful = (
        abs(jc1 - c1) <= 1 and
        abs(jc2 - c2) <= 1 and
        abs(jc3 - c3) <= 1
    )

    return {
        "id": article_id,
        "topic": topic,
        "target_c1": c1, "target_c2": c2, "target_c3": c3,
        "judge_c1": jc1, "judge_c2": jc2, "judge_c3": jc3,
        "text": article_text,
        "faithful": faithful,
        "synthetic": True,
    }


async def main():
    real_articles = load_real_articles()
    print(f"Loaded {len(real_articles)} real articles")

    # Generate diverse target score combinations
    # Cover the full -3 to +3 range with emphasis on under-represented C1+ scores
    targets = []

    # Method 1: Sample from real distribution (natural coverage)
    for art in real_articles:
        for topic in random.sample(TOPICS, 2):
            targets.append((topic, art["c1"], art["c2"], art["c3"]))

    # Method 2: Systematic coverage of C1+ space (under-represented)
    for c1 in [1, 2, 3]:
        for c2 in [-2, -1, 0, 1, 2]:
            for c3 in [-2, -1, 0, 1, 2]:
                topic = random.choice(TOPICS)
                targets.append((topic, c1, c2, c3))
                targets.append((random.choice(TOPICS), c1, c2, c3))

    # Method 3: Extreme combinations for calibration
    for c1 in [-3, 3]:
        for c2 in [-3, 3]:
            for c3 in [-3, 3]:
                targets.append((random.choice(TOPICS), c1, c2, c3))

    random.shuffle(targets)
    print(f"Total generation targets: {len(targets)}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Process in batches of 8 to avoid overwhelming the server
        batch_size = 8
        all_results = []
        faithful_count = 0

        for batch_start in range(0, len(targets), batch_size):
            batch = targets[batch_start:batch_start + batch_size]
            tasks = []
            for i, (topic, c1, c2, c3) in enumerate(batch):
                aid = 10000 + batch_start + i
                tasks.append(generate_and_judge(client, real_articles, topic, c1, c2, c3, aid))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    print(f"  Error: {r}")
                    continue
                if r is None:
                    continue
                all_results.append(r)
                if r["faithful"]:
                    faithful_count += 1

            total = len(all_results)
            pct = 100 * faithful_count / total if total > 0 else 0
            print(f"Batch {batch_start//batch_size + 1}/{(len(targets) + batch_size - 1)//batch_size}: "
                  f"{total} total, {faithful_count} faithful ({pct:.0f}%)")

            # Stop if we have enough faithful articles
            if faithful_count >= 400:
                print(f"Reached {faithful_count} faithful articles, stopping.")
                break

    # Save all results (faithful and not)
    output = {
        "real_articles": real_articles,
        "synthetic_articles": all_results,
        "faithful_articles": [r for r in all_results if r["faithful"]],
        "stats": {
            "total_generated": len(all_results),
            "faithful": faithful_count,
            "faithfulness_rate": faithful_count / len(all_results) if all_results else 0,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total: {len(all_results)}, Faithful: {faithful_count} ({output['stats']['faithfulness_rate']:.0%})")

    # Score distribution of faithful articles
    faithful = output["faithful_articles"]
    if faithful:
        from collections import Counter
        c1_dist = Counter(a["judge_c1"] for a in faithful)
        print(f"\nFaithful articles C1 distribution:")
        for k in sorted(c1_dist):
            print(f"  C1={k:+d}: {c1_dist[k]}")


if __name__ == "__main__":
    asyncio.run(main())
