"""Re-score synthetic articles with few-shot judge and generate more articles.

The zero-shot judge was heavily biased (C2: 87% zero, C3: 93% positive).
This script uses the SAME few-shot prompt format as the discriminator training,
which provides calibration examples that should produce more accurate scores.

Also generates additional articles targeting underrepresented score regions.
"""

import asyncio
import json
import random
import re
from pathlib import Path
from collections import Counter

import httpx

ARTICLES_PATH = Path("/home/ubuntu/research/bias/articles.json")
SYNTHETIC_PATH = Path(__file__).parent / "synthetic_corpus.json"
OUTPUT_PATH = Path(__file__).parent / "corpus_v2.json"

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
        c1 = int(a.get("c1_score_a", "")) if a.get("c1_score_a", "") != "" else None
        c2 = int(a.get("c2_score_a", "")) if a.get("c2_score_a", "") != "" else None
        c3 = int(a.get("c3_score_a", "")) if a.get("c3_score_a", "") != "" else None
        if c1 is None or c2 is None or c3 is None:
            continue
        text = a.get("article_text", "")
        if text and len(text) > 100:
            usable.append({
                "id": a["id"], "topic": str(a.get("topic", "General")).strip(),
                "headline": str(a.get("headline", "")),
                "source": str(a.get("source", "")),
                "c1": c1, "c2": c2, "c3": c3, "text": text,
            })
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


def make_fewshot_judge_prompt(article_text, topic, real_articles):
    """Build a judge prompt with 5 diverse few-shot examples — same as disc training."""
    # Use target (0,0,0) for "neutral" fewshot selection, giving broad coverage
    fewshot = select_diverse_fewshot(real_articles, 0, 0, 0, n=5)
    examples_text = format_examples(fewshot)

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
{article_text[:5000]}
---

Analyze this article and predict integer scores for C1, C2, and C3 (each -3 to +3).
Respond with ONLY a JSON object: {{"c1_score": N, "c2_score": N, "c3_score": N}}"""


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


async def judge_article(client, article_text, topic, real_articles):
    """Score an article using few-shot judge prompt."""
    prompt = make_fewshot_judge_prompt(article_text, topic, real_articles)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Remove think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    match = re.search(r'\{[^{}]*"c1_score"[^{}]*\}', text)
    if match:
        return json.loads(match.group(0))

    brace = text.rfind("{")
    if brace >= 0:
        candidate = text[brace:]
        close = candidate.find("}")
        if close >= 0:
            return json.loads(candidate[:close + 1])

    raise ValueError(f"Could not parse judge response: {text[:200]}")


async def generate_article(client, topic, c1, c2, c3, real_articles):
    """Generate a synthetic article with few-shot examples."""
    fewshot = select_diverse_fewshot(real_articles, c1, c2, c3, n=3)
    examples_text = format_examples(fewshot, max_text_len=1200)
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


async def rescore_batch(client, articles, real_articles, batch_size=6):
    """Re-score a list of articles with few-shot judge."""
    results = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        tasks = [
            judge_article(client, a["text"], a.get("topic", "General"), real_articles)
            for a in batch
        ]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        for a, s in zip(batch, scores):
            if isinstance(s, Exception):
                print(f"  Rescore error for article {a.get('id', '?')}: {s}")
                continue
            a["judge_c1"] = int(s.get("c1_score", 0))
            a["judge_c2"] = int(s.get("c2_score", 0))
            a["judge_c3"] = int(s.get("c3_score", 0))
            results.append(a)
        print(f"  Rescored {min(i+batch_size, len(articles))}/{len(articles)}")
    return results


async def generate_and_judge_batch(client, targets, real_articles, start_id=20000, batch_size=6):
    """Generate articles for target scores, then judge them."""
    results = []
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i+batch_size]
        # Generate
        gen_tasks = [
            generate_article(client, topic, c1, c2, c3, real_articles)
            for topic, c1, c2, c3 in batch
        ]
        gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

        # Judge the successful generations
        to_judge = []
        for (topic, c1, c2, c3), gen in zip(batch, gen_results):
            if isinstance(gen, Exception):
                print(f"  Gen error: {gen}")
                continue
            if len(gen) < 100:
                continue
            to_judge.append({
                "id": start_id + i + len(to_judge),
                "topic": topic,
                "target_c1": c1, "target_c2": c2, "target_c3": c3,
                "text": gen,
                "synthetic": True,
            })

        if not to_judge:
            continue

        judge_tasks = [
            judge_article(client, a["text"], a["topic"], real_articles)
            for a in to_judge
        ]
        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        for a, s in zip(to_judge, judge_results):
            if isinstance(s, Exception):
                print(f"  Judge error: {s}")
                continue
            a["judge_c1"] = int(s.get("c1_score", 0))
            a["judge_c2"] = int(s.get("c2_score", 0))
            a["judge_c3"] = int(s.get("c3_score", 0))
            faithful = (
                abs(a["judge_c1"] - a["target_c1"]) <= 1 and
                abs(a["judge_c2"] - a["target_c2"]) <= 1 and
                abs(a["judge_c3"] - a["target_c3"]) <= 1
            )
            a["faithful"] = faithful
            results.append(a)

        total = len(results)
        faithful_n = sum(1 for r in results if r.get("faithful", False))
        print(f"  Generated {min(i+batch_size, len(targets))}/{len(targets)} — "
              f"{total} articles, {faithful_n} faithful")
    return results


async def main():
    real_articles = load_real_articles()
    print(f"Loaded {len(real_articles)} real articles")

    # Phase 1: Re-score existing synthetic articles
    print("\n=== Phase 1: Re-scoring existing synthetic articles ===")
    existing = json.load(open(SYNTHETIC_PATH))
    existing_articles = existing["synthetic_articles"]
    print(f"Existing synthetic articles: {len(existing_articles)}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        rescored = await rescore_batch(client, existing_articles, real_articles, batch_size=6)
        print(f"Successfully rescored: {len(rescored)}")

        # Check score distribution improvement
        jc1 = Counter(a["judge_c1"] for a in rescored)
        jc2 = Counter(a["judge_c2"] for a in rescored)
        jc3 = Counter(a["judge_c3"] for a in rescored)
        print(f"Rescored C1 dist: {dict(sorted(jc1.items()))}")
        print(f"Rescored C2 dist: {dict(sorted(jc2.items()))}")
        print(f"Rescored C3 dist: {dict(sorted(jc3.items()))}")

        # Phase 2: Generate more articles targeting gaps
        print("\n=== Phase 2: Generating additional articles ===")

        # Identify underrepresented score regions
        # Target: even coverage across C1, C2, C3 ranges
        targets = []

        # Systematic grid — all interesting combinations
        for c1 in range(-3, 4):
            for c2 in [-3, -2, -1, 0, 1, 2, 3]:
                for c3 in [-2, -1, 0, 1, 2]:
                    topic = random.choice(TOPICS)
                    targets.append((topic, c1, c2, c3))

        # Extra coverage for extreme C2 values (most underrepresented)
        for _ in range(30):
            c2 = random.choice([-3, -2, 2, 3])
            c1 = random.randint(-3, 3)
            c3 = random.randint(-2, 2)
            targets.append((random.choice(TOPICS), c1, c2, c3))

        random.shuffle(targets)
        # Cap at 300 generations to keep it reasonable
        targets = targets[:300]
        print(f"Generation targets: {len(targets)}")

        new_articles = await generate_and_judge_batch(
            client, targets, real_articles, start_id=20000, batch_size=6
        )
        print(f"New articles generated: {len(new_articles)}")

    # Combine rescored + new
    all_synthetic = rescored + new_articles
    faithful = [a for a in all_synthetic if a.get("faithful", False)]

    # Final distributions
    print(f"\n=== Final Corpus Stats ===")
    print(f"Total synthetic: {len(all_synthetic)}")
    print(f"Faithful: {len(faithful)}")

    fc1 = Counter(a["judge_c1"] for a in all_synthetic)
    fc2 = Counter(a["judge_c2"] for a in all_synthetic)
    fc3 = Counter(a["judge_c3"] for a in all_synthetic)
    print(f"C1 dist: {dict(sorted(fc1.items()))}")
    print(f"C2 dist: {dict(sorted(fc2.items()))}")
    print(f"C3 dist: {dict(sorted(fc3.items()))}")

    output = {
        "real_articles": real_articles,
        "synthetic_articles": all_synthetic,
        "stats": {
            "total_real": len(real_articles),
            "total_synthetic": len(all_synthetic),
            "faithful": len(faithful),
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
