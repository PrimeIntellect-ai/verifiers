"""Bias Detector — GAN-like adversarial bias scoring.

Two agents with opposing objectives:
- Generator: Given topic + target bias scores + few-shot examples, writes a
  realistic news article. Rewarded when discriminator mispredicts the scores.
- Discriminator: Reads the generated article, predicts C1/C2/C3 bias scores.
  Rewarded for accuracy.

Data: 49 labeled articles from /home/ubuntu/research/bias/articles.json
Criteria: C1 Frame (-3 to +3), C2 Temperature (-3 to +3), C3 Evidence (-3 to +3)
"""

from __future__ import annotations

import json as _json
import logging
import os
import random
import re as _re
from pathlib import Path
from typing import Any

import httpx

from datasets import Dataset

import verifiers as vf
from verifiers.types import AssistantMessage, Messages, State, SystemMessage, UserMessage
from verifiers.utils.message_utils import content_to_text

GENERATOR = "generator"
DISCRIMINATOR = "discriminator"
AGENT_ORDER = [GENERATOR, DISCRIMINATOR]
PHASES = ["generate", "discriminate"]

ARTICLES_PATH = Path(__file__).parent / "articles.json"
# Fallback to research directory
ARTICLES_PATH_ALT = Path("/home/ubuntu/research/bias/articles.json")

# Module-level cache to avoid storing articles in every dataset row
_ARTICLES_CACHE: list[dict] | None = None

# Per-example error history for adaptive advantage (V60)
_ERROR_HISTORY: dict[int, list[float]] = {}

# V71: Self-distillation — store model's own prior predictions per article
_PREDICTION_HISTORY: dict[int, dict] = {}  # source_id -> {"c1": int, "c2": int, "c3": int}

# V78: Anti-regression — track best-ever error per article
_BEST_ERROR_HISTORY: dict[int, float] = {}  # source_id -> best (lowest) MAE ever seen

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

GENERATOR_PROMPT = """You are a news article generator for bias research.

Your task: Given a TOPIC and TARGET BIAS SCORES, write a realistic news article
that embodies those specific bias characteristics.

{criteria}

Study the example articles below carefully — they show what different score
combinations look like in practice.

{examples}

YOUR ASSIGNMENT:
Topic: {topic}
Target scores: C1={c1}, C2={c2}, C3={c3}

Write a realistic 300-600 word news article on this topic that matches these
target bias scores. Think about how the scores should manifest in your writing:
- C1 affects which causes/remedies you emphasize
- C2 affects your word choices and tone
- C3 affects how many/what kinds of sources you cite

Submit your article in the handoff block."""

DISCRIMINATOR_PROMPT = """You are a media bias analyst.

Your task: Read the article below and predict its bias scores on three criteria.

{criteria}

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

{examples}

{topic_hint}ARTICLE TO ANALYZE:
---
{article}
---

Analyze this article and predict integer scores for C1, C2, and C3 (each -3 to +3).
Submit your predictions in the handoff block."""

DISCRIMINATOR_PROMPT_CALIBRATED = """You are a media bias analyst. You must give precise, calibrated scores.

{criteria}

CALIBRATION GUIDANCE:
- Scores of 0 mean TRULY neutral/balanced. Most articles are NOT zero on all criteria.
- Extreme scores (-3, +3) are rare but real. Don't shy away from them when warranted.
- C1: Look at HOW the article explains causes. Government/system blame = negative.
  Personal responsibility / market solutions = positive.
- C2: Most news articles have SOME rhetorical heat — true zero is rare. Read the
  adjectives and framing. Loaded words like 'crisis', 'devastating' = negative.
- C3: Count evidence types explicitly. 1 type = -2 or -3. 4+ types = +2 or +3.
  Types: statistics, expert quotes, government data, anecdotes, counter-arguments, polls.

Study the example articles below carefully — they show what different score
combinations look like in practice.

{examples}

{topic_hint}ARTICLE TO ANALYZE:
---
{article}
---

Predict integer scores for C1, C2, and C3 (each -3 to +3).
Submit your predictions in the handoff block."""

DISCRIMINATOR_PROMPT_ANTIPRIOR = """You are a media bias analyst. You must give precise, calibrated scores.

{criteria}

CRITICAL RULES — READ CAREFULLY:
1. TOPIC ≠ BIAS. A politically charged topic (immigration, climate, healthcare) can be
   reported with ZERO bias. Score the WRITING STYLE, not the subject matter.
2. QUOTES ≠ ARTICLE VOICE. If the article neutrally reports inflammatory quotes from
   officials, that does NOT make the article's C2 negative. Score the journalist's
   framing, not the people being quoted.
3. SOURCE ≠ SCORE. Ignore the publication name. A Vox article CAN have positive C1.
   A Fox article CAN be neutral. Score only what's on the page.
4. Wire-service style reporting (factual, presents multiple sides) = C1=0, C2=0.

CALIBRATION:
- Scores of 0 mean TRULY neutral/balanced. Most articles are NOT zero on all criteria.
- C1: Score HOW causation is explained, not the topic itself.
- C2: Score the JOURNALIST'S own language only. Quotes from officials don't count.
- C3: Count evidence types explicitly. 1 type = -2 or -3. 4+ types = +2 or +3.

Study the example articles below — they show what different scores look like.

{examples}

{topic_hint}ARTICLE TO ANALYZE:
---
{article}
---

Score the WRITING STYLE, not the topic. Predict C1, C2, C3 (each -3 to +3).
Submit your predictions in the handoff block."""

DISCRIMINATOR_PROMPT_SELFCORRECT = """You are a media bias analyst. You must give precise, calibrated scores.

{criteria}

Study the example articles below — they show what different scores look like.

{examples}

{topic_hint}ARTICLE TO ANALYZE:
---
{article}
---

SCORING PROCESS — follow these steps:
1. Make your initial assessment of C1, C2, C3 (each -3 to +3).
2. SELF-CHECK: For any score of -3 or +3, ask yourself:
   - C1: Am I scoring the TOPIC or HOW the article explains causation?
   - C2: Am I scoring QUOTED speech or the JOURNALIST'S own language?
   - C3: Did I actually COUNT distinct evidence types?
3. If your extreme score is about the topic/quotes rather than the writing, adjust it.

CALIBRATION:
- 0 = truly neutral. Score the writing, not the subject matter.
- C2: Quotes from officials ≠ article's rhetorical heat.
- C3: Count types explicitly. 1 type = -2/-3. 4+ types = +2/+3.

Submit your final predictions in the handoff block."""

DISCRIMINATOR_PROMPT_CAUSAL = """You are a media bias analyst. You must give precise, calibrated scores.

{criteria}

ANALYSIS PROTOCOL — Follow these steps in order:

STEP 1 — CAUSAL LOGIC (for C1):
Identify the article's causal claims. Complete these sentences:
- "This article says the CAUSE of the problem is: ___"
- "This article implies the SOLUTION is: ___"
Then score: systemic/structural cause + collective solution = C1 negative.
Individual agency + market/personal solution = C1 positive.
Pure event reporting with no causal explanation = C1=0.

STEP 2 — VOICE DECOMPOSITION (for C2):
List 2-3 sentences written by the JOURNALIST (not quoted speech).
Score ONLY those sentences. Loaded adjectives = C2 negative. Hedged/clinical = C2 positive.

STEP 3 — EVIDENCE ENUMERATION (for C3):
List every distinct evidence type: statistics, expert quotes, government data,
anecdotes, legal documents, counter-arguments, polls, stock data. Count them.
1 type = C3 negative. 4+ types = C3 positive.

Study the example articles below.

{examples}

{topic_hint}ARTICLE TO ANALYZE:
---
{article}
---

Follow the 3-step protocol above, then submit your C1, C2, C3 predictions in the handoff block."""

DISCRIMINATOR_PROMPT_CONTRASTIVE = """You are a media bias analyst. You will see TWO articles and must score the SECOND one.

{criteria}

REFERENCE ARTICLE (scores provided for calibration):
Source: {ref_source} | Topic: {ref_topic}
Scores: C1={ref_c1}, C2={ref_c2}, C3={ref_c3}
---
{ref_article}
---

{topic_hint}NOW SCORE THIS ARTICLE (compare against the reference above):
---
{article}
---

Use the reference article's known scores as your calibration anchor.
If this article is more heated than the reference, its C2 should be more negative.
If it has more diverse sourcing, its C3 should be more positive.
Score the WRITING STYLE, not the topic.

Submit your predictions in the handoff block."""


DISCRIMINATOR_PROMPT_COMPARATIVE = """You are a media bias analyst. You will compare TWO articles and score the SECOND one.

{criteria}

COMPARATIVE SCORING — relative judgments are more reliable than absolute ones.
First study ARTICLE A whose scores are already known, then score ARTICLE B.

=== ARTICLE A (anchor — scores known) ===
Source: {anchor_source} | Topic: {anchor_topic}
Known scores: C1={anchor_c1}, C2={anchor_c2}, C3={anchor_c3}
---
{anchor_article}
---

=== ARTICLE B (your target — predict its scores) ===
{topic_hint}---
{article}
---

COMPARISON PROCESS:
1. For each criterion, decide: is Article B HIGHER, LOWER, or SIMILAR to Article A?
2. Estimate the MAGNITUDE of the difference (0, 1, 2, or more points).
3. Add/subtract from Article A's known score to get Article B's score.
4. Clamp final scores to the -3 to +3 range.

For example, if Article A has C2=-1 and Article B uses more loaded language,
Article B's C2 might be about 2 points lower → C2=-3.

IMPORTANT: Your final scores for Article B must be integers from -3 to +3.

Submit your predictions for Article B in the handoff block."""


def _strip_topic_nouns(text: str, topic: str) -> str:
    """Replace topic-identifying proper nouns with generic placeholders."""
    # Common topic-specific noun groups
    _topic_subs = {
        "Immigration": [
            (r'\b(ICE|immigration|immigrant|migrant|deport\w*|asylum|undocument\w*|border)\b', '[POLICY-ENTITY]'),
            (r'\b(DHS|Homeland Security|CBP|Border Patrol)\b', '[AGENCY]'),
        ],
        "Climate/Energy": [
            (r'\b(climate|carbon|emission\w*|fossil|renewable|solar|wind|offshore)\b', '[ENERGY-TOPIC]'),
            (r'\b(EPA|DOE|Energy Department)\b', '[AGENCY]'),
        ],
        "Healthcare": [
            (r'\b(gender.affirming|transgender|trans\b|HHS|FDA)\b', '[HEALTH-TOPIC]'),
            (r'\b(Medicaid|Medicare|ACA|Obamacare)\b', '[PROGRAM]'),
        ],
        "Crime/Policing": [
            (r'\b(polic\w+|crime|violen\w+|gun|shoot\w*|murder|homicide)\b', '[SAFETY-TOPIC]'),
        ],
    }
    result = text
    for pattern, replacement in _topic_subs.get(topic, []):
        result = _re.sub(pattern, replacement, result, flags=_re.IGNORECASE)
    return result


_SOURCE_NAMES = [
    "New York Times", "NYT", "NY Times", "Wall Street Journal", "WSJ",
    "Washington Post", "The Atlantic", "Vox", "Fox News", "CNN", "MSNBC",
    "NBC News", "ABC News", "CBS News", "NPR", "BBC", "Reuters", "AP",
    "Associated Press", "The Guardian", "National Review", "Breitbart",
    "HuffPost", "Huffington Post", "Politico", "The Hill", "Slate",
    "The Intercept", "Daily Wire", "Jacobin", "Mother Jones",
    "The Economist", "Bloomberg", "USA Today", "Los Angeles Times",
    "Chicago Tribune", "Boston Globe", "ProPublica",
]
_SOURCE_PATTERN = _re.compile(
    r'\b(' + '|'.join(_re.escape(s) for s in sorted(_SOURCE_NAMES, key=len, reverse=True)) + r')\b',
    _re.IGNORECASE,
)


def _strip_source_names(text: str) -> str:
    """Replace publication/source names with a generic placeholder."""
    return _SOURCE_PATTERN.sub('[NEWS SOURCE]', text)


# V62: Article perturbation patterns for differential scoring
_LOADED_WORDS = [
    "crisis", "devastating", "alarming", "shocking", "outrageous", "catastrophic",
    "disastrous", "rampant", "skyrocketing", "plummeting", "unprecedented",
    "controversial", "radical", "extreme", "reckless", "dangerous",
]
_NEUTRAL_WORDS = [
    "situation", "significant", "notable", "unexpected", "substantial", "considerable",
    "notable", "widespread", "increasing", "decreasing", "unusual",
    "debated", "proposed", "notable", "bold", "significant",
]


def _perturb_article(text: str, perturbation_type: str) -> tuple[str, dict[str, int]]:
    """Apply a structured perturbation to an article. Returns (modified_text, expected_delta)."""
    if perturbation_type == "c2_neutralize":
        modified = text
        count = 0
        for loaded, neutral in zip(_LOADED_WORDS, _NEUTRAL_WORDS):
            pattern = _re.compile(r'\b' + _re.escape(loaded) + r'\b', _re.IGNORECASE)
            new_text = pattern.sub(neutral, modified)
            if new_text != modified:
                count += 1
                modified = new_text
        # Neutralizing loaded words should push C2 toward positive (more measured)
        delta = min(3, max(1, count))  # 1-3 point shift
        return modified, {"c1": 0, "c2": delta, "c3": 0}

    elif perturbation_type == "c3_strip_sources":
        # Remove quoted expert sources to reduce evidence diversity
        modified = _re.sub(r'"[^"]{30,}"', '[REDACTED]', text)
        modified = _re.sub(r'according to [^,\.]{5,50}[,\.]', 'reportedly,', modified, flags=_re.IGNORECASE)
        modified = _re.sub(r'said [A-Z][a-z]+ [A-Z][a-z]+', 'said a spokesperson', modified)
        return modified, {"c1": 0, "c2": 0, "c3": -2}  # Less evidence diversity

    elif perturbation_type == "c1_reframe":
        # Shift causal framing from structural to individual
        structural = [
            (r'systemic\b', 'individual'),
            (r'institutional\b', 'personal'),
            (r'structural\b', 'individual'),
            (r'inequality\b', 'choices'),
            (r'collective action\b', 'personal responsibility'),
        ]
        modified = text
        count = 0
        for pattern, replacement in structural:
            new_text = _re.sub(pattern, replacement, modified, flags=_re.IGNORECASE)
            if new_text != modified:
                count += 1
                modified = new_text
        delta = min(2, max(1, count))
        return modified, {"c1": delta, "c2": 0, "c3": 0}  # Shift toward agency frame

    return text, {"c1": 0, "c2": 0, "c3": 0}


def _load_articles() -> list[dict]:
    """Load and filter usable articles (have all 3 rater A scores + text)."""
    global _ARTICLES_CACHE
    if _ARTICLES_CACHE is not None:
        return _ARTICLES_CACHE

    path = ARTICLES_PATH if ARTICLES_PATH.exists() else ARTICLES_PATH_ALT
    with open(path) as f:
        raw = _json.load(f)

    usable = []
    for a in raw:
        try:
            c1 = int(a.get("c1_score_a", ""))
            c2 = int(a.get("c2_score_a", ""))
            c3 = int(a.get("c3_score_a", ""))
            text = a.get("article_text", "")
            if text and len(text) > 100:
                usable.append({
                    "id": a["id"],
                    "topic": str(a.get("topic", "General")).strip(),
                    "headline": str(a.get("headline", "")),
                    "source": str(a.get("source", "")),
                    "c1": c1, "c2": c2, "c3": c3,
                    "text": text,
                })
        except (ValueError, TypeError):
            pass

    _ARTICLES_CACHE = usable
    return usable


def _select_fewshot(articles: list[dict], target_c1: int, target_c2: int,
                    target_c3: int, n: int = 3, exclude_id: int = -1) -> list[dict]:
    """Select n articles closest to target scores by Manhattan distance."""
    scored = []
    for a in articles:
        if a["id"] == exclude_id:
            continue
        dist = abs(a["c1"] - target_c1) + abs(a["c2"] - target_c2) + abs(a["c3"] - target_c3)
        scored.append((dist, a))
    scored.sort(key=lambda x: x[0])
    # Take top n, but add slight randomness: pick from top n+2 to avoid always same examples
    pool = scored[:n + 2]
    random.shuffle(pool)
    return [x[1] for x in pool[:n]]


def _select_diverse_fewshot(articles: list[dict], target_c1: int, target_c2: int,
                            target_c3: int, n: int = 5, exclude_id: int = -1) -> list[dict]:
    """Select n examples: closest + spread across score range for calibration."""
    available = [a for a in articles if a["id"] != exclude_id]
    if len(available) <= n:
        return available

    # First pick 2 closest (anchors)
    scored = [(abs(a["c1"]-target_c1)+abs(a["c2"]-target_c2)+abs(a["c3"]-target_c3), a)
              for a in available]
    scored.sort(key=lambda x: x[0])
    selected = [scored[0][1], scored[1][1]]
    selected_ids = {selected[0]["id"], selected[1]["id"]}

    # Then pick remaining from diverse score regions
    remaining = [a for a in available if a["id"] not in selected_ids]
    # Sort by distance from target DESCENDING to get contrasting examples
    remaining.sort(key=lambda a: -(abs(a["c1"]-target_c1)+abs(a["c2"]-target_c2)+abs(a["c3"]-target_c3)))
    # Pick 1 far example (calibration anchor at opposite end)
    if remaining:
        selected.append(remaining[0])
        selected_ids.add(remaining[0]["id"])
    # Fill rest from mid-distance (moderate examples)
    mid = [a for a in available if a["id"] not in selected_ids]
    mid.sort(key=lambda a: abs(abs(a["c1"]-target_c1)+abs(a["c2"]-target_c2)+abs(a["c3"]-target_c3) - 6))
    for a in mid[:n - len(selected)]:
        selected.append(a)
    random.shuffle(selected)
    return selected[:n]


_log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading a media bias analyst's explanation. The analyst was given a news article and asked to predict bias scores on 3 criteria (C1 Frame, C2 Temperature, C3 Evidence, each -3 to +3).

CRITERIA:
{criteria}

ARTICLE (excerpt):
{article}

TARGET SCORES (ground truth): C1={true_c1}, C2={true_c2}, C3={true_c3}
ANALYST'S PREDICTED SCORES: C1={pred_c1}, C2={pred_c2}, C3={pred_c3}
ANALYST'S REASONING:
{reasoning}

Grade the analyst's reasoning on a scale of 0-10:
- Does the reasoning correctly identify WHY the article has its bias characteristics?
- Does it cite specific evidence from the article (quotes, named sources, rhetorical choices)?
- Does it address all 3 criteria with distinct, relevant observations?
- Is the reasoning consistent with the predicted scores?
- Would this explanation help a human understand the article's biases?

Respond with ONLY a JSON object: {{"score": N}} where N is 0-10."""

_JUDGE_CLIENT: httpx.AsyncClient | None = None


def _get_judge_client() -> httpx.AsyncClient:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = httpx.AsyncClient(timeout=60.0)
    return _JUDGE_CLIENT


async def _llm_judge_reasoning(
    article: str,
    reasoning: str,
    true_c1: int, true_c2: int, true_c3: int,
    pred_c1: int, pred_c2: int, pred_c3: int,
    base_url: str = "http://localhost:8000/v1",
    model: str = "",
) -> float:
    """Call the vLLM server to judge disc reasoning quality. Returns 0.0-1.0."""
    if not reasoning or reasoning == "brief explanation":
        return 0.0

    prompt = JUDGE_PROMPT.format(
        criteria=CRITERIA_DESC,
        article=article[:3000],
        true_c1=true_c1, true_c2=true_c2, true_c3=true_c3,
        pred_c1=pred_c1, pred_c2=pred_c2, pred_c3=pred_c3,
        reasoning=reasoning,
    )

    client = _get_judge_client()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    try:
        resp = await client.post(f"{base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.warning("LLM judge call failed: %s — falling back to heuristic", e)
        return _score_reasoning_quality(reasoning)

    match = _re.search(r'"score"\s*:\s*(\d+)', text)
    if match:
        return min(1.0, int(match.group(1)) / 10.0)

    match = _re.search(r'\b(\d+)\b', text)
    if match:
        return min(1.0, int(match.group(1)) / 10.0)

    return 0.5


def _score_reasoning_quality(reasoning: str) -> float:
    """Fast heuristic fallback for reasoning quality (0.0-1.0)."""
    if not reasoning or reasoning == "brief explanation":
        return 0.0

    r_lower = reasoning.lower()
    score = 0.0
    c1_terms = ["c1", "frame", "causal", "agency", "structural", "systemic", "individual", "market", "collective"]
    c2_terms = ["c2", "temperature", "rhetorical", "tone", "alarmist", "measured", "clinical", "loaded", "emotional"]
    c3_terms = ["c3", "evidence", "source", "diversity", "counter", "expert", "statistic", "quote", "anecdot"]
    score += 0.33 * (any(t in r_lower for t in c1_terms) + any(t in r_lower for t in c2_terms) + any(t in r_lower for t in c3_terms))
    if len(reasoning) > 50:
        score = min(1.0, score + 0.1)
    return min(1.0, score)


def _format_examples(articles: list[dict], max_text_len: int = 1500) -> str:
    """Format few-shot articles for prompt."""
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


class BiasDetectorEnv(vf.MultiAgentEnv):
    AGENT_ORDER = [GENERATOR, DISCRIMINATOR]

    def __init__(self, disc_blind: bool = False, disc_num_examples: int = 3,
                 disc_diverse_examples: bool = False, disc_topic_hint: bool = False,
                 disc_calibrated: bool = False, disc_antiprior: bool = False,
                 disc_selfcorrect: bool = False, disc_mixed_prompt: bool = False,
                 disc_mask_quotes: float = 0.0,
                 disc_contrastive: float = 0.0,
                 disc_comparative: float = 0.0,
                 disc_topic_neutral: float = 0.0,
                 disc_iterative: bool = False,
                 disc_source_blind: float = 0.0,
                 disc_topic_deconfound: bool = False,
                 disc_double_iterative: bool = False,
                 disc_causal_prompt: bool = False,
                 disc_magnitude_feedback: bool = False,
                 disc_confidence_weight: bool = False,
                 disc_hindsight_explain: bool = False,
                 disc_decomposed_iterative: bool = False,
                 disc_sequential_scoring: bool = False,
                 disc_adaptive_advantage: bool = False,
                 disc_relative_reward: bool = False,
                 disc_perturbation: bool = False,
                 disc_sub_features: bool = False,
                 disc_progressive_reveal: bool = False,
                 disc_self_distill: bool = False,
                 disc_socratic: bool = False,
                 disc_anti_regression: bool = False,
                 disc_anchor_scoring: bool = False,
                 disc_majority_vote: bool = False,
                 disc_ensemble_distill: bool = False,
                 disc_length_stratify: bool = False,
                 disc_blind_selfcorrect: bool = False,
                 disc_devils_advocate: bool = False,
                 disc_first_pass_bonus: float = 0.0,
                 disc_oracle_withdrawal: float = 0.0,
                 disc_evidence_first: bool = False,
                 disc_pairwise_rank: bool = False,
                 disc_consistency_training: bool = False,
                 use_real_articles: bool = False, train_articles: list[dict] | None = None,
                 **kwargs):
        self.disc_blind = disc_blind
        self.disc_num_examples = disc_num_examples
        self.disc_diverse_examples = disc_diverse_examples
        self.disc_topic_hint = disc_topic_hint
        self.disc_calibrated = disc_calibrated
        self.disc_antiprior = disc_antiprior
        self.disc_selfcorrect = disc_selfcorrect
        self.disc_mixed_prompt = disc_mixed_prompt
        self.disc_mask_quotes = disc_mask_quotes
        self.disc_contrastive = disc_contrastive
        self.disc_comparative = disc_comparative
        self.disc_topic_neutral = disc_topic_neutral
        self.disc_iterative = disc_iterative
        self.disc_source_blind = disc_source_blind
        self.disc_topic_deconfound = disc_topic_deconfound
        self.disc_double_iterative = disc_double_iterative
        self.disc_causal_prompt = disc_causal_prompt
        self.disc_magnitude_feedback = disc_magnitude_feedback
        self.disc_confidence_weight = disc_confidence_weight
        self.disc_hindsight_explain = disc_hindsight_explain
        self.disc_decomposed_iterative = disc_decomposed_iterative
        self.disc_sequential_scoring = disc_sequential_scoring
        self.disc_adaptive_advantage = disc_adaptive_advantage
        self.disc_relative_reward = disc_relative_reward
        self.disc_perturbation = disc_perturbation
        self.disc_sub_features = disc_sub_features
        self.disc_progressive_reveal = disc_progressive_reveal
        self.disc_self_distill = disc_self_distill
        self.disc_socratic = disc_socratic
        self.disc_anti_regression = disc_anti_regression
        self.disc_anchor_scoring = disc_anchor_scoring
        self.disc_majority_vote = disc_majority_vote
        self.disc_ensemble_distill = disc_ensemble_distill
        self.disc_length_stratify = disc_length_stratify
        self.disc_blind_selfcorrect = disc_blind_selfcorrect
        self.disc_devils_advocate = disc_devils_advocate
        self.disc_first_pass_bonus = disc_first_pass_bonus
        self.disc_oracle_withdrawal = disc_oracle_withdrawal
        self.disc_evidence_first = disc_evidence_first
        self.disc_pairwise_rank = disc_pairwise_rank
        self.disc_consistency_training = disc_consistency_training
        self.use_real_articles = use_real_articles
        self.train_articles = train_articles or []
        max_t = 6
        if disc_hindsight_explain:
            max_t = 8
        if disc_decomposed_iterative:
            max_t = 10
        if disc_sequential_scoring:
            max_t = 10
        if disc_sub_features:
            max_t = 8  # initial sub-features + composition + optional iterative
        if disc_progressive_reveal:
            max_t = 12  # headline + para + full + reveal + optional iterative
        if disc_socratic:
            max_t = 10  # gen + questions + answers + scoring (+ optional iterative)
        if disc_majority_vote:
            max_t = 10  # gen + 3 vote passes + optional iterative
        if disc_ensemble_distill:
            max_t = 8  # gen + first score + expert hints + revision (+ optional iterative)
        if disc_evidence_first:
            max_t = 10  # gen + evidence extraction + scoring + optional devil/blind/iterative
        if disc_consistency_training:
            max_t = 10  # gen + first score + second score + optional self-check
        super().__init__(tools=[], max_turns=max_t, **kwargs)

    def get_all_actors(self, state: State) -> dict[str, str]:
        # System prompts are set dynamically per-rollout in get_prompt_for_actor
        return {
            GENERATOR: "You are a news article generator.",
            DISCRIMINATOR: "You are a media bias analyst.",
        }

    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        return GENERATOR

    def get_next_actor_id(self, state: State) -> str:
        phase = state.get("phase", PHASES[0])
        if phase == "generate":
            return GENERATOR
        return DISCRIMINATOR

    def get_handoff_schema(self, actor_id: str, state: State) -> dict[str, Any]:
        if actor_id == GENERATOR:
            if self.use_real_articles:
                return {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                    },
                    "required": ["status"],
                    "additionalProperties": True,
                }
            return {
                "type": "object",
                "properties": {
                    "article": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["article"],
                "additionalProperties": False,
            }
        else:
            # Hindsight explanation phase: just needs explanation text
            if self.disc_hindsight_explain and state.get("disc_hindsight_phase"):
                return {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string"},
                    },
                    "required": ["explanation"],
                    "additionalProperties": True,
                }
            # Sequential scoring: single-criterion schemas
            seq_phase = state.get("disc_sequential_phase")
            if self.disc_sequential_scoring and seq_phase in ("c1", "c2", "c3"):
                key = f"{seq_phase}_score"
                return {
                    "type": "object",
                    "properties": {
                        key: {"type": "integer", "minimum": -3, "maximum": 3},
                        "reasoning": {"type": "string"},
                    },
                    "required": [key],
                    "additionalProperties": True,
                }
            # Decomposed iterative: single-criterion correction schema
            decomp_criterion = state.get("disc_decomposed_criterion")
            if self.disc_decomposed_iterative and decomp_criterion in ("c1", "c2", "c3"):
                key = f"{decomp_criterion}_score"
                return {
                    "type": "object",
                    "properties": {
                        key: {"type": "integer", "minimum": -3, "maximum": 3},
                        "reasoning": {"type": "string"},
                    },
                    "required": [key],
                    "additionalProperties": True,
                }
            # Socratic probing (V74): phase-dependent schemas
            socratic_phase = state.get("disc_socratic_phase")
            if self.disc_socratic and socratic_phase == "questions":
                return {
                    "type": "object",
                    "properties": {
                        "c1_question": {"type": "string"},
                        "c2_question": {"type": "string"},
                        "c3_question": {"type": "string"},
                    },
                    "required": ["c1_question", "c2_question", "c3_question"],
                    "additionalProperties": True,
                }
            if self.disc_socratic and socratic_phase == "answers":
                return {
                    "type": "object",
                    "properties": {
                        "c1_answer": {"type": "string"},
                        "c2_answer": {"type": "string"},
                        "c3_answer": {"type": "string"},
                    },
                    "required": ["c1_answer", "c2_answer", "c3_answer"],
                    "additionalProperties": True,
                }
            # Sub-features (V69): phase-dependent schemas
            sub_phase = state.get("disc_sub_feature_phase")
            if self.disc_sub_features and sub_phase == "sub_scores":
                return {
                    "type": "object",
                    "properties": {
                        # C1 sub-features
                        "c1_headline": {"type": "integer", "minimum": -3, "maximum": 3},
                        "c1_sources": {"type": "integer", "minimum": -3, "maximum": 3},
                        "c1_emphasis": {"type": "integer", "minimum": -3, "maximum": 3},
                        # C2 sub-features
                        "c2_emotional": {"type": "integer", "minimum": -3, "maximum": 3},
                        "c2_certainty": {"type": "integer", "minimum": -3, "maximum": 3},
                        # C3 sub-features
                        "c3_sourcing": {"type": "integer", "minimum": -3, "maximum": 3},
                        "c3_data": {"type": "integer", "minimum": -3, "maximum": 3},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["c1_headline", "c1_sources", "c1_emphasis",
                                 "c2_emotional", "c2_certainty",
                                 "c3_sourcing", "c3_data"],
                    "additionalProperties": True,
                }
            props = {
                "c1_score": {"type": "integer", "minimum": -3, "maximum": 3},
                "c2_score": {"type": "integer", "minimum": -3, "maximum": 3},
                "c3_score": {"type": "integer", "minimum": -3, "maximum": 3},
                "reasoning": {"type": "string"},
            }
            required = ["c1_score", "c2_score", "c3_score"]
            if self.disc_confidence_weight:
                props["confidence"] = {"type": "integer", "minimum": 1, "maximum": 10}
                required.append("confidence")
            return {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            }

    def parse_handoff(
        self, actor_id: str, last_message: AssistantMessage, state: State
    ) -> dict[str, Any]:
        text = content_to_text(last_message.content).strip()
        if not text:
            raise vf.InvalidModelResponseError(f"Actor '{actor_id}' produced empty response.")

        # Try <handoff> tags first
        match = _re.search(r"<handoff>(.*?)</handoff>", text, _re.DOTALL)
        if match:
            handoff_content = match.group(1).strip()
            # Extract first JSON object from handoff (model sometimes puts text after the JSON)
            brace_start = handoff_content.find("{")
            if brace_start >= 0:
                depth = 0
                for j, c in enumerate(handoff_content[brace_start:]):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            raw = handoff_content[brace_start:brace_start + j + 1]
                            break
                else:
                    raw = handoff_content
            else:
                raw = handoff_content
        else:
            # Find last complete JSON object
            raw = None
            last_open = text.rfind("{")
            if last_open >= 0:
                for i in range(last_open, -1, -1):
                    if text[i] == "{":
                        candidate = text[i:]
                        depth = 0
                        for j, c in enumerate(candidate):
                            if c == "{":
                                depth += 1
                            elif c == "}":
                                depth -= 1
                                if depth == 0:
                                    raw = candidate[: j + 1]
                                    break
                        if raw:
                            break
            if not raw:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' must submit in a <handoff>{{...}}</handoff> block."
                )

        try:
            payload = _json.loads(raw)
        except _json.JSONDecodeError as e:
            raise vf.InvalidModelResponseError(f"Actor '{actor_id}' produced invalid JSON: {e}") from e

        if not isinstance(payload, dict):
            raise vf.InvalidModelResponseError(f"Actor '{actor_id}' handoff must be a JSON object.")

        # Coerce discriminator scores to int
        if actor_id == DISCRIMINATOR:
            for key in ("c1_score", "c2_score", "c3_score"):
                if key in payload:
                    try:
                        payload[key] = max(-3, min(3, int(float(payload[key]))))
                    except (ValueError, TypeError):
                        payload[key] = 0

        schema = state["handoff_schemas"][actor_id]
        self.validate_handoff(payload, schema, actor_id)
        return payload

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["phase"] = "generate"
        state["generated_article"] = ""
        state["pred_scores"] = {"c1": 0, "c2": 0, "c3": 0}
        return state

    async def apply_handoff(
        self, actor_id: str, handoff: dict[str, Any], state: State
    ) -> str | None:
        phase = state["phase"]

        if phase == "generate":
            if self.use_real_articles:
                pool = self.train_articles if self.train_articles else _load_articles()
                source_id = state.get("info", {}).get("source_id", 0)
                match = next((a for a in pool if a["id"] == source_id), pool[0])
                article_text = match["text"]

                # V62: Apply perturbation with 50% probability
                if self.disc_perturbation and random.random() < 0.5:
                    ptype = random.choice(["c2_neutralize", "c3_strip_sources", "c1_reframe"])
                    modified, delta = _perturb_article(article_text, ptype)
                    if modified != article_text:
                        info = state.get("info", {})
                        new_c1 = max(-3, min(3, info.get("c1", 0) + delta["c1"]))
                        new_c2 = max(-3, min(3, info.get("c2", 0) + delta["c2"]))
                        new_c3 = max(-3, min(3, info.get("c3", 0) + delta["c3"]))
                        state["info"] = {**info, "c1": new_c1, "c2": new_c2, "c3": new_c3}
                        state["disc_perturbed"] = True
                        state["disc_perturbation_type"] = ptype
                        article_text = modified

                state["generated_article"] = article_text
            else:
                article = str(handoff.get("article", ""))
                state["generated_article"] = article
                if len(article) < 50:
                    state["final_env_response"] = (
                        f"Generator produced article too short ({len(article)} chars). Game aborted."
                    )
                    return state["final_env_response"]

            state["phase"] = "discriminate"
            return f"Article ready ({len(state['generated_article'])} chars). Passing to discriminator."

        if phase == "discriminate":
            # Hindsight explanation sub-phase
            if state.get("disc_hindsight_phase"):
                explanation = str(handoff.get("explanation", ""))
                state["disc_hindsight_explanation"] = explanation
                state["hindsight_reasoning_quality"] = _score_reasoning_quality(explanation)
                info = state.get("info", {})
                pred = state.get("pred_scores", {})
                error = state.get("disc_error", 0)
                state["final_env_response"] = (
                    f"Game complete (with hindsight explanation)!\n"
                    f"  Target scores: C1={info.get('c1',0)}, C2={info.get('c2',0)}, C3={info.get('c3',0)}\n"
                    f"  Predicted scores: C1={pred.get('c1',0)}, C2={pred.get('c2',0)}, C3={pred.get('c3',0)}\n"
                    f"  Total error: {error}/18\n"
                    f"  Explanation quality: {state['hindsight_reasoning_quality']:.2f}"
                )
                return state["final_env_response"]

            # V87: Majority vote — collect 3 independent scoring passes, take median
            if self.disc_majority_vote:
                votes = state.get("disc_votes", [])
                c1 = max(-3, min(3, int(handoff.get("c1_score", 0))))
                c2 = max(-3, min(3, int(handoff.get("c2_score", 0))))
                c3 = max(-3, min(3, int(handoff.get("c3_score", 0))))
                votes.append({"c1": c1, "c2": c2, "c3": c3})
                state["disc_votes"] = votes
                if len(votes) < 3:
                    return (
                        f"Scoring pass {len(votes)}/3 recorded (C1={c1}, C2={c2}, C3={c3}).\n"
                        f"Re-read the article from scratch and score independently.\n"
                        f"Ignore your previous scores — this is a fresh evaluation.\n"
                        f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "..."}}</handoff>'
                    )
                # 3 votes collected — take median per criterion
                median_c1 = sorted(v["c1"] for v in votes)[1]
                median_c2 = sorted(v["c2"] for v in votes)[1]
                median_c3 = sorted(v["c3"] for v in votes)[1]
                state["pred_scores"] = {"c1": median_c1, "c2": median_c2, "c3": median_c3}
                state["disc_reasoning"] = str(handoff.get("reasoning", ""))
                info = state.get("info", {})
                true_c1, true_c2, true_c3 = info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)
                error = abs(median_c1 - true_c1) + abs(median_c2 - true_c2) + abs(median_c3 - true_c3)
                state["disc_error"] = error
                # Store for self-distillation
                source_id = info.get("source_id", -1)
                if source_id >= 0:
                    _PREDICTION_HISTORY[source_id] = {"c1": median_c1, "c2": median_c2, "c3": median_c3}
                state["final_env_response"] = (
                    f"Game complete (majority vote)!\n"
                    f"  Votes: {[(v['c1'], v['c2'], v['c3']) for v in votes]}\n"
                    f"  Median: C1={median_c1}, C2={median_c2}, C3={median_c3}\n"
                    f"  Target: C1={true_c1}, C2={true_c2}, C3={true_c3}\n"
                    f"  Total error: {error}/18"
                )
                return state["final_env_response"]

            # Sequential scoring sub-phases (V61): single-criterion returns
            seq_phase = state.get("disc_sequential_phase")
            if self.disc_sequential_scoring and seq_phase in ("c1", "c2", "c3"):
                score = int(handoff.get(f"{seq_phase}_score", 0))
                pred = state.get("pred_scores", {"c1": 0, "c2": 0, "c3": 0})
                pred[seq_phase] = score
                state["pred_scores"] = pred
                next_phases = {"c1": "c2", "c2": "c3", "c3": None}
                next_p = next_phases[seq_phase]
                if next_p:
                    state["disc_sequential_phase"] = next_p
                    prompts = {
                        "c2": (
                            f"You scored C1={pred['c1']}. Now focus ONLY on C2 (Rhetorical Temperature).\n"
                            f"C2 scores the JOURNALIST'S language, not quoted speech. "
                            f"Loaded/alarmist = negative. Measured/clinical = positive.\n"
                            f"List 2-3 sentences written by the journalist and assess their tone.\n"
                            f'Submit: <handoff>{{"c2_score": N, "reasoning": "..."}}</handoff>'
                        ),
                        "c3": (
                            f"You scored C1={pred['c1']}, C2={pred['c2']}. Now focus ONLY on C3 (Evidence Diversity).\n"
                            f"C3 counts DISTINCT evidence types: statistics, expert quotes, government data, "
                            f"anecdotes, counter-arguments, polls, legal documents.\n"
                            f"List each type you find in the article. 1 type = negative. 4+ = positive.\n"
                            f'Submit: <handoff>{{"c3_score": N, "reasoning": "..."}}</handoff>'
                        ),
                    }
                    return prompts[next_p]
                # All 3 scored — compute error
                info = state.get("info", {})
                c1, c2, c3 = pred["c1"], pred["c2"], pred["c3"]
                true_c1, true_c2, true_c3 = info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)
                errors = [abs(c1 - true_c1), abs(c2 - true_c2), abs(c3 - true_c3)]
                error = sum(errors)
                state["disc_error"] = error
                state["disc_sequential_phase"] = "done"
                # V80: Allow iterative correction after sequential scoring
                if self.disc_iterative and max(errors) >= 2:
                    state["disc_sequential_phase"] = "done_iterating"
                    worst_idx = errors.index(max(errors))
                    criterion_names = ["C1 (Frame)", "C2 (Temperature)", "C3 (Evidence)"]
                    preds = [c1, c2, c3]
                    golds = [true_c1, true_c2, true_c3]
                    direction = "too negative" if preds[worst_idx] < golds[worst_idx] else "too positive"
                    if self.disc_magnitude_feedback:
                        magnitude = abs(preds[worst_idx] - golds[worst_idx])
                        return (
                            f"Your sequential scores: C1={c1}, C2={c2}, C3={c3}\n"
                            f"Your {criterion_names[worst_idx]} seems {direction} (off by ~{magnitude} points). "
                            f"Please reconsider and submit revised final scores.\n"
                            f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "..."}}</handoff>'
                        )
                    return (
                        f"Your sequential scores: C1={c1}, C2={c2}, C3={c3}\n"
                        f"Your {criterion_names[worst_idx]} seems {direction}. "
                        f"Please reconsider and submit revised final scores.\n"
                        f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "..."}}</handoff>'
                    )
                state["final_env_response"] = (
                    f"Game complete (sequential scoring)!\n"
                    f"  Target: C1={true_c1}, C2={true_c2}, C3={true_c3}\n"
                    f"  Predicted: C1={c1}, C2={c2}, C3={c3}\n"
                    f"  Total error: {error}/18"
                )
                return state["final_env_response"]

            # Decomposed iterative sub-phases (V57): per-criterion correction returns
            decomp_criterion = state.get("disc_decomposed_criterion")
            if self.disc_decomposed_iterative and decomp_criterion in ("c1", "c2", "c3"):
                score = int(handoff.get(f"{decomp_criterion}_score", 0))
                pred = state.get("pred_scores", {"c1": 0, "c2": 0, "c3": 0})
                pred[decomp_criterion] = score
                state["pred_scores"] = pred
                # Move to next criterion that needs correction
                decomp_queue = state.get("disc_decomposed_queue", [])
                if decomp_queue:
                    next_crit = decomp_queue.pop(0)
                    state["disc_decomposed_queue"] = decomp_queue
                    state["disc_decomposed_criterion"] = next_crit
                    return self._decomposed_correction_prompt(next_crit, pred, state)
                # All corrections done — compute final error
                state["disc_decomposed_criterion"] = None
                info = state.get("info", {})
                c1, c2, c3 = pred["c1"], pred["c2"], pred["c3"]
                true_c1, true_c2, true_c3 = info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)
                error = abs(c1 - true_c1) + abs(c2 - true_c2) + abs(c3 - true_c3)
                state["disc_error"] = error
                state["final_env_response"] = (
                    f"Game complete (decomposed iterative)!\n"
                    f"  Target: C1={true_c1}, C2={true_c2}, C3={true_c3}\n"
                    f"  Predicted: C1={c1}, C2={c2}, C3={c3}\n"
                    f"  Total error: {error}/18"
                )
                return state["final_env_response"]

            # Sub-features composition (V69): process sub-score handoff
            sub_phase = state.get("disc_sub_feature_phase")
            if self.disc_sub_features and sub_phase == "sub_scores":
                # Extract sub-features and compose into final scores
                c1_headline = max(-3, min(3, int(handoff.get("c1_headline", 0))))
                c1_sources = max(-3, min(3, int(handoff.get("c1_sources", 0))))
                c1_emphasis = max(-3, min(3, int(handoff.get("c1_emphasis", 0))))
                c2_emotional = max(-3, min(3, int(handoff.get("c2_emotional", 0))))
                c2_certainty = max(-3, min(3, int(handoff.get("c2_certainty", 0))))
                c3_sourcing = max(-3, min(3, int(handoff.get("c3_sourcing", 0))))
                c3_data = max(-3, min(3, int(handoff.get("c3_data", 0))))
                # Compose: simple mean, clamped to integer range
                c1 = max(-3, min(3, round((c1_headline + c1_sources + c1_emphasis) / 3)))
                c2 = max(-3, min(3, round((c2_emotional + c2_certainty) / 2)))
                c3 = max(-3, min(3, round((c3_sourcing + c3_data) / 2)))
                state["disc_sub_feature_phase"] = "done"
                state["disc_sub_scores"] = {
                    "c1_headline": c1_headline, "c1_sources": c1_sources, "c1_emphasis": c1_emphasis,
                    "c2_emotional": c2_emotional, "c2_certainty": c2_certainty,
                    "c3_sourcing": c3_sourcing, "c3_data": c3_data,
                }
                state["pred_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                info = state.get("info", {})
                true_c1, true_c2, true_c3 = info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)
                errors = [abs(c1 - true_c1), abs(c2 - true_c2), abs(c3 - true_c3)]
                error = sum(errors)
                state["disc_error"] = error
                # Allow iterative correction on composed scores if enabled
                if self.disc_iterative and max(errors) >= 2:
                    state["disc_sub_feature_phase"] = "done_iterating"
                    worst_idx = errors.index(max(errors))
                    criterion_names = ["C1 (Frame)", "C2 (Temperature)", "C3 (Evidence)"]
                    preds = [c1, c2, c3]
                    golds = [true_c1, true_c2, true_c3]
                    direction = "too negative" if preds[worst_idx] < golds[worst_idx] else "too positive"
                    return (
                        f"Your composed scores: C1={c1}, C2={c2}, C3={c3}\n"
                        f"Your {criterion_names[worst_idx]} seems {direction}. "
                        f"Please reconsider and submit revised final scores."
                    )
                state["final_env_response"] = (
                    f"Game complete (sub-feature decomposition)!\n"
                    f"  Target: C1={true_c1}, C2={true_c2}, C3={true_c3}\n"
                    f"  Composed: C1={c1}, C2={c2}, C3={c3}\n"
                    f"  Total error: {error}/18"
                )
                return state["final_env_response"]

            # Socratic probing (V74): question and answer phases
            socratic_phase = state.get("disc_socratic_phase")
            if self.disc_socratic and socratic_phase == "questions":
                # Store questions, move to answer phase
                state["disc_socratic_questions"] = {
                    "c1": str(handoff.get("c1_question", "")),
                    "c2": str(handoff.get("c2_question", "")),
                    "c3": str(handoff.get("c3_question", "")),
                }
                state["disc_socratic_phase"] = "answers"
                qs = state["disc_socratic_questions"]
                return (
                    f"Now answer your own questions by citing SPECIFIC text from the article.\n\n"
                    f"C1 Question: {qs['c1']}\n"
                    f"C2 Question: {qs['c2']}\n"
                    f"C3 Question: {qs['c3']}\n\n"
                    f"For each answer, quote the relevant sentence(s) from the article.\n"
                    f'Submit: <handoff>{{"c1_answer": "...", "c2_answer": "...", "c3_answer": "..."}}</handoff>'
                )
            if self.disc_socratic and socratic_phase == "answers":
                # Store answers, move to scoring phase
                state["disc_socratic_answers"] = {
                    "c1": str(handoff.get("c1_answer", "")),
                    "c2": str(handoff.get("c2_answer", "")),
                    "c3": str(handoff.get("c3_answer", "")),
                }
                # Score answer quality: check for quotation marks (citing specific text)
                answers = state["disc_socratic_answers"]
                cite_count = sum(1 for a in answers.values() if '"' in a or "'" in a)
                state["disc_socratic_answer_quality"] = cite_count / 3.0  # 0.0 to 1.0
                state["disc_socratic_phase"] = "scoring"
                qs = state["disc_socratic_questions"]
                return (
                    f"Based on your analysis:\n"
                    f"  C1: {qs['c1']} → {answers['c1'][:200]}\n"
                    f"  C2: {qs['c2']} → {answers['c2'][:200]}\n"
                    f"  C3: {qs['c3']} → {answers['c3'][:200]}\n\n"
                    f"Now predict the bias scores. Your answers should constrain your scoring.\n"
                    f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "..."}}</handoff>'
                )

            # Progressive reveal (V70): handle stage-specific returns
            reveal_stage = state.get("disc_reveal_stage")
            if self.disc_progressive_reveal and reveal_stage in ("headline", "paragraph", "reveal"):
                c1 = max(-3, min(3, int(handoff.get("c1_score", 0))))
                c2 = max(-3, min(3, int(handoff.get("c2_score", 0))))
                c3 = max(-3, min(3, int(handoff.get("c3_score", 0))))
                info = state.get("info", {})
                true_c1, true_c2, true_c3 = info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)
                errors = [abs(c1 - true_c1), abs(c2 - true_c2), abs(c3 - true_c3)]
                # Store per-stage predictions
                stage_preds = state.get("disc_stage_preds", {})
                stage_preds[reveal_stage] = {"c1": c1, "c2": c2, "c3": c3, "errors": errors}
                state["disc_stage_preds"] = stage_preds
                article = state.get("generated_article", "")
                if reveal_stage == "headline":
                    # Stage 2: First paragraph
                    para_end = article.find("\n\n", 100)
                    if para_end == -1:
                        para_end = min(500, len(article))
                    first_para = article[:para_end]
                    state["disc_reveal_stage"] = "paragraph"
                    return (
                        f"Now read the first paragraph and revise your scores:\n\n"
                        f"{first_para}\n\n"
                        f"Your headline-only predictions were: C1={c1}, C2={c2}, C3={c3}\n"
                        f"Revise your C1, C2, C3 scores with this additional context."
                    )
                elif reveal_stage == "paragraph":
                    # Stage 3: Full article (same as normal discriminate — already seen)
                    state["disc_reveal_stage"] = "reveal"
                    # Reveal gold for one criterion (C2, typically the most reliable label)
                    return (
                        f"The correct C2 score for this article is: C2={true_c2}\n\n"
                        f"Your current predictions: C1={c1}, C2={c2}, C3={c3}\n"
                        f"Knowing C2={true_c2}, revise your C1 and C3 scores.\n"
                        f"Think about what C2={true_c2:+d} tells you about the article's style, "
                        f"and how that constrains C1 and C3."
                    )
                elif reveal_stage == "reveal":
                    # Final stage — compute TD-weighted error
                    state["pred_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                    error = sum(errors)
                    state["disc_error"] = error
                    state["disc_reveal_stage"] = "done"
                    state["final_env_response"] = (
                        f"Game complete (progressive reveal)!\n"
                        f"  Target: C1={true_c1}, C2={true_c2}, C3={true_c3}\n"
                        f"  Final: C1={c1}, C2={c2}, C3={c3}\n"
                        f"  Total error: {error}/18"
                    )
                    return state["final_env_response"]

            c1 = int(handoff.get("c1_score", 0))
            c2 = int(handoff.get("c2_score", 0))
            c3 = int(handoff.get("c3_score", 0))
            reasoning = str(handoff.get("reasoning", ""))
            state["pred_scores"] = {"c1": c1, "c2": c2, "c3": c3}
            state["disc_reasoning"] = reasoning
            state["reasoning_quality"] = _score_reasoning_quality(reasoning)

            if "disc_first_scores" not in state:
                state["disc_first_scores"] = {"c1": c1, "c2": c2, "c3": c3}

            if self.disc_confidence_weight:
                conf = handoff.get("confidence", 5)
                state["disc_confidence"] = max(1, min(10, int(conf)))

            info = state.get("info", {})
            true_c1 = info.get("c1", 0)
            true_c2 = info.get("c2", 0)
            true_c3 = info.get("c3", 0)

            errors = [abs(c1 - true_c1), abs(c2 - true_c2), abs(c3 - true_c3)]
            error = sum(errors)

            # V100: Evidence-first — extract evidence before finalizing scores
            if self.disc_evidence_first and not state.get("disc_evidence_extracted"):
                state["disc_evidence_extracted"] = True
                state["disc_initial_reasoning"] = reasoning
                return (
                    f"Your initial scores: C1={c1}, C2={c2}, C3={c3}.\n\n"
                    f"Now GROUND each score in specific article text.\n"
                    f"For EACH criterion, quote 1-2 specific passages that support your score:\n"
                    f"- C1 Frame: Which passages reveal the causal framing?\n"
                    f"- C2 Temperature: Quote the journalist's most loaded OR measured language.\n"
                    f"- C3 Evidence: List each distinct evidence type found (stat, quote, data, anecdote).\n\n"
                    f"After citing evidence, submit FINAL scores (you may revise based on evidence review).\n"
                    f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "evidence-based explanation"}}</handoff>'
                )

            # V102: Consistency training — second pass with different example framing
            if self.disc_consistency_training and not state.get("disc_consistency_pass2"):
                state["disc_consistency_pass2"] = True
                state["disc_pass1_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                return (
                    f"You scored: C1={c1}, C2={c2}, C3={c3}.\n\n"
                    f"Now re-evaluate from a DIFFERENT ANGLE:\n"
                    f"- For C1: Think about WHO the article blames vs who benefits.\n"
                    f"- For C2: Read ONLY the journalist's own words (ignore all quotes).\n"
                    f"- For C3: Count how many DISTINCT information sources are used.\n\n"
                    f"Score again independently. Consistent scores suggest high confidence.\n"
                    f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "re-analysis"}}</handoff>'
                )

            # Sub-features trigger (V69): after initial scoring, ask for concrete sub-features
            if self.disc_sub_features and not state.get("disc_sub_feature_phase"):
                state["disc_sub_feature_phase"] = "sub_scores"
                return (
                    f"Now decompose your judgment into CONCRETE, OBSERVABLE sub-features.\n\n"
                    f"C1 Sub-features (each -3 to +3):\n"
                    f"  c1_headline: Does the headline use loaded framing? (-3=loaded structural, +3=loaded agency, 0=neutral)\n"
                    f"  c1_sources: Are cited sources balanced across perspectives? (-3=one-sided, +3=balanced)\n"
                    f"  c1_emphasis: What aspects get disproportionate emphasis? (-3=systemic problems, +3=individual solutions)\n\n"
                    f"C2 Sub-features:\n"
                    f"  c2_emotional: Emotional appeals in JOURNALIST'S language? (-3=highly emotional, +3=detached)\n"
                    f"  c2_certainty: Hedging vs absolute claims? (-3=absolute/alarming, +3=heavily hedged)\n\n"
                    f"C3 Sub-features:\n"
                    f"  c3_sourcing: Are claims attributed to specific sources? (-3=unattributed, +3=well-sourced)\n"
                    f"  c3_data: Are statistics/data cited? (-3=no data, +3=rich data)\n\n"
                    f"For each, cite the specific text that justifies your score.\n"
                    f'Submit: <handoff>{{"c1_headline": N, "c1_sources": N, "c1_emphasis": N, '
                    f'"c2_emotional": N, "c2_certainty": N, "c3_sourcing": N, "c3_data": N, "reasoning": "..."}}</handoff>'
                )

            # Sequential scoring trigger (V61): start C1-only phase
            if self.disc_sequential_scoring and not state.get("disc_sequential_phase"):
                state["disc_sequential_phase"] = "c1"
                return (
                    f"Now let's score each criterion individually with full attention.\n"
                    f"Focus ONLY on C1 (Frame — Cause & Remedy).\n"
                    f"Identify the article's causal claims:\n"
                    f"- What CAUSE does the article attribute the problem to?\n"
                    f"- What SOLUTION/remedy does it imply?\n"
                    f"Systemic/structural cause + collective solution = C1 negative.\n"
                    f"Individual agency + market/personal solution = C1 positive.\n"
                    f'Submit: <handoff>{{"c1_score": N, "reasoning": "..."}}</handoff>'
                )

            # Decomposed iterative trigger (V57): per-criterion correction
            if self.disc_decomposed_iterative and not state.get("disc_decomposed_started"):
                state["disc_decomposed_started"] = True
                bad_criteria = [("c1", errors[0]), ("c2", errors[1]), ("c3", errors[2])]
                bad_criteria = [(c, e) for c, e in bad_criteria if e >= 2]
                if bad_criteria:
                    state["disc_first_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                    first_crit = bad_criteria[0][0]
                    queue = [c for c, _ in bad_criteria[1:]]
                    state["disc_decomposed_queue"] = queue
                    state["disc_decomposed_criterion"] = first_crit
                    return self._decomposed_correction_prompt(first_crit, {"c1": c1, "c2": c2, "c3": c3}, state)

            # V90: Blind self-correction — no oracle, just "re-evaluate your weakest criterion"
            # PRODUCTION-VIABLE: this prompt can be used at inference time
            if self.disc_blind_selfcorrect and not state.get("disc_blind_revised"):
                state["disc_blind_revised"] = True
                state["disc_first_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                return (
                    f"Your initial scores: C1={c1}, C2={c2}, C3={c3}.\n\n"
                    f"Before finalizing, double-check each criterion:\n"
                    f"- C1: Did you score the CAUSAL FRAME or just react to the topic? "
                    f"An immigration article CAN have positive C1 if it frames via individual agency.\n"
                    f"- C2: Did you score the JOURNALIST'S language only? Quotes from officials don't count.\n"
                    f"- C3: Did you COUNT distinct evidence types? 1 type = negative, 4+ = positive.\n\n"
                    f"Which criterion are you LEAST confident about? Reconsider that one carefully.\n"
                    f'Submit revised scores: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "..."}}</handoff>'
                )

            # V91: Devil's advocate — argue AGAINST your own scores, then re-score
            # PRODUCTION-VIABLE: no oracle needed, model challenges itself
            if self.disc_devils_advocate and not state.get("disc_devils_phase"):
                state["disc_devils_phase"] = "challenge"
                state["disc_first_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                return (
                    f"Your initial scores: C1={c1}, C2={c2}, C3={c3}.\n\n"
                    f"Now play DEVIL'S ADVOCATE. For each criterion, argue why your score might be WRONG:\n"
                    f"- If you scored C1={c1:+d}, what evidence would support a different C1?\n"
                    f"- If you scored C2={c2:+d}, could the tone actually be more {'measured' if c2 < 0 else 'loaded'}?\n"
                    f"- If you scored C3={c3:+d}, did you miss any evidence types?\n\n"
                    f"After your counter-arguments, submit FINAL revised scores.\n"
                    f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "counter-arguments and final judgment"}}</handoff>'
                )

            # V88: Ensemble distillation — after first prediction, provide expert perspective hints
            if self.disc_ensemble_distill and not state.get("disc_ensemble_phase"):
                state["disc_ensemble_phase"] = "revision"
                state["disc_first_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                info = state.get("info", {})
                true_c1, true_c2, true_c3 = info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)
                # Generate expert hints based on true scores (like a teacher providing clues)
                c1_hint = ("structural/systemic" if true_c1 < 0 else "agency/individual" if true_c1 > 0 else "balanced")
                c2_hint = ("emotionally loaded" if true_c2 < 0 else "clinical/measured" if true_c2 > 0 else "neutral")
                c3_hint = ("limited sourcing" if true_c3 < 0 else "diverse evidence" if true_c3 > 0 else "mixed sourcing")
                return (
                    f"Your initial scores: C1={c1}, C2={c2}, C3={c3}.\n\n"
                    f"Expert panel perspectives:\n"
                    f"  • Political Analyst: The causal framing leans {c1_hint}.\n"
                    f"  • Linguist: The journalist's tone is {c2_hint}.\n"
                    f"  • Research Methodologist: The article uses {c3_hint}.\n\n"
                    f"Revise your scores based on these expert perspectives.\n"
                    f'Submit: <handoff>{{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "..."}}</handoff>'
                )

            # Standard iterative mode (with optional oracle withdrawal)
            revision_count = state.get("disc_revision_count", 0)
            max_revisions = 2 if self.disc_double_iterative else (1 if self.disc_iterative else 0)
            # Oracle withdrawal: randomly skip correction to force first-pass learning
            if self.disc_oracle_withdrawal > 0 and revision_count == 0:
                import random
                if random.random() < self.disc_oracle_withdrawal:
                    max_revisions = 0
            if max_revisions > 0 and revision_count < max_revisions:
                worst_idx = errors.index(max(errors))
                if errors[worst_idx] >= 2:
                    if revision_count == 0 and "disc_first_scores" not in state:
                        state["disc_first_scores"] = {"c1": c1, "c2": c2, "c3": c3}
                    state["disc_revision_count"] = revision_count + 1
                    state["disc_revised"] = True
                    criterion_names = ["C1 (Frame)", "C2 (Temperature)", "C3 (Evidence)"]
                    hints = [
                        "C1 scores the CAUSAL FRAME, not the topic. Immigration articles CAN be C1=0.",
                        "C2 scores the JOURNALIST'S language only. Quoted speech from officials does NOT count.",
                        "C3 counts DISTINCT evidence types. 1 type = negative. 4+ types = positive.",
                    ]
                    preds = [c1, c2, c3]
                    golds = [true_c1, true_c2, true_c3]
                    direction = "too negative" if preds[worst_idx] < golds[worst_idx] else "too positive"
                    if revision_count >= 1:
                        bad_criteria = [i for i, e in enumerate(errors) if e >= 2]
                        parts = []
                        for idx in bad_criteria:
                            d = "too negative" if preds[idx] < golds[idx] else "too positive"
                            parts.append(f"Your {criterion_names[idx]} score seems {d}. {hints[idx]}")
                        return " ".join(parts) + " Please submit final revised scores."
                    magnitude_text = ""
                    if self.disc_magnitude_feedback:
                        magnitude_text = f" (off by about {errors[worst_idx]} points)"
                    return (
                        f"Your {criterion_names[worst_idx]} score seems {direction}{magnitude_text}. "
                        f"Remember: {hints[worst_idx]} "
                        f"Please reconsider and submit revised scores."
                    )

            state["disc_error"] = error

            # V71: Update prediction history for self-distillation
            if self.disc_self_distill:
                source_id = info.get("source_id", -1)
                if source_id >= 0:
                    _PREDICTION_HISTORY[source_id] = {"c1": c1, "c2": c2, "c3": c3}

            # Hindsight explanation trigger
            if self.disc_hindsight_explain:
                state["disc_hindsight_phase"] = True
                return (
                    f"Your predictions: C1={c1}, C2={c2}, C3={c3}.\n"
                    f"The CORRECT scores are: C1={true_c1}, C2={true_c2}, C3={true_c3}.\n\n"
                    f"Now explain WHY these are the correct scores. "
                    f"For each criterion, cite specific evidence from the article:\n"
                    f"- C1={true_c1:+d}: What causal frame does the article use?\n"
                    f"- C2={true_c2:+d}: What is the journalist's rhetorical style?\n"
                    f"- C3={true_c3:+d}: What evidence types are present?\n\n"
                    f"Submit your explanation in a <handoff> block: "
                    f'{{\"explanation\": \"your explanation\"}}'
                )

            state["final_env_response"] = (
                f"Game complete!\n"
                f"  Target scores: C1={true_c1}, C2={true_c2}, C3={true_c3}\n"
                f"  Predicted scores: C1={c1}, C2={c2}, C3={c3}\n"
                f"  Total error: {error}/18\n"
                f"  Reasoning quality: {state['reasoning_quality']:.2f}\n"
                f"  Article length: {len(state['generated_article'])} chars"
            )
            return state["final_env_response"]

        return "Unknown phase."

    def _decomposed_correction_prompt(self, criterion: str, pred: dict, state: State) -> str:
        """Generate criterion-specific correction prompt for decomposed iterative (V57)."""
        info = state.get("info", {})
        preds = {"c1": pred.get("c1", 0), "c2": pred.get("c2", 0), "c3": pred.get("c3", 0)}
        golds = {"c1": info.get("c1", 0), "c2": info.get("c2", 0), "c3": info.get("c3", 0)}
        direction = "too negative" if preds[criterion] < golds[criterion] else "too positive"
        error = abs(preds[criterion] - golds[criterion])

        prompts = {
            "c1": (
                f"Focus ONLY on C1 (Frame). Your C1={preds['c1']} seems {direction} (off by ~{error}).\n"
                f"Re-read the article and complete these sentences:\n"
                f"- 'This article says the CAUSE of the problem is: ___'\n"
                f"- 'This article implies the SOLUTION is: ___'\n"
                f"Systemic/structural = C1 negative. Individual agency/market = C1 positive.\n"
                f'Submit revised C1: <handoff>{{"c1_score": N, "reasoning": "..."}}</handoff>'
            ),
            "c2": (
                f"Focus ONLY on C2 (Temperature). Your C2={preds['c2']} seems {direction} (off by ~{error}).\n"
                f"List 2-3 sentences written by the JOURNALIST (not quoted speech).\n"
                f"Score ONLY those sentences. Loaded adjectives = C2 negative. Hedged/clinical = C2 positive.\n"
                f'Submit revised C2: <handoff>{{"c2_score": N, "reasoning": "..."}}</handoff>'
            ),
            "c3": (
                f"Focus ONLY on C3 (Evidence). Your C3={preds['c3']} seems {direction} (off by ~{error}).\n"
                f"List every distinct evidence type: statistics, expert quotes, government data, "
                f"anecdotes, legal documents, counter-arguments, polls.\n"
                f"Count them. 1 type = C3 negative. 4+ types = C3 positive.\n"
                f'Submit revised C3: <handoff>{{"c3_score": N, "reasoning": "..."}}</handoff>'
            ),
        }
        return prompts[criterion]

    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        actor_id = state["trajectory_id"]
        info = state.get("info", {})
        articles = _load_articles()
        target_c1 = info.get("c1", 0)
        target_c2 = info.get("c2", 0)
        target_c3 = info.get("c3", 0)
        topic = info.get("topic", "General")

        # Select few-shot examples
        fewshot = _select_fewshot(articles, target_c1, target_c2, target_c3,
                                  n=3, exclude_id=info.get("source_id", -1))
        examples_text = _format_examples(fewshot)

        if actor_id == GENERATOR:
            if self.use_real_articles:
                return [SystemMessage(content="Acknowledge the bias analysis task."),
                        UserMessage(content='Submit: <handoff>{"status": "ready"}</handoff>')]
            prompt = GENERATOR_PROMPT.format(
                criteria=CRITERIA_DESC,
                examples=examples_text,
                topic=topic,
                c1=target_c1, c2=target_c2, c3=target_c3,
            )
            return [SystemMessage(content=prompt),
                    UserMessage(content="Write the article now. Submit your article in a <handoff> block with format: {\"article\": \"your article text\", \"reasoning\": \"brief explanation\"}")]

        else:  # DISCRIMINATOR
            article = state.get("generated_article", "No article available.")

            # Progressive reveal (V70): Start with headline only
            if self.disc_progressive_reveal and not state.get("disc_reveal_stage"):
                state["disc_reveal_stage"] = "headline"
                # Extract headline (first line or first sentence)
                lines = article.strip().split("\n")
                headline = lines[0] if lines else article[:100]
                if len(headline) < 20 and len(lines) > 1:
                    headline = lines[1]  # Sometimes first line is blank
                prompt = (
                    f"You are a media bias analyst. Predict bias scores from LIMITED information.\n\n"
                    f"{CRITERIA_DESC}\n\n"
                    f"Based on this HEADLINE ONLY, estimate C1/C2/C3 bias scores:\n\n"
                    f"\"{headline}\"\n\n"
                    f"This is your first impression — make your best guess with limited info.\n"
                    f"Submit scores in a <handoff> block: "
                    f'{{\"c1_score\": N, \"c2_score\": N, \"c3_score\": N, \"reasoning\": \"...\"}}'
                )
                return [SystemMessage(content=prompt),
                        UserMessage(content="Predict bias scores from the headline alone.")]

            # Select disc-specific few-shot examples (potentially more/diverse)
            if self.disc_blind:
                disc_examples = ""
            elif self.disc_diverse_examples:
                disc_fewshot = _select_diverse_fewshot(
                    articles, target_c1, target_c2, target_c3,
                    n=self.disc_num_examples, exclude_id=info.get("source_id", -1))
                disc_examples = _format_examples(disc_fewshot)
            elif self.disc_num_examples != 3:
                disc_fewshot = _select_fewshot(
                    articles, target_c1, target_c2, target_c3,
                    n=self.disc_num_examples, exclude_id=info.get("source_id", -1))
                disc_examples = _format_examples(disc_fewshot)
            else:
                disc_examples = examples_text

            topic_hint = f"Topic: {topic}\n" if self.disc_topic_hint else ""
            if self.disc_causal_prompt:
                disc_template = DISCRIMINATOR_PROMPT_CAUSAL
            elif self.disc_mixed_prompt:
                # Randomly select prompt type each rollout for robustness
                disc_template = random.choice([
                    DISCRIMINATOR_PROMPT_CALIBRATED,
                    DISCRIMINATOR_PROMPT_ANTIPRIOR,
                    DISCRIMINATOR_PROMPT_SELFCORRECT,
                ])
            elif self.disc_selfcorrect:
                disc_template = DISCRIMINATOR_PROMPT_SELFCORRECT
            elif self.disc_antiprior:
                disc_template = DISCRIMINATOR_PROMPT_ANTIPRIOR
            elif self.disc_calibrated:
                disc_template = DISCRIMINATOR_PROMPT_CALIBRATED
            else:
                disc_template = DISCRIMINATOR_PROMPT
            # Optionally mask quoted speech to force focus on journalist voice
            disc_article = article[:6000]
            if self.disc_mask_quotes > 0 and random.random() < self.disc_mask_quotes:
                disc_article = _re.sub(r'"[^"]{20,}"', '[QUOTE REMOVED]', disc_article)
                disc_article = _re.sub(r"'[^']{20,}'", '[QUOTE REMOVED]', disc_article)

            # Optionally neutralize topic-identifying nouns
            if self.disc_topic_neutral > 0 and random.random() < self.disc_topic_neutral:
                disc_article = _strip_topic_nouns(disc_article, topic)

            # Optionally strip source/publication identifiers
            if self.disc_source_blind > 0 and random.random() < self.disc_source_blind:
                disc_article = _strip_source_names(disc_article)

            # Comparative prompt: show anchor article with known scores, ask model to score target by comparison
            use_comparative = self.disc_comparative > 0 and random.random() < self.disc_comparative
            if use_comparative:
                pool = self.train_articles if self.train_articles else articles
                source_id = info.get("source_id", -1)
                anchor_candidates = [
                    a for a in pool
                    if a["id"] != source_id
                    and (abs(a["c1"] - target_c1) + abs(a["c2"] - target_c2) + abs(a["c3"] - target_c3)) >= 2
                ]
                if anchor_candidates:
                    anchor = random.choice(anchor_candidates[:15])
                    prompt = DISCRIMINATOR_PROMPT_COMPARATIVE.format(
                        criteria=CRITERIA_DESC,
                        anchor_source=anchor["source"], anchor_topic=anchor["topic"],
                        anchor_c1=anchor["c1"], anchor_c2=anchor["c2"], anchor_c3=anchor["c3"],
                        anchor_article=anchor["text"][:2500],
                        article=disc_article,
                        topic_hint=topic_hint,
                    )
                    return [SystemMessage(content=prompt),
                            UserMessage(content="Compare Article B against Article A and predict Article B's bias scores. Submit in a <handoff> block with format: {\"c1_score\": N, \"c2_score\": N, \"c3_score\": N, \"reasoning\": \"brief explanation\"}")]

            # Contrastive prompt: show a reference article with known scores
            use_contrastive = self.disc_contrastive > 0 and random.random() < self.disc_contrastive
            if use_contrastive:
                ref_candidates = [a for a in articles if a["id"] != info.get("source_id", -1)
                                  and (a["c1"] != target_c1 or a["c2"] != target_c2)]
                if ref_candidates:
                    ref = random.choice(ref_candidates[:10])
                    prompt = DISCRIMINATOR_PROMPT_CONTRASTIVE.format(
                        criteria=CRITERIA_DESC,
                        ref_source=ref["source"], ref_topic=ref["topic"],
                        ref_c1=ref["c1"], ref_c2=ref["c2"], ref_c3=ref["c3"],
                        ref_article=ref["text"][:2000],
                        article=disc_article,
                        topic_hint=topic_hint,
                    )
                    return [SystemMessage(content=prompt),
                            UserMessage(content="Analyze the article and predict bias scores. Submit in a <handoff> block with format: {\"c1_score\": N, \"c2_score\": N, \"c3_score\": N, \"reasoning\": \"brief explanation\"}")]

            # V83: Anchor scoring — use CLOSEST article as anchor, model estimates small deltas
            # Key difference from disc_comparative: anchor is close (delta ~1-2), not distant (delta ~5+)
            if self.disc_anchor_scoring:
                pool = self.train_articles if self.train_articles else articles
                source_id = info.get("source_id", -1)
                candidates = [
                    a for a in pool
                    if a["id"] != source_id
                ]
                if candidates:
                    # Sort by Manhattan distance to target, pick the CLOSEST
                    candidates.sort(key=lambda a: abs(a["c1"] - target_c1) + abs(a["c2"] - target_c2) + abs(a["c3"] - target_c3))
                    anchor = candidates[0]
                    anchor_dist = abs(anchor["c1"] - target_c1) + abs(anchor["c2"] - target_c2) + abs(anchor["c3"] - target_c3)
                    state["anchor_scores"] = {"c1": anchor["c1"], "c2": anchor["c2"], "c3": anchor["c3"]}
                    prompt = (
                        f"You are a media bias analyst. You will score an article by comparing it to a REFERENCE.\n\n"
                        f"{CRITERIA_DESC}\n\n"
                        f"{disc_examples}\n\n"
                        f"=== REFERENCE ARTICLE (scores known) ===\n"
                        f"Source: {anchor['source']} | Topic: {anchor['topic']}\n"
                        f"Known scores: C1={anchor['c1']:+d}, C2={anchor['c2']:+d}, C3={anchor['c3']:+d}\n"
                        f"---\n{anchor['text'][:2000]}\n---\n\n"
                        f"=== TARGET ARTICLE (predict its scores) ===\n"
                        f"{topic_hint}---\n{disc_article}\n---\n\n"
                        f"SCORING PROCESS:\n"
                        f"1. Compare the target to the reference on each criterion.\n"
                        f"2. The reference and target are similar — expect small differences (0-2 points).\n"
                        f"3. For each criterion: is the target HIGHER, LOWER, or SAME as the reference?\n"
                        f"4. Add your estimated delta to the reference score. Clamp to [-3, +3].\n"
                    )
                    return [SystemMessage(content=prompt),
                            UserMessage(content="Compare the target article to the reference and predict its bias scores. Submit in a <handoff> block with format: {\"c1_score\": N, \"c2_score\": N, \"c3_score\": N, \"reasoning\": \"brief explanation\"}")]

            # V101: Pairwise ranking — show reference article, ask for relative comparison + absolute scores
            if self.disc_pairwise_rank:
                pool = self.train_articles if self.train_articles else articles
                source_id = info.get("source_id", -1)
                candidates = [a for a in pool if a["id"] != source_id]
                if candidates:
                    import random as _rng
                    candidates.sort(key=lambda a: abs(a["c1"] - target_c1) + abs(a["c2"] - target_c2) + abs(a["c3"] - target_c3))
                    # Pick from top-5 closest for variety
                    anchor = _rng.choice(candidates[:min(5, len(candidates))])
                    prompt = (
                        f"You are a media bias analyst. Score the TARGET article using both COMPARISON and ABSOLUTE judgment.\n\n"
                        f"{CRITERIA_DESC}\n\n"
                        f"{disc_examples}\n\n"
                        f"=== REFERENCE ARTICLE ===\n"
                        f"Scores: C1={anchor['c1']:+d}, C2={anchor['c2']:+d}, C3={anchor['c3']:+d}\n"
                        f"---\n{anchor['text'][:1500]}\n---\n\n"
                        f"=== TARGET ARTICLE ===\n"
                        f"{topic_hint}---\n{disc_article}\n---\n\n"
                        f"PROCESS:\n"
                        f"1. For each criterion, determine if the TARGET is more positive, more negative, or similar to the REFERENCE.\n"
                        f"2. Use your comparison + the reference scores to estimate absolute scores.\n"
                        f"3. Cross-check against the examples above.\n"
                    )
                    return [SystemMessage(content=prompt),
                            UserMessage(content="Score the target article. Submit: <handoff>{\"c1_score\": N, \"c2_score\": N, \"c3_score\": N, \"reasoning\": \"comparison + absolute analysis\"}</handoff>")]

            # Topic deconfounding: add same-topic articles with different C1 scores
            deconfound_note = ""
            if self.disc_topic_deconfound:
                same_topic = [a for a in articles if a["topic"] == topic
                              and a["id"] != info.get("source_id", -1)]
                if len(same_topic) >= 2:
                    same_topic.sort(key=lambda a: a["c1"])
                    low_c1 = same_topic[0]
                    high_c1 = same_topic[-1]
                    if low_c1["c1"] != high_c1["c1"]:
                        deconfound_note = (
                            f"\n\nIMPORTANT — Topic does NOT determine C1 score. "
                            f"Two '{topic}' articles from your examples:\n"
                            f"  • '{low_c1.get('headline', 'Article')[:60]}' → C1={low_c1['c1']:+d} "
                            f"(frames via {'structural/systemic causes' if low_c1['c1'] < 0 else 'agency/market causes' if low_c1['c1'] > 0 else 'balanced framing'})\n"
                            f"  • '{high_c1.get('headline', 'Article')[:60]}' → C1={high_c1['c1']:+d} "
                            f"(frames via {'structural/systemic causes' if high_c1['c1'] < 0 else 'agency/market causes' if high_c1['c1'] > 0 else 'balanced framing'})\n"
                            f"Score C1 based on HOW the article frames causation, NOT what topic it covers."
                        )

            prompt = disc_template.format(
                criteria=CRITERIA_DESC,
                examples=disc_examples,
                article=disc_article,
                topic_hint=topic_hint,
            )
            if deconfound_note:
                prompt += deconfound_note

            # V71: Self-distillation — append prior predictions as soft anchor
            if self.disc_self_distill:
                source_id = info.get("source_id", -1)
                prior = _PREDICTION_HISTORY.get(source_id)
                if prior:
                    prompt += (
                        f"\n\nYour previous analysis of this article yielded: "
                        f"C1={prior['c1']:+d}, C2={prior['c2']:+d}, C3={prior['c3']:+d}.\n"
                        f"Re-read the article and submit your current scores. "
                        f"You may agree with or revise your prior judgment."
                    )

            # V74: Socratic probing — start with question generation
            if self.disc_socratic and not state.get("disc_socratic_phase"):
                state["disc_socratic_phase"] = "questions"
                return [SystemMessage(content=prompt), UserMessage(content=(
                    "Before scoring, generate 3 probing questions — one per criterion — "
                    "that will help you analyze this article's bias accurately.\n\n"
                    "Good C1 questions ask about SPECIFIC causal claims or implied solutions.\n"
                    "Good C2 questions ask about the JOURNALIST'S word choices (not quotes).\n"
                    "Good C3 questions ask about CONCRETE evidence types present.\n\n"
                    'Submit: <handoff>{"c1_question": "...", "c2_question": "...", "c3_question": "..."}</handoff>'
                ))]

            # Build user instruction with correct handoff format
            handoff_format = '{"c1_score": N, "c2_score": N, "c3_score": N, "reasoning": "brief explanation"}'
            if self.disc_confidence_weight:
                handoff_format = '{"c1_score": N, "c2_score": N, "c3_score": N, "confidence": N, "reasoning": "brief explanation"}'
            user_msg = f"Analyze the article and predict bias scores. Submit in a <handoff> block with format: {handoff_format}"
            if self.disc_confidence_weight:
                user_msg += "\nConfidence is 1-10: how sure are you of your predictions? 1=very unsure, 10=very confident."
            return [SystemMessage(content=prompt),
                    UserMessage(content=user_msg)]


CORPUS_PATH = Path(__file__).parent / "corpus_v2.json"


def _load_corpus() -> list[dict]:
    """Load combined corpus (real + synthetic articles) for training."""
    if not CORPUS_PATH.exists():
        return []
    with open(CORPUS_PATH) as f:
        data = _json.load(f)
    synthetic = data.get("synthetic_articles", [])
    # Normalize synthetic articles to same format as real articles
    result = []
    for a in synthetic:
        result.append({
            "id": a["id"],
            "topic": a.get("topic", "General"),
            "headline": f"Synthetic — {a.get('topic', 'General')}",
            "source": "synthetic",
            "c1": a.get("judge_c1", 0),
            "c2": a.get("judge_c2", 0),
            "c3": a.get("judge_c3", 0),
            "text": a["text"],
            "synthetic": True,
        })
    return result


def load_environment(
    advantage_center: float = 0.3,
    num_seed_rows: int = 200,
    gen_advantage_scale: float = 1.0,
    disc_blind: bool = False,
    per_criterion_scales: list[float] | None = None,
    disc_advantage_scale: float = 1.0,
    disc_per_criterion_scales: list[float] | None = None,
    disc_num_examples: int = 3,
    disc_diverse_examples: bool = False,
    disc_topic_hint: bool = False,
    disc_calibrated: bool = False,
    disc_antiprior: bool = False,
    disc_selfcorrect: bool = False,
    disc_mixed_prompt: bool = False,
    disc_mask_quotes: float = 0.0,
    disc_contrastive: float = 0.0,
    disc_comparative: float = 0.0,
    disc_topic_neutral: float = 0.0,
    disc_iterative: bool = False,
    disc_double_iterative: bool = False,
    disc_causal_prompt: bool = False,
    disc_magnitude_feedback: bool = False,
    disc_confidence_weight: bool = False,
    disc_hindsight_explain: bool = False,
    disc_decomposed_iterative: bool = False,
    disc_sequential_scoring: bool = False,
    disc_adaptive_advantage: bool = False,
    disc_relative_reward: bool = False,
    disc_perturbation: bool = False,
    disc_sub_features: bool = False,
    disc_progressive_reveal: bool = False,
    disc_self_distill: bool = False,
    disc_socratic: bool = False,
    disc_anti_regression: bool = False,
    disc_source_blind: float = 0.0,
    reasoning_quality_bonus: float = 0.0,
    disc_difficulty_boost: float = 0.0,
    disc_topic_deconfound: bool = False,
    use_llm_judge: bool = False,
    judge_base_url: str = "http://localhost:8000/v1",
    judge_model: str = "",
    use_real_articles: bool = False,
    c1_pos_oversample: int = 1,
    use_corpus: bool = False,
    real_oversample: int = 1,
    synthetic_max_mismatch: int = 99,
    c1_balance: bool = False,
    disc_curriculum: bool = False,
    disc_accuracy_reward: bool = False,
    disc_anchor_scoring: bool = False,
    disc_majority_vote: bool = False,
    disc_ensemble_distill: bool = False,
    disc_length_stratify: bool = False,
    disc_blind_selfcorrect: bool = False,
    disc_devils_advocate: bool = False,
    disc_first_pass_bonus: float = 0.0,
    disc_oracle_withdrawal: float = 0.0,
    disc_evidence_first: bool = False,
    disc_pairwise_rank: bool = False,
    disc_consistency_training: bool = False,
) -> vf.Environment:
    articles = _load_articles()
    assert len(articles) > 0, f"No usable articles found at {ARTICLES_PATH} or {ARTICLES_PATH_ALT}"

    # Build the training article pool
    train_articles = list(articles)  # Real articles always included
    if use_corpus:
        corpus_raw = _load_corpus()
        if synthetic_max_mismatch < 99:
            corpus_data = _json.loads(CORPUS_PATH.read_text())
            synth_raw = {a["id"]: a for a in corpus_data.get("synthetic_articles", [])}
            corpus = []
            for a in corpus_raw:
                raw = synth_raw.get(a["id"])
                if raw:
                    mismatches = [
                        abs(raw.get("target_c1", 0) - raw.get("judge_c1", 0)),
                        abs(raw.get("target_c2", 0) - raw.get("judge_c2", 0)),
                        abs(raw.get("target_c3", 0) - raw.get("judge_c3", 0)),
                    ]
                    if max(mismatches) <= synthetic_max_mismatch:
                        corpus.append(a)
                else:
                    corpus.append(a)
            _log.info("Filtered synthetic: %d/%d kept (max_mismatch=%d)",
                       len(corpus), len(corpus_raw), synthetic_max_mismatch)
        else:
            corpus = corpus_raw
        _log.info("Loaded %d synthetic articles from corpus", len(corpus))
        train_articles.extend(corpus)
        _log.info("Total training pool: %d articles (%d real + %d synthetic)",
                   len(train_articles), len(articles), len(corpus))

    # Build score tuples from the training pool
    score_tuples = []
    for a in train_articles:
        is_real = not a.get("synthetic", False)
        repeat = real_oversample if is_real else 1
        if c1_pos_oversample > 1 and a["c1"] > 0:
            repeat *= c1_pos_oversample
        for _ in range(repeat):
            score_tuples.append((a["c1"], a["c2"], a["c3"], a["topic"], a["id"]))

    # Balance C1 distribution by oversampling underrepresented bins
    if c1_balance:
        c1_bins = {}
        for t in score_tuples:
            c1_bins.setdefault(t[0], []).append(t)
        max_bin = max(len(v) for v in c1_bins.values())
        balanced = []
        for c1_val, entries in c1_bins.items():
            repeats = max(1, round(max_bin / len(entries)))
            balanced.extend(entries * repeats)
        _log.info("C1 balance: %d -> %d tuples (max_bin=%d, bins=%s)",
                   len(score_tuples), len(balanced), max_bin,
                   {k: len(v) for k, v in sorted(c1_bins.items())})
        score_tuples = balanced

    if disc_length_stratify:
        # Sort by article length (short first) — short articles have 2.8x lower MAE
        article_lens = {a["id"]: len(a.get("text", "")) for a in train_articles}
        score_tuples.sort(key=lambda t: article_lens.get(t[4], 5000))
        _log.info("Length-stratified: sorted %d tuples by article length (short first)", len(score_tuples))
    elif disc_curriculum:
        # Sort by difficulty (Manhattan distance from dataset mean) — easy first
        mean_c1 = sum(t[0] for t in score_tuples) / len(score_tuples)
        mean_c2 = sum(t[1] for t in score_tuples) / len(score_tuples)
        mean_c3 = sum(t[2] for t in score_tuples) / len(score_tuples)
        score_tuples.sort(key=lambda t: abs(t[0] - mean_c1) + abs(t[1] - mean_c2) + abs(t[2] - mean_c3))
        _log.info("Curriculum: sorted %d tuples easy-first (mean=%.1f,%.1f,%.1f)",
                   len(score_tuples), mean_c1, mean_c2, mean_c3)
    else:
        random.shuffle(score_tuples)
    _log.info("Score tuples: %d total (%d unique articles)",
              len(score_tuples), len(set(t[4] for t in score_tuples)))

    # Load base model predictions for V64 relative reward
    base_preds = {}
    if disc_relative_reward:
        base_pred_path = Path(__file__).parent / "base_predictions.json"
        if base_pred_path.exists():
            with open(base_pred_path) as f:
                raw_preds = _json.load(f)
            base_preds = {int(k): v for k, v in raw_preds.items()}
            _log.info("Loaded %d base model predictions for relative reward", len(base_preds))
        else:
            _log.warning("disc_relative_reward=True but no base_predictions.json found")

    # Create dataset: each row samples a target from the shuffled pool
    rows = []
    for s in range(num_seed_rows):
        c1, c2, c3, topic, source_id = score_tuples[s % len(score_tuples)]
        info_dict = {
            "seed": s,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "topic": topic,
            "source_id": source_id,
        }
        if source_id in base_preds:
            info_dict["base_pred"] = base_preds[source_id]
        rows.append({
            "prompt": [{"role": "user", "content": "Bias detector game."}],
            "task": "bias_detector",
            "info": info_dict,
        })

    dataset = Dataset.from_list(rows)

    def has_clean_rollout(state: State) -> bool:
        if state.get("error") is not None:
            return False
        if state.get("final_env_response") is None:
            return False
        if state.get("malformed_handoff"):
            return False
        return bool(state.get("rollout_completed_cleanly", True))

    async def game_reward(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        if disc_accuracy_reward:
            error = state.get("disc_error", 18)
            return max(0.0, 1.0 - error / 18.0)
        return 1.0

    async def per_agent_advantage(state: State) -> float:
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return 0.0

        info = state.get("info", {})
        pred = state.get("pred_scores", {})

        errors = [
            abs(pred.get("c1", 0) - info.get("c1", 0)),
            abs(pred.get("c2", 0) - info.get("c2", 0)),
            abs(pred.get("c3", 0) - info.get("c3", 0)),
        ]

        if per_criterion_scales:
            gen_adv = gen_advantage_scale * sum(s * (e / 6.0 - advantage_center) for s, e in zip(per_criterion_scales, errors)) / 3.0
        else:
            gen_adv = gen_advantage_scale * (sum(errors) / 18.0 - advantage_center)

        if disc_per_criterion_scales:
            disc_adv = disc_advantage_scale * sum(ds * ((1.0 - e / 6.0) - advantage_center) for ds, e in zip(disc_per_criterion_scales, errors)) / 3.0
        else:
            disc_adv = disc_advantage_scale * sum((1.0 - e / 6.0) - advantage_center for e in errors) / 3.0

        if reasoning_quality_bonus > 0:
            if use_llm_judge:
                rq = await _llm_judge_reasoning(
                    article=state.get("generated_article", ""),
                    reasoning=state.get("disc_reasoning", ""),
                    true_c1=info.get("c1", 0), true_c2=info.get("c2", 0), true_c3=info.get("c3", 0),
                    pred_c1=pred.get("c1", 0), pred_c2=pred.get("c2", 0), pred_c3=pred.get("c3", 0),
                    base_url=judge_base_url, model=judge_model,
                )
                state["reasoning_quality"] = rq
            else:
                rq = state.get("reasoning_quality", 0.0)
            disc_adv += reasoning_quality_bonus * (rq - 0.5)

        if disc_difficulty_boost > 0:
            max_score = max(abs(info.get("c1", 0)), abs(info.get("c2", 0)), abs(info.get("c3", 0)), 1)
            difficulty_scale = 1.0 + disc_difficulty_boost * (3 - max_score) / 3.0
            disc_adv *= difficulty_scale

        # Confidence weighting: scale advantage by confidence/10
        if disc_confidence_weight:
            conf = state.get("disc_confidence", 5)
            conf_scale = conf / 10.0  # 0.1 to 1.0
            disc_adv *= conf_scale

        # Adaptive advantage (V60): boost hard examples, dampen easy ones
        if disc_adaptive_advantage:
            source_id = info.get("source_id", 0)
            per_crit_mae = sum(errors) / 3.0
            hist = _ERROR_HISTORY.get(source_id, [])
            if len(hist) >= 3:
                mean_err = sum(hist) / len(hist)
                if mean_err > 1.5:
                    disc_adv *= 1.5  # Boost persistently-wrong examples
                elif mean_err < 0.5:
                    disc_adv *= 0.5  # Dampen already-mastered examples
            hist.append(per_crit_mae)
            _ERROR_HISTORY[source_id] = hist[-5:]  # Keep last 5

        # V64: Relative reward — advantage based on improvement over base model
        if disc_relative_reward:
            base_pred = info.get("base_pred", {})
            if base_pred:
                base_errors = [
                    abs(base_pred.get("c1", 0) - info.get("c1", 0)),
                    abs(base_pred.get("c2", 0) - info.get("c2", 0)),
                    abs(base_pred.get("c3", 0) - info.get("c3", 0)),
                ]
                if disc_per_criterion_scales:
                    improvement = [
                        ds * ((be - le) / 6.0)
                        for ds, be, le in zip(disc_per_criterion_scales, base_errors, errors)
                    ]
                    disc_adv = disc_advantage_scale * (sum(improvement) / 3.0 - advantage_center)
                else:
                    improvement = (sum(base_errors) - sum(errors)) / 18.0
                    disc_adv = disc_advantage_scale * (improvement - advantage_center)

        # V94+: First-pass bonus — reward good initial scores in iterative training
        # Prevents oracle dependency by giving separate advantage to first-pass accuracy
        if disc_first_pass_bonus > 0:
            first = state.get("disc_first_scores", {})
            if first:
                first_errors = [
                    abs(first.get("c1", 0) - info.get("c1", 0)),
                    abs(first.get("c2", 0) - info.get("c2", 0)),
                    abs(first.get("c3", 0) - info.get("c3", 0)),
                ]
                first_accuracy = 1.0 - sum(first_errors) / 18.0
                disc_adv += disc_first_pass_bonus * (first_accuracy - advantage_center)

        # V102: Consistency bonus — reward when two independent scorings agree
        if disc_consistency_training:
            pass1 = state.get("disc_pass1_scores", {})
            if pass1:
                consistency_error = (
                    abs(pass1.get("c1", 0) - pred.get("c1", 0))
                    + abs(pass1.get("c2", 0) - pred.get("c2", 0))
                    + abs(pass1.get("c3", 0) - pred.get("c3", 0))
                )
                consistency_bonus = 0.2 * (1.0 - consistency_error / 18.0)
                disc_adv += consistency_bonus

        # V71: Self-distillation consistency bonus
        if disc_self_distill:
            source_id = info.get("source_id", -1)
            prior = _PREDICTION_HISTORY.get(source_id)
            if prior:
                prior_errors = [
                    abs(prior["c1"] - info.get("c1", 0)),
                    abs(prior["c2"] - info.get("c2", 0)),
                    abs(prior["c3"] - info.get("c3", 0)),
                ]
                consistent = all(abs(pred.get(k, 0) - prior[k]) <= 1 for k in ("c1", "c2", "c3"))
                improved = sum(errors) < sum(prior_errors)
                if improved:
                    disc_adv += 0.2  # Improved over prior: big bonus
                elif consistent and sum(errors) <= 3:
                    disc_adv += 0.1  # Consistent AND accurate: small bonus
                elif not consistent and sum(errors) > sum(prior_errors):
                    disc_adv -= 0.1  # Changed AND got worse: penalty

        # V74: Socratic answer quality bonus
        if disc_socratic:
            aq = state.get("disc_socratic_answer_quality", 0.0)
            disc_adv += 0.1 * (aq - 0.5)  # Bonus for citing specific text

        # V78: Anti-regression guard — penalize forgetting
        if disc_anti_regression:
            source_id = info.get("source_id", -1)
            current_error = sum(errors) / 3.0 if errors else state.get("disc_error", 6) / 3.0
            best_ever = _BEST_ERROR_HISTORY.get(source_id, 999.0)
            if current_error < best_ever:
                _BEST_ERROR_HISTORY[source_id] = current_error
                disc_adv += 0.1  # New personal best: small bonus
            elif current_error > best_ever + 0.5:
                regression = (current_error - best_ever) / 3.0
                disc_adv -= 0.3 * regression  # Penalize regression proportionally

        gen_steps = [s for s in trajectory if s.get("trajectory_id") == GENERATOR]
        disc_steps = [s for s in trajectory if s.get("trajectory_id") == DISCRIMINATOR]

        for step in gen_steps:
            step["advantage"] = gen_adv

        # Progressive reveal (V70): TD-style decaying advantage per stage
        stage_preds = state.get("disc_stage_preds", {})
        if disc_progressive_reveal and stage_preds and len(disc_steps) >= 2:
            multipliers = [3.0, 2.0, 1.0, 0.5]  # headline, paragraph, full/reveal
            true_scores = [info.get("c1", 0), info.get("c2", 0), info.get("c3", 0)]
            for i, step in enumerate(disc_steps):
                mult = multipliers[min(i, len(multipliers) - 1)]
                # Compute this step's accuracy
                stage_names = ["headline", "paragraph", "reveal"]
                if i < len(stage_names) and stage_names[i] in stage_preds:
                    sp = stage_preds[stage_names[i]]
                    step_errors = sp.get("errors", errors)
                    step_mae = sum(step_errors) / 3.0
                    step_adv = disc_advantage_scale * ((1.0 - sum(step_errors) / 18.0) - advantage_center)
                    # TD bonus: improvement over previous stage
                    if i > 0 and stage_names[i - 1] in stage_preds:
                        prev_errors = stage_preds[stage_names[i - 1]].get("errors", step_errors)
                        improvement = (sum(prev_errors) - sum(step_errors)) / 18.0
                        step_adv += 0.5 * improvement
                    step["advantage"] = step_adv * mult
                else:
                    step["advantage"] = disc_adv
        # Hindsight explanation: scoring steps get disc_adv, explanation step gets reasoning bonus
        elif disc_hindsight_explain and len(disc_steps) >= 2:
            # All but last disc step = scoring (get disc_adv)
            for step in disc_steps[:-1]:
                step["advantage"] = disc_adv
            # Last disc step = explanation (get reasoning quality advantage)
            explain_rq = state.get("hindsight_reasoning_quality", 0.0)
            explain_adv = disc_advantage_scale * (explain_rq - 0.3)
            disc_steps[-1]["advantage"] = explain_adv
        else:
            for step in disc_steps:
                step["advantage"] = disc_adv

        return 0.0

    async def discriminator_mae(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        return float(state.get("disc_error", 0)) / 3.0  # Per-criterion average

    async def c1_mae(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        info = state.get("info", {})
        pred = state.get("pred_scores", {})
        return float(abs(pred.get("c1", 0) - info.get("c1", 0)))

    async def c2_mae(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        info = state.get("info", {})
        pred = state.get("pred_scores", {})
        return float(abs(pred.get("c2", 0) - info.get("c2", 0)))

    async def c3_mae(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        info = state.get("info", {})
        pred = state.get("pred_scores", {})
        return float(abs(pred.get("c3", 0) - info.get("c3", 0)))

    async def first_pass_mae(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        first = state.get("disc_first_scores", {})
        if not first:
            return float(state.get("disc_error", 0)) / 3.0
        info = state.get("info", {})
        return (abs(first.get("c1", 0) - info.get("c1", 0))
                + abs(first.get("c2", 0) - info.get("c2", 0))
                + abs(first.get("c3", 0) - info.get("c3", 0))) / 3.0

    async def correction_delta(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        first = state.get("disc_first_scores", {})
        pred = state.get("pred_scores", {})
        if not first or not pred:
            return 0.0
        return (abs(first.get("c1", 0) - pred.get("c1", 0))
                + abs(first.get("c2", 0) - pred.get("c2", 0))
                + abs(first.get("c3", 0) - pred.get("c3", 0))) / 3.0

    async def article_length(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        return float(len(state.get("generated_article", "")))

    async def fooling_rate(state: State) -> float:
        """1.0 if discriminator total error > 6 (avg 2+ per criterion)."""
        if not has_clean_rollout(state):
            return 0.0
        return 1.0 if state.get("disc_error", 0) > 6 else 0.0

    async def game_completion_rate(state: State) -> float:
        return 1.0 if has_clean_rollout(state) else 0.0

    async def reasoning_quality(state: State) -> float:
        if not has_clean_rollout(state):
            return 0.0
        return float(state.get("reasoning_quality", 0.0))

    rubric = vf.Rubric(
        funcs=[
            game_reward, per_agent_advantage,
            discriminator_mae, c1_mae, c2_mae, c3_mae,
            first_pass_mae, correction_delta,
            article_length, fooling_rate, game_completion_rate,
            reasoning_quality,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    return BiasDetectorEnv(
        dataset=dataset, rubric=rubric, disc_blind=disc_blind,
        disc_num_examples=disc_num_examples,
        disc_diverse_examples=disc_diverse_examples,
        disc_topic_hint=disc_topic_hint,
        disc_calibrated=disc_calibrated,
        disc_antiprior=disc_antiprior,
        disc_selfcorrect=disc_selfcorrect,
        disc_mixed_prompt=disc_mixed_prompt,
        disc_mask_quotes=disc_mask_quotes,
        disc_contrastive=disc_contrastive,
        disc_comparative=disc_comparative,
        disc_topic_neutral=disc_topic_neutral,
        disc_iterative=disc_iterative,
        disc_double_iterative=disc_double_iterative,
        disc_causal_prompt=disc_causal_prompt,
        disc_magnitude_feedback=disc_magnitude_feedback,
        disc_confidence_weight=disc_confidence_weight,
        disc_hindsight_explain=disc_hindsight_explain,
        disc_decomposed_iterative=disc_decomposed_iterative,
        disc_sequential_scoring=disc_sequential_scoring,
        disc_adaptive_advantage=disc_adaptive_advantage,
        disc_relative_reward=disc_relative_reward,
        disc_perturbation=disc_perturbation,
        disc_sub_features=disc_sub_features,
        disc_progressive_reveal=disc_progressive_reveal,
        disc_self_distill=disc_self_distill,
        disc_socratic=disc_socratic,
        disc_anti_regression=disc_anti_regression,
        disc_anchor_scoring=disc_anchor_scoring,
        disc_majority_vote=disc_majority_vote,
        disc_ensemble_distill=disc_ensemble_distill,
        disc_length_stratify=disc_length_stratify,
        disc_blind_selfcorrect=disc_blind_selfcorrect,
        disc_devils_advocate=disc_devils_advocate,
        disc_first_pass_bonus=disc_first_pass_bonus,
        disc_oracle_withdrawal=disc_oracle_withdrawal,
        disc_evidence_first=disc_evidence_first,
        disc_pairwise_rank=disc_pairwise_rank,
        disc_consistency_training=disc_consistency_training,
        disc_source_blind=disc_source_blind,
        disc_topic_deconfound=disc_topic_deconfound,
        use_real_articles=use_real_articles,
        train_articles=train_articles if use_corpus else None,
        system_prompt=None, use_verifiers_advantages=True,
    )
