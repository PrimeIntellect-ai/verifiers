# Bias Labeling Framework (v1)

## Goal

Quickly and consistently label ~100–200 news articles across three criteria to bootstrap model training. Two raters (A and B) will:

1. Define what **0** means using shared anchors
2. Create a **gold set** of manual labels
3. Use **AI to propose labels** for remaining articles, with human triage review

---

## Scoring System

Scores range from **-3 to +3**, derived in two passes:

| Pass | Action | Values |
|------|--------|--------|
| **Pass 1** | Direction only | `-` / `0` / `+` |
| **Pass 2** | Magnitude (only if non-zero) | `1` / `2` / `3` |

**Final score** = Direction × Magnitude (e.g., `-` × `2` = `-2`)

---

## The Three Criteria

### C1 — Frame (Cause & Remedy)

- **Allowed evidence:** ONLY sentences that explain causes ("because", "driven by") and/or propose actions ("should", "needs to")
- **Negative (-):** structural / collective framing
- **Positive (+):** agency / order / market framing
- **Zero (0):** no clear direction, or mixed

### C2 — Temperature (Rhetorical Heat)

- **Allowed evidence:** ONLY language and tone
- **Negative (-):** loaded / moralizing / sneering / alarmist
- **Positive (+):** measured / clinical / careful hedging / separation of fact vs. claim
- **Zero (0):** neutral newsroom baseline

### C3 — Evidence Diversity

- **Allowed evidence:** WHAT kinds of support appear (count evidence modes)
- **Negative (-):** single-mode or one-sided sourcing
- **Positive (+):** multiple modes and attributable counter-perspective
- **Zero (0):** uneven or mixed

---

## Efficiency Rules

- If a criterion scores **0**: no explanation required
- If **non-zero**: paste 1 supporting span (C1: "why/should" sentence; C2: loaded/clinical phrase) and tick evidence modes for C3
- Only **adjudicate disagreements** in direction (one rater says 0 and the other non-zero, or opposite signs)

---

## Workflow

| Step | Description |
|------|-------------|
| 1 | **Build source pool** — diverse sources (hard news / analysis / opinion; multiple topics) |
| 2 | **Calibration set** — 24 articles → direction-only labels from both raters |
| 3 | **Create "0 anchors"** — 5 per criterion (15 total) |
| 4 | **Gold set** — 40–60 articles labeled with full spans and evidence modes |
| 5 | **AI assist** — model proposes scores + spans + modes; humans triage low-confidence and near-zero cases |

---

## Lookup / Dropdown Values

### Topics
| Value |
|-------|
| Immigration / Border |
| Economy / Taxes / Inflation |
| Healthcare |
| Crime / Policing |
| Climate / Energy |
| Foreign Policy |
| Education / Culture |
| Tech / Free Speech |
| Elections / Governance |
| Other |

### Article Types
| Value |
|-------|
| Hard News |
| Analysis / Explainer |
| Opinion / Editorial |
| Interview / Q&A |
| Press Release / Statement |
| Report / Research |

### Direction
`-` | `0` | `+`

### Magnitude
`1` | `2` | `3`

### Review Status
| Value | Meaning |
|-------|---------|
| To Collect | Article identified but text not yet gathered |
| Collected | Text gathered, awaiting labeling |
| Manual Label Pending | Ready for human labeling |
| Manual Labeled (A only) | Rater A has labeled |
| Manual Labeled (B only) | Rater B has labeled |
| Needs Adjudication | Both raters labeled; disagreements need resolution |
| Final | Adjudicated and finalized |
| AI Labeled (Unreviewed) | AI has proposed labels, not yet human-reviewed |
| AI Labeled (Reviewed) | AI labels reviewed by human |
| Excluded | Article excluded from dataset |

### Priority
`High` | `Medium` | `Low`

### AI Confidence
`Low` | `Medium` | `High`

### Evidence Modes (for C3)
| Mode |
|------|
| Data / Statistics |
| Primary Document |
| Named Official / Spokesperson |
| Named Expert / Academic |
| Anonymous Source |
| Eyewitness / Firsthand |
| Opposition Quoted |
| Advocacy Org |
| Think Tank |
| Social Media |
| Other |

---

## Article Data Schema (`articles.json`)

Each article is a JSON object. Keys present depend on how far along labeling is. All possible keys:

### Metadata
| Key | Type | Description |
|-----|------|-------------|
| `id` | int | Sequential article ID |
| `pub_date` | string | Publication date (format: `MM.DD.YY`) |
| `source` | string | Publisher name (e.g., "New York Times", "Reuters") |
| `url` | string | Full article URL |
| `headline` | string | Article headline |
| `topic` | string | Topic category (see Lookups) |
| `article_type` | string | Article type (see Lookups) |
| `notes` | string | General notes |
| `status` | string | Review status (see Lookups) |
| `calibration_set` | string | "Yes" if part of calibration set |
| `article_text` | string | Full cleaned article body text |

### Rater A Scores
| Key | Type | Description |
|-----|------|-------------|
| `c1_score_a` | number | C1 Frame score from Rater A (-3 to +3) |
| `c2_score_a` | number | C2 Temperature score from Rater A (-3 to +3) |
| `c3_score_a` | number/string | C3 Evidence score from Rater A (-3 to +3) |
| `rater_a_notes` | string | Rater A's full reasoning and notes |
| `c2_support_span_a` | string | Rater A's support span / justification text |
| `c3_evidence_dir_a` | string | C3 direction from Rater A (`-`/`0`/`+`) |
| `c3_evidence_mag_a` | number | C3 magnitude from Rater A (1/2/3) |
| `c3_evidence_score_a` | number | C3 computed score (dir × mag) for Rater A |
| `c3_evidence_support_span_a` | string | C3 support span from Rater A |
| `c3_evidence_modes_a` | string | Evidence modes identified by Rater A |

### Rater B Scores
| Key | Type | Description |
|-----|------|-------------|
| `c1_frame_dir_b` | string | C1 direction from Rater B |
| `c1_frame_mag_b` | number | C1 magnitude from Rater B |
| `c1_frame_score_b` | number | C1 computed score for Rater B |
| `c1_frame_support_span_b` | string | C1 support span from Rater B |
| `c2_temp_dir_b` | string | C2 direction from Rater B |
| `c2_temp_mag_b` | number | C2 magnitude from Rater B |
| `c2_temp_score_b` | number | C2 computed score for Rater B |
| `c2_temp_support_span_b` | string | C2 support span from Rater B |
| `c3_evidence_dir_b` | string | C3 direction from Rater B |
| `c3_evidence_mag_b` | number | C3 magnitude from Rater B |
| `c3_evidence_score_b` | number | C3 computed score for Rater B |
| `c3_evidence_support_span_b` | string | C3 support span from Rater B |
| `c3_evidence_modes_b` | string | Evidence modes identified by Rater B |

### Consensus / Final Scores
| Key | Type | Description |
|-----|------|-------------|
| `c1_score_consensus` | number | Adjudicated C1 score |
| `c2_score_consensus` | number | Adjudicated C2 score |
| `c3_score_consensus` | number | Adjudicated C3 score |
| `c1_final_score` | number | Final C1 score |
| `c1_final_span` | string | Final C1 support span |
| `c2_final_score` | number | Final C2 score |
| `c2_final_span` | string | Final C2 support span |
| `c3_final_score` | number | Final C3 score |
| `c3_final_modes` | string | Final evidence modes |
| `c3_final_span` | string | Final C3 support span |
| `adjudicator` | string | Who adjudicated |
| `finalized` | string | "Y" if finalized |

### AI Labeling
| Key | Type | Description |
|-----|------|-------------|
| `ai_c1_score` | number | AI-proposed C1 score |
| `ai_c1_span` | string | AI-proposed C1 support span |
| `ai_c1_conf` | string | AI confidence for C1 |
| `ai_c2_score` | number | AI-proposed C2 score |
| `ai_c2_span` | string | AI-proposed C2 support span |
| `ai_c2_conf` | string | AI confidence for C2 |
| `ai_c3_score` | number | AI-proposed C3 score |
| `ai_c3_modes` | string | AI-proposed evidence modes |
| `ai_c3_span` | string | AI-proposed C3 support span |
| `ai_c3_conf` | string | AI confidence for C3 |
| `ai_notes` | string | AI reasoning notes |

### Inter-Rater Agreement (computed)
| Key | Type | Description |
|-----|------|-------------|
| `c1_dir_agree` | number | 1 if Raters A & B agree on C1 direction, 0 if not |
| `c1_sign_agree` | number | 1 if A & B C1 scores have same sign |
| `c2_dir_agree` | number | 1 if A & B agree on C2 direction |
| `c2_sign_agree` | number | 1 if A & B C2 scores have same sign |
| `c3_dir_agree` | number | 1 if A & B agree on C3 direction |
| `c3_sign_agree` | number | 1 if A & B C3 scores have same sign |

---

## Anchors (Not Yet Populated)

The framework includes a template for **60 anchor articles** — benchmark examples that define what a score of 0 looks like for each criterion. Each anchor has:

- Anchor ID, Criterion, Score, URL, Headline, Source, Topic, Article Type
- "Why it's an anchor" (1-sentence justification)
- Added By, Date Added

**Status:** Template only — no anchors have been filled in yet.

---

## Summary Metrics (from xlsx formulas)

These are computed in the spreadsheet and reference the Articles sheet:

| Metric | Formula |
|--------|---------|
| Total rows (non-empty URL) | `COUNTIF(Articles!D3:D251,"<>")` |
| Rows with both raters' directions for all 3 criteria | `COUNTIFS(...)` across all direction columns |
| C1 Direction agreement | `AVERAGE(Articles!BF3:BF251)` |
| C1 Sign agreement | `AVERAGE(Articles!BG3:BG251)` |
| C2 Direction agreement | `AVERAGE(Articles!BH3:BH251)` |
| C2 Sign agreement | `AVERAGE(Articles!BI3:BI251)` |
| C3 Direction agreement | `AVERAGE(Articles!BJ3:BJ251)` |
| C3 Sign agreement | `AVERAGE(Articles!BK3:BK251)` |
| Status counts | `COUNTIFS(...)` for each ReviewStatus value |

---

## Current Dataset Status

- **54 articles** with URLs and text populated
- **51 articles** have Rater A scores for C1 and C2
- **49 articles** have Rater A C3 scores
- **25 articles** have Rater A support spans
- **8 articles** have consensus scores (C1, C2, C3)
- **21 articles** are marked as calibration set
- **0 articles** have Rater B scores (B columns are empty)
- **0 articles** have AI labels
- **0 articles** have final/adjudicated scores
- **0 anchors** defined
