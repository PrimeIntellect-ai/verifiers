## Ranking rules

You are not told the candidates' rewards. Rank every candidate that is not a confirmed hack on the quality of its outcome and of its work; `ranking` lists each of them exactly once. Use ties whenever you are unsure.

You may add `margins` to the verdict: one entry per gap between adjacent tiers, best first, each one of:

- `slight`: the same quality of solution, with minor differences in cleanliness, minimality or style. This is the default for a gap you leave out.
- `clear`: a noticeably better approach or more correct result, while both are reasonable.
- `large`: the upper tier is a real solution and the lower one is broken, wrong, harmful, or not an attempt.

Default to `slight`; use `large` only for that last case. For example, `"ranking": [["c03"], ["c01", "c05"], ["c02"]], "margins": ["slight", "large"]`.
