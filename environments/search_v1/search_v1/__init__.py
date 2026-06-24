"""search-v1 — QUEST / OpenSeeker / REDSearcher research tasksets on vf.Taskset.

One harness-agnostic ``SearchTaskset`` whose ``backend`` config field selects the
dataset + scoring strategy. Pair with an agent harness that gives the model
web-search tools and writes the final answer to ``/task/answer.txt`` (e.g. the
``rlm`` harness with ``websearch`` / ``open_webpage`` skills — see ``rlm_search_v1``).

``backend`` selects the taskset::

    vf eval search-v1 -a '{"backend": "openseeker"}'
    vf eval search-v1 -a '{"backend": "redsearcher", "difficulty": "easy"}'
    vf eval search-v1 -a '{"backend": "quest", "category": "objective"}'
"""

from search_v1._base import SearchConfig, SearchTask, SearchTaskset

__all__ = ["SearchTaskset", "SearchConfig", "SearchTask"]
