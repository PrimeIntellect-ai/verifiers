"""wikispeedia: navigate Wikipedia by clicking links to reach a target article.

A stateful tool example, authored as a `vf.Toolset` (`WikiToolset` below). The taskset
generates `(source, target)` pairs from the SNAP graph (loaded once, host-side) and, per
rollout, launches a toolset holding the navigation state (current article + path), built in
its `setup`. The harness calls `wiki_click_link` to move; the reward reads the toolset's
`TARGET REACHED` marker off the trace.
"""

import random

from pydantic import PrivateAttr

import verifiers.v1 as vf

from wikispeedia_v1.graph import WikiGraph, format_article

SYSTEM = (
    "This game is easy and fun: starting from the first Wikipedia article, reach the "
    "second one by following links. Each article ends with `Available links: ...` — "
    "those are the only links you may follow. Use the `wiki_click_link` tool to "
    "navigate, and `wiki_go_back` to undo. Think about which broader concepts connect "
    "the source to the target, and aim for the article most likely to link your "
    "destination."
)

_wiki: WikiGraph | None = None


def graph() -> WikiGraph:
    global _wiki
    if _wiki is None:
        _wiki = WikiGraph.load()
    return _wiki


def sample_pairs(wiki: WikiGraph, n: int, lo: int, hi: int, seed: int):
    """Sample `n` unique `(source, target, dist)` tuples within the distance band."""
    rng = random.Random(seed)
    articles = sorted(wiki.articles)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, int]] = []
    for _ in range(n * 200):
        if len(pairs) >= n:
            break
        s, t = rng.choice(articles), rng.choice(articles)
        if s == t or (s, t) in seen:
            continue
        dist = wiki.shortest_path_length(s, t)
        if dist is not None and lo <= dist <= hi:
            pairs.append((s, t, dist))
            seen.add((s, t))
    return pairs


class WikiTask(vf.Task):
    source: str
    target: str
    shortest_path: int
    """The graph's shortest-path hop count from source to target."""


class WikispeediaConfig(vf.TasksetConfig):
    num_tasks: int = 20
    min_dist: int = 3
    max_dist: int = 8
    seed: int = 0
    max_turns: int = 30
    links_only: bool = True
    """Show only each article's outgoing-link menu (no prose) — the classic
    Wikispeedia formulation; keeps the prompt small."""


class WikiToolset(vf.Toolset):
    """Holds the rollout's navigation state (current article + path) and serves
    `click_link`/`go_back`. On reaching the target it emits a `TARGET REACHED` marker in the
    tool result — which lands in the trace, where the reward reads it."""

    source: str
    target: str
    links_only: bool = True
    _wiki: WikiGraph = PrivateAttr(default=None)
    _path: list[str] = PrivateAttr(default_factory=list)

    async def setup(self) -> None:
        self._wiki = WikiGraph.load()
        self._path = [self.source]

    @vf.tool
    def click_link(self, article: str) -> str:
        """Navigate to a linked article (must be an available link from the current one)."""
        current = self._path[-1]
        available = self._wiki.get_links(current)
        target = self._wiki.normalize_name(article)
        if target is None or target not in available:
            return (
                f"'{article}' is not a valid link from '{current}'.\n"
                f"Available links: {', '.join(available) or '(none)'}"
            )
        self._path.append(target)
        page = format_article(self._wiki, target, self.links_only)
        if target == self.target:
            return (
                f"TARGET REACHED 🎯 You navigated to the target '{self.target}'.\n\n{page}"
            )
        return page

    @vf.tool
    def go_back(self) -> str:
        """Go back to the previous article (undo the last click)."""
        if len(self._path) <= 1:
            return "You are already at the starting article. Cannot go back."
        self._path.pop()
        return format_article(self._wiki, self._path[-1], self.links_only)


class WikispeediaTaskset(vf.Taskset[WikiTask, WikispeediaConfig]):
    def load_tasks(self) -> list[WikiTask]:
        wiki = graph()
        pairs = sample_pairs(
            wiki,
            self.config.num_tasks,
            self.config.min_dist,
            self.config.max_dist,
            self.config.seed,
        )
        return [
            WikiTask(
                idx=i,
                name=f"{source} -> {target}",
                source=source,
                target=target,
                shortest_path=dist,
                instruction=(
                    f"{SYSTEM}\n\nYour mission: {source} >> {target}\n\n"
                    f"Here is the starting article:\n\n"
                    f"{format_article(wiki, source, self.config.links_only)}"
                ),
            )
            for i, (source, target, dist) in enumerate(pairs)
        ]

    def tools(self, task: WikiTask) -> list[vf.Toolset]:
        return [
            WikiToolset(
                name="wiki",
                source=task.source,
                target=task.target,
                links_only=self.config.links_only,
            )
        ]

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        # Halt once the target is reached (no answer needed) or turns are exhausted.
        reached = any(
            "TARGET REACHED" in (m.content or "") for m in trace.tool_messages
        )
        return reached or trace.num_turns >= self.config.max_turns

    @vf.reward(weight=1.0)
    async def reached_target(
        self, task: WikiTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        # The server emits "TARGET REACHED" in the click result on success; that
        # tool message is recorded in the trace.
        return float(
            any("TARGET REACHED" in (m.content or "") for m in trace.tool_messages)
        )

    @vf.metric
    async def clicks(
        self, task: WikiTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        return float(
            sum(
                tc.name == "wiki_click_link"
                for m in trace.assistant_messages
                for tc in (m.tool_calls or [])
            )
        )


def load_taskset(config: WikispeediaConfig) -> WikispeediaTaskset:
    return WikispeediaTaskset(config)
