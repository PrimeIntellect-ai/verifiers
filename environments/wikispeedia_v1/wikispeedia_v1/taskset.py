"""wikispeedia: navigate Wikipedia by clicking links to reach a target article.

A stateful tool example, authored as a `vf.Toolset` (`WikiToolset` in `servers/wiki.py`). The
taskset generates `(source, target)` pairs from the SNAP graph (loaded once, host-side) and, per
rollout, launches a toolset holding the navigation state (current article + path), built in its
`setup`. The harness calls `wiki_click_link` to move; the reward reads the toolset's
`TARGET REACHED` marker off the trace.
"""

import random
from collections.abc import Iterator

import verifiers.v1 as vf

from wikispeedia_v1.graph import WikiGraph, format_article
from wikispeedia_v1.servers.wiki import WikiToolset, WikiToolsetConfig

SYSTEM = (
    "This game is easy and fun: starting from the first Wikipedia article, reach the "
    "second one by following links. Each article ends with `Available links: ...` — "
    "those are the only links you may follow. Use the `wiki_click_link` tool to "
    "navigate, and `wiki_go_back` to undo. Think about which broader concepts connect "
    "the source to the target, and aim for the article most likely to link your "
    "destination."
)


def sample_pairs(
    wiki: WikiGraph, lo: int, hi: int, seed: int
) -> Iterator[tuple[str, str, int]]:
    """Yield unique `(source, target, dist)` tuples within the distance band, forever — the
    `(source, target)` space is effectively unbounded, so `seed` is the knob that reshuffles
    which pairs a run draws (in place of `--shuffle`)."""
    rng = random.Random(seed)
    articles = sorted(wiki.articles)
    seen: set[tuple[str, str]] = set()
    while True:
        s, t = rng.choice(articles), rng.choice(articles)
        if s == t or (s, t) in seen:
            continue
        dist = wiki.shortest_path_length(s, t)
        if dist is not None and lo <= dist <= hi:
            seen.add((s, t))
            yield s, t, dist


class WikiTask(vf.Task):
    source: str
    target: str
    shortest_path: int
    """The graph's shortest-path hop count from source to target."""


class WikispeediaConfig(vf.TasksetConfig):
    min_dist: int = 3
    max_dist: int = 8
    seed: int = 0
    """Seeds the pair sampler — change it to draw a different sample of pairs (the unbounded
    taskset's reproducible stand-in for `--shuffle`); how many are drawn is the eval's `-n`."""
    max_turns: int = 30
    tools: WikiToolsetConfig = WikiToolsetConfig()


class WikispeediaTaskset(vf.Taskset[WikiTask, WikispeediaConfig]):
    UNBOUNDED = True

    def load_tasks(self) -> Iterator[WikiTask]:
        wiki = WikiGraph.load(include_text=not self.config.tools.links_only)
        pairs = sample_pairs(
            wiki, self.config.min_dist, self.config.max_dist, self.config.seed
        )
        for i, (source, target, dist) in enumerate(pairs):
            yield WikiTask(
                idx=i,
                name=f"{source} -> {target}",
                source=source,
                target=target,
                shortest_path=dist,
                prompt=(
                    f"{SYSTEM}\n\nYour mission: {source} >> {target}\n\n"
                    f"Here is the starting article:\n\n"
                    f"{format_article(wiki, source, self.config.tools.links_only)}"
                ),
            )

    def tools(self, task: WikiTask) -> list[vf.Toolset]:
        return [WikiToolset(self.config.tools)]

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
