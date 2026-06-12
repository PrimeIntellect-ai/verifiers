"""wikispeedia: navigate Wikipedia by clicking links to reach a target article.

A stateful tool example. The taskset generates `(source, target)` pairs from the
SNAP graph (loaded once, host-side) and, per rollout, launches a host-side server
holding the navigation state (see `server.py`). The harness calls `wiki_click_link`
to move; the reward reads the server's `TARGET REACHED` marker off the trace. The
corpus stays host-side — nothing is shipped into the runtime.
"""

import random
import sys

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
    # NON-COLOCATED, NON-SHARED: the server runs in its own runtime (`tools.runtime`,
    # default subprocess/host), one per rollout, reached via the harness's runtime
    # (localhost or a tunnel, auto-resolved). Not colocated — its ~100MB graph
    # shouldn't be re-fetched inside a fresh harness runtime every rollout.
    tools: vf.ToolsConfig = vf.ToolsConfig(colocated=False)
    num_tasks: int = 20
    min_dist: int = 3
    max_dist: int = 8
    seed: int = 0
    max_turns: int = 30
    links_only: bool = True
    """Show only each article's outgoing-link menu (no prose) — the classic
    Wikispeedia formulation; keeps the prompt small."""


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

    def tools(self, task: WikiTask) -> list[vf.Tools]:
        return [
            vf.Tools(
                name="wiki",
                command=[sys.executable, "-m", "wikispeedia_v1.server"],
                env={
                    "WIKISPEEDIA_SOURCE": task.source,
                    "WIKISPEEDIA_TARGET": task.target,
                    "WIKISPEEDIA_LINKS_ONLY": "1" if self.config.links_only else "0",
                },
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
