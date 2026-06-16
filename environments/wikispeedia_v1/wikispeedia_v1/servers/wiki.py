import verifiers.v1 as vf


class WikiToolsetConfig(vf.ToolsetConfig):
    links_only: bool = True
    """Genuine config (CLI-tunable): show only each article's outgoing-link menu (no prose) — the
    classic Wikispeedia formulation; keeps the prompt small. The per-task source/target are read
    off the task in `setup`, not config."""


class WikiToolset(vf.Toolset[WikiToolsetConfig]):
    """Holds the rollout's navigation state (current article + path) and serves
    `click_link`/`go_back`. On reaching the target it emits a `TARGET REACHED` marker in the
    tool result — which lands in the trace, where the reward reads it. `setup` pulls + parses
    the SNAP article/link graph itself (stdlib only, cached on disk)."""

    TOOL_PREFIX = "wiki"  # the model sees `wiki_click_link` / `wiki_go_back`

    async def setup(self) -> None:
        import os
        import tarfile
        import urllib.request
        from pathlib import Path

        cache = Path(
            os.environ.get(
                "WIKISPEEDIA_CACHE_DIR", str(Path.home() / ".cache" / "wikispeedia")
            )
        )
        cache.mkdir(parents=True, exist_ok=True)
        base = "https://snap.stanford.edu/data/wikispeedia"
        for tar_name, subdir in [
            ("wikispeedia_paths-and-graph.tar.gz", "wikispeedia_paths-and-graph"),
            ("wikispeedia_articles_plaintext.tar.gz", "plaintext_articles"),
        ]:
            tar = cache / tar_name
            if not tar.exists():
                urllib.request.urlretrieve(f"{base}/{tar_name}", f"{tar}.part")
                os.rename(f"{tar}.part", tar)
            if not (cache / subdir).exists():
                with tarfile.open(tar, "r:gz") as t:
                    t.extractall(cache, filter="data")
        gdir, adir = cache / "wikispeedia_paths-and-graph", cache / "plaintext_articles"

        def rows(path):
            with open(path, encoding="utf-8") as f:
                return [s for ln in f if (s := ln.strip()) and not s.startswith("#")]

        self.articles = {}
        for name in rows(gdir / "articles.tsv"):
            text = adir / f"{name}.txt"
            if text.exists():
                self.articles[name] = text.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
        self.links = {name: [] for name in self.articles}
        for line in rows(gdir / "links.tsv"):
            src, _, dst = line.partition("\t")
            if dst in self.articles and src in self.links:
                self.links[src].append(dst)
        self._lookup = {name.lower(): name for name in self.articles}

    async def setup_task(self, task) -> None:
        self.target = task.target  # per-task input, from the task
        self.path = [
            task.source
        ]  # per-task + mutable nav state (current article is path[-1])

    def _normalize(self, name: str) -> str | None:
        if name in self.articles:
            return name
        underscored = name.replace(" ", "_")
        if underscored in self.articles:
            return underscored
        return self._lookup.get(name.lower()) or self._lookup.get(underscored.lower())

    def _article(self, name: str) -> str:
        links = ", ".join(sorted(self.links.get(name, []))) or "(no outgoing links)"
        body = "" if self.config.links_only else f"{self.articles[name]}\n\n---\n"
        return f"# {name}\n\n{body}Available links: {links}"

    @vf.tool
    def click_link(self, article: str) -> str:
        """Navigate to a linked article (must be an available link from the current one)."""
        current = self.path[-1]
        available = sorted(self.links.get(current, []))
        target = self._normalize(article)
        if target is None or target not in available:
            return (
                f"'{article}' is not a valid link from '{current}'.\n"
                f"Available links: {', '.join(available) or '(none)'}"
            )
        self.path.append(target)
        page = self._article(target)
        if target == self.target:
            return f"TARGET REACHED 🎯 You navigated to the target '{self.target}'.\n\n{page}"
        return page

    @vf.tool
    def go_back(self) -> str:
        """Go back to the previous article (undo the last click)."""
        if len(self.path) <= 1:
            return "You are already at the starting article. Cannot go back."
        self.path.pop()
        return self._article(self.path[-1])


if __name__ == "__main__":
    WikiToolset.run()
