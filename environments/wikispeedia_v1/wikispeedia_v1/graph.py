"""Wikispeedia article graph: download, parse, and query the SNAP dataset.

Stdlib-only (urllib + tarfile). The first load pulls two tarballs from Stanford
SNAP (~100 MB) into ~/.cache/wikispeedia and parses the article texts, the link
adjacency, and the precomputed shortest-path distance matrix.
"""

from __future__ import annotations

import logging
import os
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

SNAP_BASE = "https://snap.stanford.edu/data/wikispeedia"
GRAPH_TAR = "wikispeedia_paths-and-graph.tar.gz"
ARTICLES_TAR = "wikispeedia_articles_plaintext.tar.gz"
DEFAULT_CACHE_DIR = Path(
    os.environ.get("WIKISPEEDIA_CACHE_DIR", str(Path.home() / ".cache" / "wikispeedia"))
)
GRAPH_SUBDIR = "wikispeedia_paths-and-graph"
ARTICLES_SUBDIR = "plaintext_articles"


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` atomically (via a .part file)."""
    if dest.exists():
        return
    part = dest.with_suffix(dest.suffix + ".part")
    part.unlink(missing_ok=True)
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, part)
    part.rename(dest)


def _ensure_data(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_tar, articles_tar = cache_dir / GRAPH_TAR, cache_dir / ARTICLES_TAR
    _download(f"{SNAP_BASE}/{GRAPH_TAR}", graph_tar)
    _download(f"{SNAP_BASE}/{ARTICLES_TAR}", articles_tar)
    graph_dir, articles_dir = cache_dir / GRAPH_SUBDIR, cache_dir / ARTICLES_SUBDIR
    if not graph_dir.exists():
        with tarfile.open(graph_tar, "r:gz") as tar:
            tar.extractall(cache_dir, filter="data")
    if not articles_dir.exists():
        with tarfile.open(articles_tar, "r:gz") as tar:
            tar.extractall(cache_dir, filter="data")
    return graph_dir, articles_dir


def _parse_tsv_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [s for line in f if (s := line.strip()) and not s.startswith("#")]


def _load_articles(graph_dir: Path, articles_dir: Path) -> dict[str, str]:
    articles: dict[str, str] = {}
    for name in _parse_tsv_lines(graph_dir / "articles.tsv"):
        text_path = articles_dir / f"{name}.txt"
        if text_path.exists():
            articles[name] = text_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
    return articles


def _load_links(graph_dir: Path, valid: set[str]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {name: [] for name in valid}
    for line in _parse_tsv_lines(graph_dir / "links.tsv"):
        parts = line.split("\t")
        if len(parts) == 2 and parts[0] in valid and parts[1] in valid:
            adj[parts[0]].append(parts[1])
    return adj


def _load_distance_matrix(
    graph_dir: Path, names: list[str]
) -> dict[str, dict[str, int]]:
    """Each row is single-digit distances ('_' = unreachable), one char per target."""
    distances: dict[str, dict[str, int]] = {}
    for i, row in enumerate(
        _parse_tsv_lines(graph_dir / "shortest-path-distance-matrix.txt")
    ):
        distances[names[i]] = {
            names[j]: int(ch) for j, ch in enumerate(row) if ch != "_"
        }
    return distances


class WikiGraph:
    """The Wikispeedia article graph backed by the SNAP dataset."""

    def __init__(self, articles, links, distances) -> None:
        self.articles = articles
        self.links = links
        self.distances = distances
        self._name_lookup = {name.lower(): name for name in articles}

    @classmethod
    def load(cls, cache_dir: Path | None = None) -> WikiGraph:
        graph_dir, articles_dir = _ensure_data(cache_dir or DEFAULT_CACHE_DIR)
        articles = _load_articles(graph_dir, articles_dir)
        valid = set(articles)
        links = _load_links(graph_dir, valid)
        order = [n for n in _parse_tsv_lines(graph_dir / "articles.tsv") if n in valid]
        distances = _load_distance_matrix(graph_dir, order)
        return cls(articles, links, distances)

    def get_links(self, article: str) -> list[str]:
        return sorted(self.links.get(article, []))

    def shortest_path_length(self, source: str, target: str) -> int | None:
        return self.distances.get(source, {}).get(target)

    def normalize_name(self, name: str) -> str | None:
        """Match a user-provided name to a canonical article name."""
        if name in self.articles:
            return name
        underscored = name.replace(" ", "_")
        if underscored in self.articles:
            return underscored
        return self._name_lookup.get(name.lower()) or self._name_lookup.get(
            underscored.lower()
        )


def format_article(wiki: WikiGraph, article: str, links_only: bool = True) -> str:
    """Render an article for the harness: its outgoing-link menu, optionally preceded by
    the article text. `links_only` (the default) keeps the prompt small and makes the
    task purely about the link graph — the classic Wikispeedia formulation."""
    links = wiki.get_links(article)
    links_str = ", ".join(links) if links else "(no outgoing links)"
    body = "" if links_only else f"{wiki.articles[article]}\n\n---\n"
    return f"# {article}\n\n{body}Available links: {links_str}"
