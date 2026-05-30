"""Wikispeedia article graph: download, parse, and query the SNAP dataset."""

import logging
import os
import random
import shutil
import tarfile
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(f"verifiers.{__name__}")

SNAP_BASE = "https://snap.stanford.edu/data/wikispeedia"
GRAPH_TAR = "wikispeedia_paths-and-graph.tar.gz"
ARTICLES_TAR = "wikispeedia_articles_plaintext.tar.gz"
DEFAULT_CACHE_DIR = Path(
    os.environ.get("WIKISPEEDIA_CACHE_DIR", str(Path.home() / ".cache" / "wikispeedia"))
)
GRAPH_SUBDIR = "wikispeedia_paths-and-graph"
ARTICLES_SUBDIR = "plaintext_articles"


def _download(url: str, dest: Path) -> None:
    """Download url to dest."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{dest.name}.", suffix=".part", dir=dest.parent
    )
    os.close(fd)
    tmp = Path(tmp_str)
    logger.info("Downloading %s ...", url)
    try:
        urllib.request.urlretrieve(url, tmp)
        try:
            os.rename(tmp, dest)
        except OSError:
            # Another worker won the race. Their dest must be present.
            if not dest.exists():
                raise
    finally:
        # If rename succeeded the tmp is gone; this is just for failure paths.
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    logger.info("Saved to %s", dest)


def _ensure_data(cache_dir: Path) -> tuple[Path, Path]:
    """Download and extract both tarballs. Returns (graph_dir, articles_dir)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    graph_tar = cache_dir / GRAPH_TAR
    articles_tar = cache_dir / ARTICLES_TAR
    _download(f"{SNAP_BASE}/{GRAPH_TAR}", graph_tar)
    _download(f"{SNAP_BASE}/{ARTICLES_TAR}", articles_tar)

    graph_dir = cache_dir / GRAPH_SUBDIR
    articles_dir = cache_dir / ARTICLES_SUBDIR

    _extract_atomic(graph_tar, graph_dir, cache_dir, GRAPH_TAR)
    _extract_atomic(articles_tar, articles_dir, cache_dir, ARTICLES_TAR)

    return graph_dir, articles_dir


def _extract_atomic(tar_path: Path, dest_dir: Path, parent: Path, label: str) -> None:
    """Extract tar_path so dest_dir only appears with complete contents."""
    fence = dest_dir / ".extraction_complete"
    if fence.exists():
        return
    if dest_dir.exists():
        # Stale partial extraction. Wipe and retry.
        shutil.rmtree(dest_dir)
    parent.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s ...", label)
    tmp_root = Path(tempfile.mkdtemp(prefix=f".{dest_dir.name}.", dir=parent))
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_root, filter="data")
        extracted = tmp_root / dest_dir.name
        if not extracted.exists():
            raise RuntimeError(
                f"{label}: expected wrapping directory '{dest_dir.name}' inside tarball"
            )
        (extracted / ".extraction_complete").touch()
        try:
            os.rename(extracted, dest_dir)
        except OSError:
            # Another worker won the race. Trust their result iff complete.
            if not (dest_dir / ".extraction_complete").exists():
                raise
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _parse_tsv_lines(path: Path) -> list[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def _load_articles(graph_dir: Path, articles_dir: Path) -> dict[str, str]:
    """Load article names from the graph TSV, texts from the plaintext directory."""
    article_names = _parse_tsv_lines(graph_dir / "articles.tsv")
    articles: dict[str, str] = {}

    for name in article_names:
        text_path = articles_dir / f"{name}.txt"
        if text_path.exists():
            articles[name] = text_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        else:
            logger.debug("No text file for article: %s", name)

    logger.info("Loaded %d / %d articles with text.", len(articles), len(article_names))
    return articles


def _load_links(graph_dir: Path, valid: set[str]) -> dict[str, list[str]]:
    """Parse links.tsv into an adjacency list, keeping only valid articles."""
    adj: dict[str, list[str]] = {name: [] for name in valid}
    for line in _parse_tsv_lines(graph_dir / "links.tsv"):
        parts = line.split("\t")
        if len(parts) == 2:
            src, tgt = parts
            if src in valid and tgt in valid:
                adj[src].append(tgt)
    total_edges = sum(len(v) for v in adj.values())
    logger.info("Loaded links: %d nodes, %d edges.", len(adj), total_edges)
    return adj


def _load_distance_matrix(
    graph_dir: Path, valid: set[str]
) -> dict[str, dict[str, int]]:
    """Load the precomputed shortest-path distance matrix.

    Each row is a string of single-digit distances (or '_' for unreachable),
    one character per target article, in the same order as articles.tsv.

    The matrix axis is the FULL articles.tsv (i.e. the original SNAP order),
    not the filtered set. We re-read articles.tsv to recover that ordering and
    then look up names against valid — so this loader stays correct even
    when a partial article cache makes valid smaller than the matrix.
    """
    full_axis = _parse_tsv_lines(graph_dir / "articles.tsv")
    rows = _parse_tsv_lines(graph_dir / "shortest-path-distance-matrix.txt")
    if len(rows) != len(full_axis):
        logger.warning(
            "Distance matrix row count (%d) does not match articles.tsv (%d); "
            "trimming to the intersection.",
            len(rows),
            len(full_axis),
        )
    n = min(len(rows), len(full_axis))
    distances: dict[str, dict[str, int]] = {}
    for i in range(n):
        src = full_axis[i]
        if src not in valid:
            continue
        row = rows[i]
        row_dists: dict[str, int] = {}
        for j, ch in enumerate(row):
            if j >= n or ch == "_":
                continue
            tgt = full_axis[j]
            if tgt in valid:
                row_dists[tgt] = int(ch)
        distances[src] = row_dists
    logger.info("Loaded distance matrix for %d articles.", len(distances))
    return distances


HumanStats = dict[str, float | int | None]
WikiPair = tuple[str, str, int]


def load_wiki_graph(cache_dir: str | Path | None = None) -> "WikiGraph":
    cache_key = str(Path(cache_dir).expanduser()) if cache_dir is not None else ""
    return cached_wiki_graph(cache_key)


@lru_cache(maxsize=None)
def cached_wiki_graph(cache_key: str) -> "WikiGraph":
    cache_dir = Path(cache_key) if cache_key else None
    return WikiGraph.load(cache_dir=cache_dir)


def _load_human_stats(graph_dir: Path) -> dict[tuple[str, str], HumanStats]:
    """Aggregate human-play stats per (source, target) pair from SNAP paths_*.tsv.

    Each entry exposes:
      - human_attempts: total finished + unfinished plays (>= 1)
      - human_success_rate: finished / attempts, in [0, 1]
      - human_avg_rating: mean self-reported rating (1=easy, 5=brutal), or None
        if no rater submitted one for this pair
    """
    from collections import defaultdict

    raw: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"finished": 0, "unfinished": 0, "ratings": []}
    )

    finished_path = graph_dir / "paths_finished.tsv"
    if finished_path.exists():
        for line in _parse_tsv_lines(finished_path):
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            _, _, _dur, path_str, rating = parts
            nodes = [n for n in path_str.split(";") if n != "<"]
            if len(nodes) < 2:
                continue
            src, tgt = nodes[0], nodes[-1]
            s = raw[(src, tgt)]
            s["finished"] += 1
            if rating != "NULL":
                try:
                    s["ratings"].append(int(rating))
                except ValueError:
                    pass

    unfinished_path = graph_dir / "paths_unfinished.tsv"
    if unfinished_path.exists():
        for line in _parse_tsv_lines(unfinished_path):
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            _, _, _dur, path_str, target = parts[:5]
            nodes = [n for n in path_str.split(";") if n != "<"]
            if not nodes:
                continue
            raw[(nodes[0], target)]["unfinished"] += 1

    out: dict[tuple[str, str], HumanStats] = {}
    for key, s in raw.items():
        attempts = s["finished"] + s["unfinished"]
        if attempts == 0:
            continue
        success_rate = s["finished"] / attempts
        avg_rating = sum(s["ratings"]) / len(s["ratings"]) if s["ratings"] else None
        out[key] = {
            "human_attempts": attempts,
            "human_success_rate": round(success_rate, 3),
            "human_avg_rating": round(avg_rating, 2)
            if avg_rating is not None
            else None,
        }

    logger.info("Loaded human play stats for %d pairs.", len(out))
    return out


class WikiGraph:
    """The Wikispeedia article graph backed by the SNAP dataset."""

    def __init__(
        self,
        articles: dict[str, str],
        links: dict[str, list[str]],
        distances: dict[str, dict[str, int]],
        human_stats: dict[tuple[str, str], HumanStats] | None = None,
    ):
        self.articles = articles
        self.links = links
        self.distances = distances
        self.human_stats = human_stats or {}
        self._name_lookup: dict[str, str] = {name.lower(): name for name in articles}

    @classmethod
    def load(cls, cache_dir: Path | None = None) -> "WikiGraph":
        cache_dir = cache_dir or DEFAULT_CACHE_DIR
        graph_dir, articles_dir = _ensure_data(cache_dir)

        articles = _load_articles(graph_dir, articles_dir)
        valid = set(articles.keys())
        links = _load_links(graph_dir, valid)
        distances = _load_distance_matrix(graph_dir, valid)
        human_stats = _load_human_stats(graph_dir)
        return cls(
            articles=articles, links=links, distances=distances, human_stats=human_stats
        )

    def get_text(self, article: str) -> str:
        return self.articles[article]

    def get_links(self, article: str) -> list[str]:
        return sorted(self.links.get(article, []))

    def get_human_stats(self, source: str, target: str) -> HumanStats | None:
        """Return aggregated human-play stats for a pair, or None if no plays."""
        return self.human_stats.get((source, target))

    def split_pairs(
        self,
        train_size: int,
        eval_size: int,
        min_dist: int,
        max_dist: int,
        eval_target_fraction: float,
        seed: int,
        stratify: bool = True,
    ) -> tuple[list[WikiPair], list[WikiPair]]:
        """Random train/eval split with disjoint target articles."""
        articles = sorted(self.articles)
        rng = random.Random(seed)
        shuffled = articles.copy()
        rng.shuffle(shuffled)
        n_eval_targets = max(int(len(articles) * eval_target_fraction), 1)
        eval_targets = shuffled[:n_eval_targets]
        train_targets = shuffled[n_eval_targets:]
        train = self.sample_pairs(
            sources=articles,
            targets=train_targets,
            n=train_size,
            min_dist=min_dist,
            max_dist=max_dist,
            seed=seed + 1,
            stratify=stratify,
        )
        eval_ = self.sample_pairs(
            sources=articles,
            targets=eval_targets,
            n=eval_size,
            min_dist=min_dist,
            max_dist=max_dist,
            seed=seed + 2,
            stratify=stratify,
        )
        return train, eval_

    def sample_pairs(
        self,
        sources: list[str],
        targets: list[str],
        n: int,
        min_dist: int,
        max_dist: int,
        seed: int,
        stratify: bool = True,
    ) -> list[WikiPair]:
        """Sample (source, target, shortest_path) tuples in the distance band.

        With `stratify=True`, every valid pair is bucketed by shortest path
        and sampled evenly across non-empty buckets. With `stratify=False`,
        accept-reject sampling mirrors the graph's natural distribution.
        """
        rng = random.Random(seed)

        if not stratify:
            seen: set[tuple[str, str]] = set()
            pairs: list[WikiPair] = []
            max_attempts = n * 100
            for _ in range(max_attempts):
                if len(pairs) >= n:
                    break
                s = rng.choice(sources)
                t = rng.choice(targets)
                if s == t or (s, t) in seen:
                    continue
                d = self.shortest_path_length(s, t)
                if d is None or not (min_dist <= d <= max_dist):
                    continue
                pairs.append((s, t, d))
                seen.add((s, t))
            return pairs

        targets_set = set(targets)
        by_bucket: dict[int, list[WikiPair]] = {
            d: [] for d in range(min_dist, max_dist + 1)
        }
        for s in sources:
            row = self.distances.get(s)
            if not row:
                continue
            for t, d in row.items():
                if s == t or t not in targets_set:
                    continue
                if min_dist <= d <= max_dist:
                    by_bucket[d].append((s, t, d))

        bucket_sizes = {d: len(b) for d, b in by_bucket.items()}
        n_nonempty_buckets = sum(1 for v in bucket_sizes.values() if v > 0)
        per_bucket_target = n // n_nonempty_buckets if n_nonempty_buckets else 0
        nonzero_sizes = [v for v in bucket_sizes.values() if v > 0]
        per_bucket = min([per_bucket_target] + nonzero_sizes) if nonzero_sizes else 0

        sampled: list[WikiPair] = []
        for d in sorted(by_bucket):
            pool = by_bucket[d]
            if not pool:
                continue
            rng.shuffle(pool)
            sampled.extend(pool[:per_bucket])
        rng.shuffle(sampled)

        logger.info(
            "Stratified sample: requested=%d, per-bucket-target=%d, "
            "per-bucket-actual=%d, available=%s, returned=%d",
            n,
            per_bucket_target,
            per_bucket,
            bucket_sizes,
            len(sampled),
        )
        return sampled

    def shortest_path_length(self, source: str, target: str) -> int | None:
        return self.distances.get(source, {}).get(target)

    def normalize_name(self, name: str) -> str | None:
        """Match a user-provided name to a canonical article name."""
        if name in self.articles:
            return name
        with_underscores = name.replace(" ", "_")
        if with_underscores in self.articles:
            return with_underscores
        return self._name_lookup.get(name.lower()) or self._name_lookup.get(
            with_underscores.lower()
        )
