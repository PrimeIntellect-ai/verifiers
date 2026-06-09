"""The wiki-search corpus: a HuggingFace dataset of wiki pages + a chroma index.

Loads the full corpus (id -> title/content) and builds a persistent chroma
collection over the page titles for semantic search. Embeddings use chroma's
default local model (onnx MiniLM) — no API key, no per-query cost. The index is
built once (cached on disk) and reused; the tool server connects to it per rollout.
"""

import os
from pathlib import Path

# chromadb (via opentelemetry) imports protobuf-generated code that predates
# protobuf 6.x; on a venv where a newer protobuf is co-installed (e.g. alongside
# vllm), force the pure-Python impl so importing it doesn't fail. A no-op
# (setdefault) where the stubs already match; set before chromadb is imported below.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

DATASET = "willcb/rare-wiki-pages"
CACHE = Path(
    os.environ.get("WIKI_SEARCH_CACHE", str(Path.home() / ".cache" / "wiki_search"))
)

_corpus: dict[str, dict[str, str]] | None = None
_collection = None


def normalize_id(text: str) -> str:
    """lowercase, spaces -> underscores (used for section ids)."""
    return text.strip().lower().replace(" ", "_")


def corpus() -> dict[str, dict[str, str]]:
    """The full corpus as `{page_id: {"title", "content"}}` (loaded once)."""
    global _corpus
    if _corpus is None:
        from datasets import load_dataset

        rows = load_dataset(DATASET, split="train")
        _corpus = {
            r["id"]: {"title": r["title"], "content": r["content"]} for r in rows
        }
    return _corpus


def collection():
    """A persistent chroma collection over the page titles (built once, reused)."""
    global _collection
    if _collection is None:
        import chromadb

        pages = corpus()
        client = chromadb.PersistentClient(path=str(CACHE / "chroma"))
        col = client.get_or_create_collection("wiki_titles")  # default local embedder
        ids = list(pages)
        existing: set[str] = set()
        for i in range(0, len(ids), 500):
            existing.update(col.get(ids=ids[i : i + 500]).get("ids", []))
        missing = [pid for pid in ids if pid not in existing]
        for i in range(0, len(missing), 256):
            batch = missing[i : i + 256]
            col.upsert(ids=batch, documents=[pages[pid]["title"] for pid in batch])
        _collection = col
    return _collection
