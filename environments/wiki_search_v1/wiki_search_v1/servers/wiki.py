"""wiki-search tool server: a `vf.Toolset` exposing read-only search/view/read tools.

Self-launching — the framework starts it with `python -m wiki_search_v1.servers.wiki`.
"""

import verifiers.v1 as vf

DATASET = "willcb/rare-wiki-pages"


class WikiSearchToolset(vf.Toolset[vf.ToolsetConfig]):
    """Read-only search/view/read over the wiki corpus. The corpus + chroma index (expensive) are
    built once in `setup`, in the server process. Every tool call is a read."""

    TOOL_PREFIX = "wiki"  # the model sees `wiki_search_pages` / `wiki_view_sections` / `wiki_read_section`

    async def setup(self) -> None:
        import os
        from pathlib import Path

        # chromadb (via opentelemetry) imports protobuf-generated code predating protobuf 6.x;
        # force the pure-Python impl so it imports next to a newer protobuf (e.g. vllm's).
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        import chromadb
        from datasets import load_dataset

        rows = load_dataset(DATASET, split="train")
        self.pages = {
            r["id"]: {"title": r["title"], "content": r["content"]} for r in rows
        }
        cache = os.environ.get(
            "WIKI_SEARCH_CACHE", str(Path.home() / ".cache" / "wiki_search")
        )
        col = chromadb.PersistentClient(
            path=f"{cache}/chroma"
        ).get_or_create_collection(
            "wiki_titles"  # default local embedder (onnx MiniLM) — no API key
        )
        ids = list(self.pages)
        have: set[str] = set()
        for i in range(0, len(ids), 500):
            have.update(col.get(ids=ids[i : i + 500]).get("ids", []))
        missing = [pid for pid in ids if pid not in have]
        for i in range(0, len(missing), 256):
            batch = missing[i : i + 256]
            col.upsert(ids=batch, documents=[self.pages[pid]["title"] for pid in batch])
        self.collection = col

    @staticmethod
    def _norm(text: str) -> str:
        """lowercase, spaces -> underscores (section ids)."""
        return text.strip().lower().replace(" ", "_")

    @vf.tool
    def search_pages(self, query: str) -> list[dict]:
        """Search the wiki for the 10 most relevant pages (by title). Returns
        `[{page_id, title}, ...]` — pass a page_id to view_sections."""
        result = self.collection.query(query_texts=[query], n_results=10)
        return [
            {"page_id": pid, "title": self.pages[pid]["title"]}
            for pid in result["ids"][0]
        ]

    @vf.tool
    def view_sections(self, page_id: str) -> list[dict]:
        """List a page's sections. Returns `[{section_id, section_name}, ...]` — pass a
        section_id to read_section."""
        content = self.pages[page_id]["content"]
        sections = [
            {
                "section_id": f"{page_id}:{self._norm(line.lstrip('#').strip())}",
                "section_name": line.lstrip("#").strip(),
            }
            for line in content.split("\n")
            if line.startswith("#")
        ]
        return sections or [
            {"section_id": f"{page_id}:full", "section_name": "Full Page"}
        ]

    @vf.tool
    def read_section(self, section_id: str) -> str:
        """Read the text of a section (`page_id:section`; use `page_id:full` for all)."""
        page_id, section = section_id.split(":", 1)
        content = self.pages[page_id]["content"]
        if section == "full":
            return content
        lines = content.split("\n")
        start = end = None
        for i, line in enumerate(lines):
            if line.startswith("#"):
                if self._norm(line.lstrip("#").strip()) == section and start is None:
                    start = i
                elif start is not None:
                    end = i
                    break
        if start is None:
            return f"Section not found: {section_id}"
        return "\n".join(lines[start : end or len(lines)])


if __name__ == "__main__":
    WikiSearchToolset.run()
