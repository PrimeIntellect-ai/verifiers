"""Wiki-search tool server: read-only search/view/read over the wiki corpus.

Launched by the harness as a host subprocess per rollout. Connects to the chroma
index + corpus (host-side, see corpus.py) and serves `search_pages`,
`view_sections`, `read_section` over streamable HTTP. Stateless — every call is a
read.
"""

from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf
from wiki_search_v1.corpus import collection, corpus, normalize_id

PAGES = corpus()
COLLECTION = collection()

mcp = FastMCP("wiki")


@mcp.tool()
def search_pages(query: str) -> list[dict]:
    """Search the wiki for the 10 most relevant pages (by title). Returns
    `[{page_id, title}, ...]` — pass a page_id to view_sections."""
    result = COLLECTION.query(query_texts=[query], n_results=10)
    return [{"page_id": pid, "title": PAGES[pid]["title"]} for pid in result["ids"][0]]


@mcp.tool()
def view_sections(page_id: str) -> list[dict]:
    """List a page's sections. Returns `[{section_id, section_name}, ...]` — pass a
    section_id to read_section."""
    content = PAGES[page_id]["content"]
    sections = [
        {
            "section_id": f"{page_id}:{normalize_id(line.lstrip('#').strip())}",
            "section_name": line.lstrip("#").strip(),
        }
        for line in content.split("\n")
        if line.startswith("#")
    ]
    return sections or [{"section_id": f"{page_id}:full", "section_name": "Full Page"}]


@mcp.tool()
def read_section(section_id: str) -> str:
    """Read the text of a section (`page_id:section`; use `page_id:full` for all)."""
    page_id, section = section_id.split(":", 1)
    content = PAGES[page_id]["content"]
    if section == "full":
        return content
    lines = content.split("\n")
    start = end = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            if normalize_id(line.lstrip("#").strip()) == section and start is None:
                start = i
            elif start is not None:
                end = i
                break
    if start is None:
        return f"Section not found: {section_id}"
    return "\n".join(lines[start : end or len(lines)])


vf.run_mcp_server(mcp)
