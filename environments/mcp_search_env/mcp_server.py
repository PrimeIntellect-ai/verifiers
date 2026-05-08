import json
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-search-v1")
STOPWORDS = {
    "about",
    "answer",
    "bundled",
    "document",
    "exact",
    "find",
    "mcp",
    "the",
    "title",
    "tool",
    "tools",
    "use",
    "what",
    "with",
}


def load_documents() -> dict[str, dict[str, str]]:
    docs_dir = Path(__file__).with_name("docs")
    documents: dict[str, dict[str, str]] = {}
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text()
        title = path.stem
        body_lines = []
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                continue
            body_lines.append(line)
        documents[path.stem] = {
            "title": title,
            "content": "\n".join(body_lines).strip(),
        }
    if not documents:
        raise RuntimeError(f"No bundled MCP search documents found in {docs_dir}.")
    return documents


DOCUMENTS = load_documents()


@mcp.tool()
def search_documents(query: str) -> str:
    matches = []
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in STOPWORDS and len(token) > 2
    ]
    for document_id, document in DOCUMENTS.items():
        title_tokens = set(re.findall(r"[a-z0-9]+", document["title"].lower()))
        content_tokens = set(re.findall(r"[a-z0-9]+", document["content"].lower()))
        score = sum(
            (3 if token in title_tokens else 0) + (1 if token in content_tokens else 0)
            for token in tokens
        )
        if score:
            matches.append(
                {
                    "document_id": document_id,
                    "title": document["title"],
                    "score": score,
                }
            )
    matches.sort(key=lambda item: (-item["score"], item["title"]))
    return json.dumps(
        [
            {"document_id": item["document_id"], "title": item["title"]}
            for item in matches
        ]
        or [
            {"document_id": key, "title": value["title"]}
            for key, value in DOCUMENTS.items()
        ]
    )


@mcp.tool()
def read_document(document_id: str) -> str:
    if document_id not in DOCUMENTS:
        raise ValueError(f"Unknown document_id: {document_id}")
    document = DOCUMENTS[document_id]
    return f"{document['title']}\n\n{document['content']}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
