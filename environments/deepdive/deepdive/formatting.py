from __future__ import annotations

from typing import Any


def truncate_text(text: str, max_length: int) -> str:
    """Truncate a large text blob with a clear sentinel."""
    if len(text) > max_length:
        return text[:max_length] + "\n...\n[truncated]"
    return text


def format_serper_results(data: dict[str, Any], num_results: int, query: str) -> str:
    """
    Render Serper 'search' JSON into a plain-text block that is concise and model-friendly.

    Expected Serper schema slices:
      - knowledgeGraph?: { title, description?, attributes? }
      - organic?: [{ title, link, snippet?, siteLinks? }, ...]
      - peopleAlsoAsk?: [{ question, snippet, title, link }, ...]
    """
    sections: list[str] = []

    knowledge_graph = data.get("knowledgeGraph") or {}
    if knowledge_graph:
        kg_lines = []
        title = (knowledge_graph.get("title") or "").strip()
        if title:
            kg_lines.append(f"Knowledge Graph: {title}")
        description = (knowledge_graph.get("description") or "").strip()
        if description:
            kg_lines.append(description)
        attributes = knowledge_graph.get("attributes") or {}
        for key, value in attributes.items():
            text = str(value).strip()
            if text:
                kg_lines.append(f"{key}: {text}")
        if kg_lines:
            sections.append("\n".join(kg_lines))

    for index, result in enumerate((data.get("organic") or [])[:num_results]):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {index}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(snippet)
        sections.append("\n".join(lines))

    people_also_ask = data.get("peopleAlsoAsk") or []
    if people_also_ask:
        max_questions = max(1, min(3, len(people_also_ask)))
        questions = []
        for item in people_also_ask[:max_questions]:
            question = (item.get("question") or "").strip()
            if not question:
                continue
            entry = f"Q: {question}"
            answer = (item.get("snippet") or "").strip()
            if answer:
                entry += f"\nA: {answer}"
            questions.append(entry)
        if questions:
            sections.append("People Also Ask:\n" + "\n".join(questions))

    if not sections:
        return f"No results returned for query: {query}"

    return "\n\n---\n\n".join(sections)
