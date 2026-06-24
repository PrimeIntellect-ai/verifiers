"""Web-search tools for s1-deepresearch-v1: `web_search` (Serper) + `web_visit` (fetch).

A stateless, read-only `vf.Toolset` — runs `shared` (one server for the whole eval). Both
tools need `SERPER_API_KEY` (search) / network access (visit) in the server process env;
the default host (`subprocess`) runtime inherits the eval process's environment.
"""

from __future__ import annotations

import html
import io
import logging
import os
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

import verifiers.v1 as vf

SERPER_URL = "https://google.serper.dev/search"
_PDF_HEADER = b"%PDF-"


class WebSearchConfig(vf.ToolsetConfig):
    num_results: int = 5
    """Organic Serper results to include per search."""
    search_timeout: float = 45.0
    visit_timeout: float = 30.0
    max_search_chars: int = 8192
    max_visit_chars: int = 50000


def _truncate(output: str, max_output: int) -> str:
    if max_output <= 0 or len(output) <= max_output:
        return output if max_output > 0 else ""
    marker = f"\n... [output truncated, {len(output)} chars total] ...\n"
    if len(marker) >= max_output:
        return marker[:max_output]
    remaining = max_output - len(marker)
    head = remaining // 2
    tail = remaining - head
    return output[:head] + marker + (output[-tail:] if tail else "")


def _format_serper_results(data: dict, query: str, num_results: int) -> str:
    sections: list[str] = []
    kg = data.get("knowledgeGraph")
    if kg:
        kg_lines: list[str] = []
        title = (kg.get("title") or "").strip()
        if title:
            kg_lines.append(f"Knowledge Graph: {title}")
        description = (kg.get("description") or "").strip()
        if description:
            kg_lines.append(description)
        for key, value in (kg.get("attributes") or {}).items():
            text = str(value).strip()
            if text:
                kg_lines.append(f"{key}: {text}")
        if kg_lines:
            sections.append("\n".join(kg_lines))
    for i, result in enumerate((data.get("organic") or [])[:num_results]):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(snippet)
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def _looks_like_html(ct: str, body: bytes) -> bool:
    if "text/html" in ct or "application/xhtml+xml" in ct:
        return True
    sample = body[:2048].lstrip().lower()
    return sample.startswith((b"<!doctype html", b"<html")) or b"<html" in sample[:512]


def _looks_like_pdf(url: str, headers: dict[str, str], body: bytes) -> bool:
    ct = (headers.get("content-type") or headers.get("Content-Type") or "").lower()
    disp = (
        headers.get("content-disposition") or headers.get("Content-Disposition") or ""
    ).lower()
    if body.startswith(_PDF_HEADER) or "application/pdf" in ct or "application/x-pdf" in ct:
        return True
    if _looks_like_html(ct, body):
        return False
    path = url.split("?", 1)[0].lower()
    ambiguous = not ct or "octet-stream" in ct or "application/download" in ct
    return ambiguous and (path.endswith(".pdf") or ("filename=" in disp and ".pdf" in disp))


def _pdf_to_text(pdf_bytes: bytes) -> str:
    from pdfminer.high_level import extract_text

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with io.BytesIO(pdf_bytes) as f:
        return extract_text(f) or ""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in {"br", "p", "div", "li", "tr", "td", "th", "hr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"}:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in {"p", "div", "li", "tr", "td", "th"}:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _html_to_text(html_text: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return ""
    text = html.unescape(parser.get_text()).replace("\xa0", " ")
    return re.sub(r"[ \t]{2,}", " ", text)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = text.replace("\f", "\n\n---\n\n")
    return text.strip()


async def _fetch_page(url: str, timeout: float) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            body = response.content
            ct = (response.headers.get("content-type") or "").lower()
            if _looks_like_pdf(url, dict(response.headers), body):
                try:
                    return _clean(_pdf_to_text(body))
                except Exception as e:
                    return f"Error parsing PDF {url}: {e}"
            text = body.decode(response.encoding or "utf-8", errors="ignore")
            if "text/html" in ct or "<html" in text.lower():
                embed = re.search(
                    r'(?:<embed|<iframe)[^>]+src=["\']([^"\']+\.pdf)[^"\']*["\']', text, re.I
                )
                if embed:
                    return await _fetch_page(urljoin(url, embed.group(1)), timeout)
                return _clean(_html_to_text(text))
            return _clean(text)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else "?"
        return f"Error fetching {url}: HTTP {status}"
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error rendering {url}: {e}"


class WebSearchToolset(vf.Toolset[WebSearchConfig]):
    """Read-only web search + page fetch. The model sees `web_search` and `web_visit`."""

    TOOL_PREFIX = "web"

    @vf.tool
    async def search(self, query: str) -> str:
        """Search Google via Serper for a query and return the top results (title, URL,
        snippet). Use it to find sources, then `web_visit` a URL to read the full page."""
        api_key = os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            return "Error: SERPER_API_KEY is not set in the tool server environment."
        try:
            async with httpx.AsyncClient(timeout=self.config.search_timeout) as client:
                resp = await client.post(
                    SERPER_URL,
                    json={"q": query},
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            return f"Serper search error ({e.response.status_code}): {body[:500]}"
        except Exception as e:
            return f"Error searching for {query!r}: {e}"
        formatted = _format_serper_results(data, query, self.config.num_results)
        return _truncate(f'Results for query "{query}":\n\n{formatted}', self.config.max_search_chars)

    @vf.tool
    async def visit(self, url: str) -> str:
        """Fetch a URL and return its readable text content (handles HTML and PDF). Use it
        to read a page found via `web_search`."""
        text = await _fetch_page(url, self.config.visit_timeout)
        return _truncate(text, self.config.max_visit_chars)


if __name__ == "__main__":
    WebSearchToolset.run()
