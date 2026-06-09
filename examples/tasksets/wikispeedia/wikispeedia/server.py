"""Wikispeedia tool server: navigate the article graph by clicking links.

Launched by the harness as a host subprocess per rollout. It holds the rollout's
navigation state (current article + path) in memory, reads the source/target from
the env the Environment injects, and serves `click_link`/`go_back` over streamable
HTTP. On reaching the target it emits a `TARGET REACHED` marker in the tool
result — which lands in the trace, where the reward reads it.
"""

import os

from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf
from wikispeedia.graph import WikiGraph, format_article

wiki = WikiGraph.load()
SOURCE = os.environ["WIKISPEEDIA_SOURCE"]
TARGET = os.environ["WIKISPEEDIA_TARGET"]
LINKS_ONLY = os.environ.get("WIKISPEEDIA_LINKS_ONLY", "1") == "1"
path = [SOURCE]

mcp = FastMCP("wiki")


@mcp.tool()
def click_link(article: str) -> str:
    """Navigate to a linked article (must be an available link from the current one)."""
    current = path[-1]
    available = wiki.get_links(current)
    target = wiki.normalize_name(article)
    if target is None or target not in available:
        return (
            f"'{article}' is not a valid link from '{current}'.\n"
            f"Available links: {', '.join(available) or '(none)'}"
        )
    path.append(target)
    page = format_article(wiki, target, LINKS_ONLY)
    if target == TARGET:
        return f"TARGET REACHED 🎯 You navigated to the target '{TARGET}'.\n\n{page}"
    return page


@mcp.tool()
def go_back() -> str:
    """Go back to the previous article (undo the last click)."""
    if len(path) <= 1:
        return "You are already at the starting article. Cannot go back."
    path.pop()
    return format_article(wiki, path[-1], LINKS_ONLY)


vf.run_mcp_server(mcp)
