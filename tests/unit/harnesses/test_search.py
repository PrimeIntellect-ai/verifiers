"""Under a network blocklist, the bash search tool drops results that link through Google's
encrypted `/goto` redirects, whose destination is unknown."""

import httpx

from verifiers.v1.harnesses.utils import core
from verifiers.v1.utils.network import network_rule_matches

GOTO = "https://www.google.com/goto?url=token"


def test_search_drops_google_redirect_links_under_a_blocklist(monkeypatch):
    organic = [
        {"link": GOTO, "title": "redirected", "snippet": "hidden destination"},
        {"link": "https://allowed.test/a", "title": "allowed", "snippet": "visible"},
    ]
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: httpx.Response(
            200, json={"organic": organic}, request=httpx.Request("POST", url)
        ),
    )
    # The bundled program gets this helper from the flat script, not an import.
    monkeypatch.setattr(
        core, "network_rule_matches", network_rule_matches, raising=False
    )

    blocked = core.run_search("q", "key", 5, ["example.org"])
    unrestricted = core.run_search("q", "key", 5)

    assert blocked == "Result 1: allowed\nURL: https://allowed.test/a\n  - visible"
    assert unrestricted.startswith(f"Result 1: redirected\nURL: {GOTO}\n")
