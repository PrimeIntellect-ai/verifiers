"""Small URL normalization helpers needed by the vendored QUEST evaluator."""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse, urldefrag, unquote


def remove_utm_parameters(url: str) -> str:
    parsed = urlparse(url)
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.lower().startswith("utm_")
        ]
    )
    return urlunparse(parsed._replace(query=query))


def normalize_url_simple(url: str) -> str:
    url_no_frag, _ = urldefrag(url.strip())
    decoded = unquote(url_no_frag)
    if decoded.endswith("/") and len(decoded) > 1 and not decoded.endswith("://"):
        decoded = decoded[:-1]
    return remove_utm_parameters(decoded)


def normalize_url_for_browser(url: str) -> str:
    return normalize_url_simple(url)
