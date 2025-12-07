import asyncio
import concurrent.futures
import functools
import io
import os
import re
from time import perf_counter
from urllib.parse import urljoin, urlparse

import aiohttp

_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max(os.cpu_count() - 4, 1))


async def run_in_process(func, *args, **kwargs):
    """
    Run a CPU-bound or C-extension-heavy function in the process pool.
    Keeps the call sites simple and avoids threading issues in libraries.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(_process_pool, func, *args)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate a large text blob with a clear sentinel."""
    if len(text) > max_length:
        return text[:max_length] + "\n...\n[truncated]"
    return text


def looks_like_pdf(url: str, headers: dict, body: bytes) -> bool:
    """Heuristic PDF detector based on headers and magic bytes."""
    # Normalize headers
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    content_disposition = headers.get("Content-Disposition") or headers.get("content-disposition") or ""
    ct = content_type.lower()
    disp = content_disposition.lower()
    path = urlparse(url).path.lower()

    # Strip leading whitespace / nulls, then look for the PDF magic header
    stripped = body.lstrip(b"\r\n\t\f\x00")
    header_is_pdf = stripped.startswith(b"%PDF-")

    return (
        # Correct content-type
        "application/pdf" in ct
        or "application/x-pdf" in ct
        # Common mislabel for downloads, but bytes say "PDF"
        or ("application/octet-stream" in ct and header_is_pdf)
        # URL hints
        or path.endswith(".pdf")
        or ("filename=" in disp and ".pdf" in disp)
        # Fallback: magic bytes trump everything
        or header_is_pdf
    )


def pdf_to_text_bytes(pdf_bytes):
    import logging

    from pdfminer.high_level import extract_text

    # Silence pdfminer warnings for this call
    pm_logger = logging.getLogger("pdfminer")
    pm_logger.setLevel(logging.ERROR)

    with io.BytesIO(pdf_bytes) as f:
        return extract_text(f)


async def clean_text_to_markdown(text):
    # Light post-processing so it's LLM-friendly
    text = re.sub(r"[ \t]+\n", "\n", text)  # trim trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # join hyphenated line breaks
    text = text.replace("\f", "\n\n---\n\n")  # page breaks → hr
    return text.strip()


# ---- Main fetch-and-extract ----
async def fetch_llm_readable(url, timeout=30, headers=None, debug=False):
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=timeout_obj, headers=headers) as session:
        t0 = perf_counter()
        # There's a bunch of error catching in there already, but the web is hell and sometimes things slip through
        try:
            async with session.get(url) as r:
                try:
                    r.raise_for_status()
                except Exception as e:
                    if debug:
                        print(f"Error fetching {url}: {e}")
                    return {
                        "type": "error",
                        "content": f"Error fetching {url}: {e}",
                        "source": url,
                        "format": "error",
                    }

                if debug:
                    print(f"Fetched {url} in {perf_counter() - t0:.2f}s")

                # Read the raw bytes once; we’ll sniff them.
                raw_bytes = await r.read()
                headers_lower = {k: v for k, v in r.headers.items()}
                content_type = headers_lower.get("Content-Type", "").lower()

                # ---- First: detect true PDFs (even with wrong headers / URLs) ----
                is_pdf = looks_like_pdf(url, headers_lower, raw_bytes)
                if is_pdf:
                    try:
                        text = await run_in_process(pdf_to_text_bytes, raw_bytes)
                        md = await clean_text_to_markdown(text or "")
                        result = {
                            "type": "markdown",
                            "content": md,
                            "source": url,
                            "format": "pdf->text(pdfminer)",
                        }
                    except Exception:
                        is_pdf = False

                # PDF detection is not 100% reliable, so is_pdf can change even if it was True initially
                if not is_pdf:
                    # ---- Not a PDF → treat as (mostly) HTML with trafilatura ----
                    # Decode bytes safely
                    encoding = r.charset or "utf-8"
                    try:
                        html_text = raw_bytes.decode(encoding, errors="ignore")
                    except LookupError:
                        html_text = raw_bytes.decode("utf-8", errors="ignore")

                    # If it looks like a regular HTML page, check for embedded PDF
                    if "text/html" in content_type or "<html" in html_text.lower():
                        m = re.search(
                            r'(?:<embed|<iframe)[^>]+src=["\']([^"\']+\.pdf)[^"\']*["\']',
                            html_text,
                            re.I,
                        )
                        if m:
                            pdf_url = urljoin(url, m.group(1))
                            # Recursively handle the embedded PDF URL
                            return await fetch_llm_readable(pdf_url, timeout, headers, debug)

                    # Fallback: run trafilatura on whatever HTML-ish/text-ish content we have
                    import trafilatura

                    md = await run_in_process(
                        trafilatura.extract,
                        html_text,
                        output_format="markdown",
                        fast=True,
                    )
                    result = {
                        "type": "markdown",
                        "content": md or "",
                        "source": url,
                        "format": "html->trafilatura",
                    }

                if debug:
                    print(f"Rendered {url} in {perf_counter() - t0:.2f}s")
        except Exception as e:
            if debug:
                print(f"Error rendering {url}: {e}")
            result = {
                "type": "error",
                "content": f"Error rendering {url}: {e}",
                "source": url,
                "format": "error",
            }
        return result


_inflight: dict[str, asyncio.Future] = {}
_url_locks: dict[str, asyncio.Lock] = {}
url_cache: dict[str, str] = {}
_fetch_semaphore = asyncio.Semaphore(64)


async def _do_fetch_and_parse(url: str, debug: bool = False) -> str:
    async with _fetch_semaphore:
        # existing fetch_llm_readable work, but using shared session
        result = await fetch_llm_readable(url, debug=debug)
        return result["content"]


async def open_one(url: str, debug: bool = False) -> str:
    t0 = perf_counter()
    if (cached := url_cache.get(url)) is not None:
        if debug:
            print(f"Open one {url} from cache in {perf_counter() - t0:.2f}s")
        return cached

    # Single-flight: ensure only one coroutine does the work per URL
    flight = _inflight.get(url)
    if flight is None:
        fut = asyncio.get_event_loop().create_future()
        _inflight[url] = fut
        try:
            content = await _do_fetch_and_parse(url, debug=debug)
            content = truncate_text(content, 20_000)
            # write-through under url-specific lock
            lock = _url_locks.setdefault(url, asyncio.Lock())
            async with lock:
                url_cache.setdefault(url, content)
            fut.set_result(url_cache[url])
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            _inflight.pop(url, None)
        result = await fut
        if debug:
            print(f"Open one {url} from webpage in {perf_counter() - t0:.2f}s")
    else:
        # Someone else is fetching it; just await
        result = await flight
        if debug:
            print(f"Open one {url} from future in {perf_counter() - t0:.2f}s")
    return result
