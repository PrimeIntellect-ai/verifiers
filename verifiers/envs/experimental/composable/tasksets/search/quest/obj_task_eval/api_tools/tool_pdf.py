"""Lightweight PDF parser shim for QUEST objective evaluation.

The upstream QUEST runtime uses a richer PDF parser. For the composable
verifiers port, URL retrieval is text-first and this parser is intentionally
minimal to avoid importing optional browser/PDF provider stacks at module load.
"""


class PDFParser:
    async def parse_url(self, url: str) -> str | None:
        return None

    def parse_bytes(self, data: bytes) -> str | None:
        return None
