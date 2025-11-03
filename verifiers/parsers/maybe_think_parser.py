from typing import Callable

from verifiers.parsers.parser import Parser


class MaybeThinkParser(Parser):
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        super().__init__(extract_fn=extract_fn)

    def parse(self, text: str) -> str:
        text = text.split("</think>")[-1].strip()
        return self.extract_fn(text)
