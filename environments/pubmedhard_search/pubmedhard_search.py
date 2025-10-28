import re
from math import log
from typing import Any, Dict, List, Optional

import verifiers as vf
from datasets import Dataset, load_dataset
 


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class RetrievalIndex:
    def __init__(self, dataset: Dataset, context_fields: tuple[str, ...] = ("context_list", "context")):
        self.N = 0
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, Dict[int, int]] = {}
        self.doc_len: Dict[int, int] = {}
        self.docs_text: Dict[int, str] = {}

        def extract_context(ex: Dict[str, Any]) -> str:
            for f in context_fields:
                if f in ex and ex[f] is not None:
                    v = ex[f]
                    if isinstance(v, list):
                        return "\n\n".join(str(x) for x in v)
                    return str(v)
            return ""

        total_len = 0
        for i, ex in enumerate(dataset):
            text = extract_context(ex)
            self.docs_text[i] = text
            tokens = _tokenize(text)
            self.N += 1
            self.doc_len[i] = len(tokens)
            total_len += len(tokens)
            freqs: Dict[str, int] = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            for tok, c in freqs.items():
                if tok not in self.postings:
                    self.postings[tok] = {}
                self.postings[tok][i] = c
                self.df[tok] = self.df.get(tok, 0) + 1
        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0

    def _idf(self, tok: str) -> float:
        df = self.df.get(tok, 0)
        return log(((self.N - df + 0.5) / (df + 0.5)) + 1.0) if self.N > 0 else 0.0

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[int]:
        if not query:
            return []
        q_tokens = _tokenize(query)
        scores: Dict[int, float] = {}
        for tok in q_tokens:
            if tok not in self.postings:
                continue
            idf = self._idf(tok)
            for doc_id, tf in self.postings[tok].items():
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + k1 * (1 - b + b * (dl / self.avgdl if self.avgdl > 0 else 0))
                score = idf * ((tf * (k1 + 1)) / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [doc for doc, _ in ranked[:k]]


class TokenF1Rubric(vf.Rubric):
    def __init__(self, **kwargs: Any):
        super().__init__(funcs=[self._f1_reward], weights=[1.0], **kwargs)

    async def _f1_reward(self, completion: vf.Messages, answer: str, **_: Any) -> float:
        text = completion.get_last_text() if hasattr(completion, "get_last_text") else str(completion)
        if not text or not answer:
            return 0.0
        p = set(_tokenize(text))
        g = set(_tokenize(answer))
        if not p or not g:
            return 0.0
        tp = len(p & g)
        prec = tp / len(p)
        rec = tp / len(g)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)


def load_environment(
    max_turns: int = 10,
    dataset_name: str = "casperhansen/pmc-oa-markdown-qa",
    dataset_split: str = "train",
) -> vf.Environment:
    ds = load_dataset(dataset_name, split=dataset_split)
    index = RetrievalIndex(ds)

    # tools
    async def search(query: str, k: int = 5) -> list[dict]:
        return [{"idx": i} for i in index.search(query, k)]

    async def read(idx: int) -> str:
        i = int(idx)
        return index.docs_text.get(i, "")

    tools = [search, read]
    parser = vf.Parser()
    tool_rubric = vf.ToolRubric(tools=tools)
    f1_rubric = TokenF1Rubric()
    rubric = vf.RubricGroup(rubrics=[tool_rubric, f1_rubric])

    system_prompt = "Use search/read tools over PubMed OA Markdown to answer concisely."
    vf_env = vf.ToolEnv(
        dataset=ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=tools,
        max_turns=max_turns,
    )
    return vf_env


__all__ = ["load_environment"]


