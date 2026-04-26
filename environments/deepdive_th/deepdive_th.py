from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import aiohttp
from datasets import Dataset, load_dataset
from openai import AsyncOpenAI

import verifiers as vf

SERPER_API_URL = "https://google.serper.dev/search"
DEFAULT_DATASET_NAME = "zai-org/DeepDive"
DEFAULT_DATASET_SPLIT = "qa_rl"
METADATA_KEYS = ["source", "category", "difficulty", "context", "metadata"]
PROMPT_SUFFIX = "\nReason step by step using the given tools and provide the final answer in \\boxed{}."

logger = logging.getLogger(__name__)

LoadedSource = Dataset | Iterable[Mapping[str, Any]] | None
Source = LoadedSource | Callable[[], LoadedSource]


class SerperAPIError(vf.InfraError):
    pass


def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n...\n[truncated]"


def format_serper_results(data: dict[str, Any], num_results: int, query: str) -> str:
    sections: list[str] = []
    knowledge_graph = data.get("knowledgeGraph") or {}
    if isinstance(knowledge_graph, dict) and knowledge_graph:
        lines = []
        title = str(knowledge_graph.get("title") or "").strip()
        if title:
            lines.append(f"Knowledge Graph: {title}")
        description = str(knowledge_graph.get("description") or "").strip()
        if description:
            lines.append(description)
        attributes = knowledge_graph.get("attributes") or {}
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                text = str(value).strip()
                if text:
                    lines.append(f"{key}: {text}")
        if lines:
            sections.append("\n".join(lines))

    organic = data.get("organic") or []
    if isinstance(organic, list):
        for index, result in enumerate(organic[:num_results]):
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "").strip() or "Untitled"
            lines = [f"Result {index}: {title}"]
            link = str(result.get("link") or "").strip()
            if link:
                lines.append(f"URL: {link}")
            snippet = str(result.get("snippet") or "").strip()
            if snippet:
                lines.append(snippet)
            sections.append("\n".join(lines))

    people_also_ask = data.get("peopleAlsoAsk") or []
    if isinstance(people_also_ask, list):
        questions = []
        for item in people_also_ask[:3]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            entry = f"Q: {question}"
            answer = str(item.get("snippet") or "").strip()
            if answer:
                entry += f"\nA: {answer}"
            questions.append(entry)
        if questions:
            sections.append("People Also Ask:\n" + "\n".join(questions))

    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def format_search_results(queries: list[str], results: list[str]) -> str:
    separator = "\n\n" + "-" * 40 + "\n\n"
    return separator.join(
        f"Results for query `{query}`:\n\n{result}"
        for query, result in zip(queries, results, strict=True)
    )


def normalize_line_ranges(lines: object) -> list[tuple[int, int]]:
    if not isinstance(lines, list | tuple):
        return []
    if len(lines) == 2 and all(isinstance(item, int) for item in lines):
        lines = [lines]
    ranges: list[tuple[int, int]] = []
    for item in lines:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        start, end = item
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start > end:
            start, end = end, start
        if end < 0:
            continue
        ranges.append((max(start, 0), max(end, 0)))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def render_line_ranges(content: str, ranges: list[tuple[int, int]]) -> str:
    content_lines = content.splitlines()
    blocks = []
    for start, end in ranges:
        if not content_lines or start >= len(content_lines):
            blocks.append(f"L{start}..{end}: ")
            continue
        slice_end = min(end, len(content_lines) - 1)
        chunk = "\n".join(content_lines[start : slice_end + 1])
        blocks.append(f"L{start}..{end}:\n\n```txt\n{chunk}\n```")
    return "\n\n".join(blocks)


def compile_search_pattern(
    pattern: str | None,
) -> tuple[re.Pattern[str] | None, str | None]:
    if not pattern:
        return None, None
    try:
        return re.compile(pattern, re.IGNORECASE), None
    except re.error as e:
        return None, str(e)


def html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", "", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|tr|td|th)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tool_metrics(state: vf.State) -> dict[str, dict[str, int]]:
    bucket = state.get("deepdive_tool_metrics")
    if not isinstance(bucket, dict):
        bucket = {"calls": {}, "errors": {}}
        state["deepdive_tool_metrics"] = bucket
    return cast(dict[str, dict[str, int]], bucket)


def record_tool_call(state: vf.State, name: str) -> None:
    metrics = tool_metrics(state)
    metrics["calls"][name] = metrics["calls"].get(name, 0) + 1


def record_tool_error(state: vf.State, name: str) -> None:
    metrics = tool_metrics(state)
    metrics["errors"][name] = metrics["errors"].get(name, 0) + 1


def make_source(
    dataset_name: str,
    dataset_split: str,
    dataset_test_size: float,
    dataset_seed: int,
    eval_source: bool,
    finish_with_tool: bool,
) -> Callable[[], Dataset]:
    def source() -> Dataset:
        dataset = load_dataset(dataset_name, split=dataset_split)
        assert isinstance(dataset, Dataset)

        def map_row(row: dict[str, object], index: int) -> dict[str, object]:
            question = str(row.get("question") or row.get("prompt") or "")
            answer = row.get("answer", "")
            prompt = question if finish_with_tool else question + PROMPT_SUFFIX
            info = {"raw_question": question}
            for key in METADATA_KEYS:
                if key in row:
                    info[key] = row[key]
            return {
                "example_id": index,
                "prompt": [{"role": "user", "content": prompt}],
                "answer": answer,
                "info": info,
            }

        mapped = dataset.map(
            map_row, with_indices=True, remove_columns=dataset.column_names
        )
        if not dataset_test_size:
            return mapped
        split = mapped.train_test_split(test_size=dataset_test_size, seed=dataset_seed)
        return cast(Dataset, split["test" if eval_source else "train"])

    return source


def make_deepdive_tools(
    serper_api_key: str,
    max_search_results: int,
    max_response_chars: int,
    serper_timeout: float,
    finish_with_tool: bool,
) -> list[object]:
    max_response_chars = max(1, int(max_response_chars))

    async def search_one(query: str, num_results: int) -> str:
        query = query.strip()
        if not query:
            return ""
        payload = {"q": query}
        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=serper_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                SERPER_API_URL, headers=headers, json=payload
            ) as response:
                content = await response.text()
                if response.status >= 400:
                    raise SerperAPIError(
                        f"Serper API error {response.status}: {content.strip()}"
                    )
        data = json.loads(content)
        limit = max(1, min(int(num_results), max_search_results))
        return truncate_text(
            format_serper_results(data, limit, query), max_response_chars
        )

    async def search_web(
        queries: list[str],
        state: vf.State,
        num_results_per_query: int = 3,
    ) -> str:
        """Search Google with up to 10 queries in parallel."""
        record_tool_call(state, "search_web")
        if not isinstance(queries, list) or any(
            not isinstance(q, str) for q in queries
        ):
            record_tool_error(state, "search_web")
            return "Error: `queries` must be a list of strings."
        clean_queries = [q.strip() for q in queries if q.strip()][:10]
        if not clean_queries:
            return ""
        try:
            results = await asyncio.gather(
                *[search_one(q, num_results_per_query) for q in clean_queries]
            )
            return format_search_results(clean_queries, results)
        except Exception:
            record_tool_error(state, "search_web")
            raise

    async def fetch_url(url: str) -> dict[str, object]:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                raw = await response.text(errors="replace")
                content_type = response.headers.get("content-type", "")
                content = html_to_text(raw) if "html" in content_type else raw
                return {
                    "url": str(response.url),
                    "status": response.status,
                    "content_type": content_type,
                    "content": content,
                }

    async def scan_page(
        url: str,
        state: vf.State,
        pattern: str | None = None,
        context_lines: int = 0,
        max_matches: int = 200,
    ) -> str:
        """Get page metadata and optional regex matches without returning the full page."""
        record_tool_call(state, "scan_page")
        try:
            result = await fetch_url(url)
            content = str(result.get("content") or "")
            compiled, pattern_error = compile_search_pattern(pattern)
            lines = [
                f"source: {result.get('url') or url}",
                f"status: {result.get('status')}",
                f"content_type: {result.get('content_type')}",
                f"char_count: {len(content)}",
                f"line_count: {len(content.splitlines())}",
            ]
            if pattern is not None:
                lines.append(f"pattern: {pattern}")
                if pattern_error:
                    record_tool_error(state, "scan_page")
                    lines.append(f"pattern_error: {pattern_error}")
                else:
                    matches = [
                        (idx, line)
                        for idx, line in enumerate(content.splitlines())
                        if compiled and compiled.search(line)
                    ][: max(0, int(max_matches))]
                    if matches:
                        lines.append("matches:")
                        for idx, line in matches:
                            lines.append(f"L{idx}: {line}")
                        if context_lines > 0:
                            ranges = normalize_line_ranges(
                                [
                                    [idx - context_lines, idx + context_lines]
                                    for idx, _ in matches
                                ]
                            )
                            lines.append("context_blocks:")
                            lines.append(render_line_ranges(content, ranges))
                    else:
                        lines.append("matches: (none)")
            return truncate_text("\n".join(lines), max_response_chars)
        except Exception:
            record_tool_error(state, "scan_page")
            raise

    async def open_lines(
        url: str,
        state: vf.State,
        lines: list[list[int]] | list[int] | None = None,
    ) -> str:
        """Open a URL and optionally return 0-based inclusive line ranges."""
        record_tool_call(state, "open_lines")
        try:
            result = await fetch_url(url)
            content = str(result.get("content") or "")
            if lines is not None:
                content = render_line_ranges(content, normalize_line_ranges(lines))
            return truncate_text(content or "(no content)", max_response_chars)
        except Exception:
            record_tool_error(state, "open_lines")
            raise

    async def finish(final_answer: str, state: vf.State) -> str:
        """Provide the final answer to the task. Stops execution."""
        record_tool_call(state, "finish")
        state["deepdive_final_answer"] = final_answer
        state["done"] = True
        return final_answer

    tools: list[object] = [search_web, scan_page, open_lines]
    if finish_with_tool:
        tools.append(finish)
    return tools


def make_deepdive_rubric(
    judge_model: str,
    judge_base_url: str | None,
    redundancy_penalty_weight: float,
    finish_with_tool: bool,
) -> vf.Rubric:
    judge_client = (
        AsyncOpenAI(base_url=judge_base_url) if judge_base_url else AsyncOpenAI()
    )
    parser = vf.MaybeThinkParser(extract_fn=vf.extract_boxed_answer)
    judge = vf.JudgeRubric(
        parser=parser,
        judge_client=judge_client,
        judge_model=judge_model,
    )

    async def judge_reward(
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
    ) -> float:
        response = state.get("deepdive_final_answer") if finish_with_tool else None
        if not isinstance(response, str) or not response:
            response = parser.parse_answer(completion) or ""
        judge_response = await judge.judge(
            prompt=prompt,
            completion=[vf.AssistantMessage(content=response)],
            answer=answer,
            state=state,
        )
        reward = 1.0 if "yes" in judge_response.lower() else 0.0
        state["judge_reward"] = reward
        return reward

    def iter_search_queries(completion: vf.Messages) -> Iterable[list[str]]:
        for message in completion:
            if not isinstance(message, vf.AssistantMessage):
                continue
            for tool_call in message.tool_calls or []:
                if tool_call.name != "search_web":
                    continue
                try:
                    arguments = json.loads(tool_call.arguments or "{}")
                except json.JSONDecodeError:
                    continue
                raw_queries = arguments.get("queries", [])
                if isinstance(raw_queries, str):
                    raw_queries = [raw_queries]
                if isinstance(raw_queries, list):
                    yield [
                        str(q).strip().lower() for q in raw_queries if str(q).strip()
                    ]

    async def redundancy_penalty(completion: vf.Messages) -> float:
        query_sets = [
            set(" ".join(qs).split()) for qs in iter_search_queries(completion)
        ]
        query_sets = [s for s in query_sets if s]
        if len(query_sets) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i, left in enumerate(query_sets):
            for right in query_sets[i + 1 :]:
                total += len(left & right) / len(left | right)
                pairs += 1
        return total / pairs if pairs else 0.0

    async def search_web_mean_queries(completion: vf.Messages) -> float:
        counts = [len(queries) for queries in iter_search_queries(completion)]
        return float(sum(counts) / len(counts)) if counts else 0.0

    def tool_error_rate_func(tool_name: str):
        async def tool_error_rate(state: vf.State) -> float:
            metrics = state.get("deepdive_tool_metrics")
            if not isinstance(metrics, dict):
                return 0.0
            calls = metrics.get("calls", {})
            errors = metrics.get("errors", {})
            total_calls = int(calls.get(tool_name, 0)) if isinstance(calls, dict) else 0
            total_errors = (
                int(errors.get(tool_name, 0)) if isinstance(errors, dict) else 0
            )
            return float(total_errors / total_calls) if total_calls else 0.0

        tool_error_rate.__name__ = f"{tool_name}_error_rate"
        return tool_error_rate

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(judge_reward)
    rubric.add_reward_func(redundancy_penalty, weight=-redundancy_penalty_weight)
    rubric.add_metric(search_web_mean_queries)
    for tool_name in ("search_web", "scan_page", "open_lines", "finish"):
        if tool_name != "finish" or finish_with_tool:
            rubric.add_metric(tool_error_rate_func(tool_name))
    return rubric


def load_taskset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    dataset_test_size: float = 0.1,
    dataset_seed: int = 2025,
    serper_api_key_var: str = "SERPER_API_KEY",
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str | None = None,
    max_search_results: int = 10,
    max_response_chars: int = 20_000,
    serper_timeout: float = 15.0,
    redundancy_penalty_weight: float = 0.0,
    finish_with_tool: bool = True,
    log_level: str = "INFO",
) -> vf.Taskset:
    logger.setLevel(getattr(logging, log_level.upper()))
    vf.ensure_keys([serper_api_key_var])
    serper_api_key = os.environ[serper_api_key_var]
    return vf.Taskset(
        source=make_source(
            dataset_name,
            dataset_split,
            dataset_test_size,
            dataset_seed,
            eval_source=False,
            finish_with_tool=finish_with_tool,
        ),
        eval_source=make_source(
            dataset_name,
            dataset_split,
            dataset_test_size,
            dataset_seed,
            eval_source=True,
            finish_with_tool=finish_with_tool,
        ),
        rubric=make_deepdive_rubric(
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            redundancy_penalty_weight=redundancy_penalty_weight,
            finish_with_tool=finish_with_tool,
        ),
        tools=make_deepdive_tools(
            serper_api_key=serper_api_key,
            max_search_results=max_search_results,
            max_response_chars=max_response_chars,
            serper_timeout=serper_timeout,
            finish_with_tool=finish_with_tool,
        ),
        name="deepdive-th",
    )


def load_harness(max_turns: int = 32) -> vf.Harness:
    return vf.Harness(
        run=vf.RunConfig(max_turns=max_turns, stop_errors=(SerperAPIError,))
    )


def load_environment(
    max_turns: int = 32,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    dataset_test_size: float = 0.1,
    dataset_seed: int = 2025,
    serper_api_key_var: str = "SERPER_API_KEY",
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str | None = None,
    max_search_results: int = 10,
    max_response_chars: int = 20_000,
    serper_timeout: float = 15.0,
    redundancy_penalty_weight: float = 0.0,
    finish_with_tool: bool = True,
    log_level: str = "INFO",
) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dataset_test_size=dataset_test_size,
            dataset_seed=dataset_seed,
            serper_api_key_var=serper_api_key_var,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            max_search_results=max_search_results,
            max_response_chars=max_response_chars,
            serper_timeout=serper_timeout,
            redundancy_penalty_weight=redundancy_penalty_weight,
            finish_with_tool=finish_with_tool,
            log_level=log_level,
        ),
        harness=load_harness(max_turns=max_turns),
    )
