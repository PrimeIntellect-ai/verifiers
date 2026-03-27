from __future__ import annotations

import asyncio
import glob
import inspect
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from verifiers.synth.prompts import (
    BACKTRANSLATE_PROMPT,
    FILTER_JUDGE_PROMPT,
    PLAN_PROMPT,
    render_env_spec,
)
from verifiers.synth.types import (
    BuildResult,
    EnvSpec,
    SynthConfig,
    SynthPlan,
    SynthSample,
)

if TYPE_CHECKING:
    from datasets import Dataset

    from verifiers.envs.environment import Environment

logger = logging.getLogger(__name__)

PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "openai": {"url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},
    "anthropic": {"url": "https://api.anthropic.com", "key": "ANTHROPIC_API_KEY"},
    "deepseek": {"url": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY"},
    "prime": {"url": "https://api.pinference.ai/api/v1", "key": "PRIME_API_KEY"},
}
DEFAULT_PROVIDER = "openai"

_MAX_CONCURRENT_LLM_CALLS = 16


def _parse_model(model_str: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model). Bare names default to openai."""
    if "/" in model_str:
        provider, model = model_str.split("/", 1)
        if provider not in PROVIDER_CONFIGS:
            return DEFAULT_PROVIDER, model_str
        return provider, model
    return DEFAULT_PROVIDER, model_str


def _make_client(model_str: str) -> tuple[AsyncOpenAI, str]:
    """Create an AsyncOpenAI client from a provider/model string."""
    import os

    provider, model = _parse_model(model_str)
    cfg = PROVIDER_CONFIGS[provider]
    api_key = os.getenv(cfg["key"], "EMPTY")
    return AsyncOpenAI(api_key=api_key, base_url=cfg["url"]), model


class SynthDataBuilder:
    """Generates synthetic datasets from seed content using an Environment's spec."""

    def __init__(
        self,
        env: Environment,
        generator_model: str = "gpt-4.1",
        filter_model: str = "gpt-4.1",
    ):
        self.env = env
        self.env_spec = self._extract_env_spec(env)
        self.generator_model = generator_model
        self.filter_model = filter_model

    def _extract_env_spec(self, env: Environment) -> EnvSpec:
        reward_fns: list[dict[str, Any]] = []
        funcs = env.rubric._get_reward_funcs()
        weights = env.rubric._get_reward_weights()
        for func, weight in zip(funcs, weights):
            reward_fns.append(
                {
                    "name": getattr(func, "__name__", repr(func)),
                    "doc": inspect.getdoc(func),
                    "weight": weight,
                }
            )

        tools = None
        if env.tool_defs:
            tools = [t.model_dump() for t in env.tool_defs]

        example_rows: list[dict[str, Any]] = []
        dataset_schema: dict[str, str] = {}
        try:
            ds = env.get_dataset(n=3)
            if ds is not None and len(ds) > 0:
                dataset_schema = {
                    col: str(ds.features[col]) for col in ds.column_names
                }
                for i in range(min(3, len(ds))):
                    example_rows.append(dict(ds[i]))
        except Exception:
            logger.debug("Could not extract example rows from environment dataset")

        parser_info = None
        if env.parser and type(env.parser).__name__ != "Parser":
            parser_info = repr(env.parser)

        return EnvSpec(
            env_type=type(env).__name__,
            system_prompt=env.system_prompt,
            tools=tools,
            max_turns=getattr(env, "max_turns", 1),
            reward_functions=reward_fns,
            dataset_schema=dataset_schema,
            example_rows=example_rows,
            parser_info=parser_info,
            few_shot=env.few_shot,
        )

    async def build(
        self,
        seeds: list[str] | Dataset | None = None,
        samples_per_subtopic: int = 5,
        subtopic_branches: int = 3,
        filter_mode: str = "standard",
        filter_threshold: float = 0.8,
        filter_ceiling: float = 0.2,
    ) -> BuildResult:
        config = SynthConfig(
            generator_model=self.generator_model,
            filter_model=self.filter_model,
            samples_per_subtopic=samples_per_subtopic,
            subtopic_branches=subtopic_branches,
            filter_mode=filter_mode,  # type: ignore[arg-type]
            filter_threshold=filter_threshold,
            filter_ceiling=filter_ceiling,
        )
        normalized = self._resolve_seeds(seeds)
        logger.info("Normalized %d seeds", len(normalized))

        plan = await self._plan(normalized, config)
        logger.info(
            "Plan: %d seeds, %d total target samples", len(plan.seeds), plan.total_target
        )

        raw = await self._generate(plan, config)
        logger.info("Generated %d raw samples", len(raw))

        filtered = await self._filter(raw, normalized, config)
        logger.info("Filtered to %d samples", len(filtered))

        stats = self._compute_stats(raw, filtered, config)
        return BuildResult(raw_samples=raw, filtered_samples=filtered, stats=stats)

    # ------------------------------------------------------------------
    # Seed resolution & normalization
    # ------------------------------------------------------------------

    def _resolve_seeds(self, seeds: list[str] | Any | None) -> list[dict[str, Any]]:
        """Resolve seeds: use explicit seeds if provided, else pull from env dataset."""
        if seeds is not None:
            return self._normalize_seeds(seeds)

        try:
            ds = self.env.get_dataset()
            if ds is not None and len(ds) > 0:
                logger.info("Using environment dataset as seeds (%d rows)", len(ds))
                return _normalize_dataset_seeds(ds)
        except Exception:
            logger.debug("Could not load dataset from environment")

        raise ValueError(
            "No seeds provided and the environment has no dataset. "
            "Pass seeds explicitly via build(seeds=...)."
        )

    @staticmethod
    def _normalize_seeds(seeds: list[str] | Any) -> list[dict[str, Any]]:
        """Unify document paths, raw strings, and HF Datasets into [{id, content}]."""
        try:
            from datasets import Dataset

            if isinstance(seeds, Dataset):
                return _normalize_dataset_seeds(seeds)
        except ImportError:
            pass

        if not isinstance(seeds, list):
            raise TypeError(f"seeds must be list[str] or Dataset, got {type(seeds)}")

        result: list[dict[str, Any]] = []
        idx = 0
        for item in seeds:
            if not isinstance(item, str):
                try:
                    from datasets import Dataset

                    if isinstance(item, Dataset):
                        result.extend(_normalize_dataset_seeds(item, offset=idx))
                        idx = len(result)
                        continue
                except ImportError:
                    pass
                raise TypeError(f"Unexpected seed type: {type(item)}")

            expanded = glob.glob(item)
            if expanded:
                for path_str in sorted(expanded):
                    p = Path(path_str)
                    if p.is_file():
                        content = p.read_text(encoding="utf-8", errors="replace")
                        result.append({"id": str(idx), "content": content})
                        idx += 1
            else:
                result.append({"id": str(idx), "content": item})
                idx += 1

        return result

    # ------------------------------------------------------------------
    # Stage 1: Planning
    # ------------------------------------------------------------------

    async def _plan(
        self, seeds: list[dict[str, Any]], config: SynthConfig
    ) -> SynthPlan:
        gen_client, gen_model = _make_client(config.generator_model)
        spec_text = render_env_spec(asdict(self.env_spec))
        sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)

        async def plan_one(seed: dict[str, Any]) -> dict[str, Any]:
            prompt = PLAN_PROMPT.format(
                env_spec=spec_text,
                seed_content=seed["content"][:8000],
                num_subtopics=config.subtopic_branches,
                samples_per_subtopic=config.samples_per_subtopic,
            )
            async with sem:
                resp = await gen_client.chat.completions.create(
                    model=gen_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
            text = (resp.choices[0].message.content or "").strip()
            subtopics = _parse_json_array(text, fallback_count=config.subtopic_branches)
            return {**seed, "subtopics": subtopics}

        planned = await asyncio.gather(*[plan_one(s) for s in seeds])
        total = sum(
            len(s["subtopics"]) * config.samples_per_subtopic for s in planned
        )
        return SynthPlan(seeds=list(planned), total_target=total)

    # ------------------------------------------------------------------
    # Stage 2: Back-translation + fan-out
    # ------------------------------------------------------------------

    async def _generate(
        self, plan: SynthPlan, config: SynthConfig
    ) -> list[SynthSample]:
        gen_client, gen_model = _make_client(config.generator_model)
        spec_text = render_env_spec(asdict(self.env_spec))
        sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)

        async def gen_one(
            seed: dict[str, Any], subtopic: str
        ) -> list[SynthSample]:
            samples: list[SynthSample] = []
            for _ in range(config.samples_per_subtopic):
                prompt = BACKTRANSLATE_PROMPT.format(
                    env_spec=spec_text,
                    seed_content=seed["content"][:8000],
                    subtopic=subtopic,
                    seed_id=seed["id"],
                )
                async with sem:
                    resp = await gen_client.chat.completions.create(
                        model=gen_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.9,
                    )
                text = (resp.choices[0].message.content or "").strip()
                parsed = _parse_json_object(text)
                if parsed and "question" in parsed and "answer" in parsed:
                    samples.append(
                        SynthSample(
                            question=parsed["question"],
                            answer=parsed["answer"],
                            info=parsed.get("info", {}),
                            seed_id=seed["id"],
                            subtopic=subtopic,
                        )
                    )
                else:
                    logger.warning("Failed to parse sample from LLM response")
            return samples

        tasks = []
        for seed in plan.seeds:
            for subtopic in seed.get("subtopics", []):
                tasks.append(gen_one(seed, subtopic))

        results = await asyncio.gather(*tasks)
        return [s for batch in results for s in batch]

    # ------------------------------------------------------------------
    # Stage 3: Filtering
    # ------------------------------------------------------------------

    async def _filter(
        self,
        samples: list[SynthSample],
        seeds: list[dict[str, Any]],
        config: SynthConfig,
    ) -> list[SynthSample]:
        if not samples:
            return []

        filter_client, filter_model = _make_client(config.filter_model)
        seed_map = {s["id"]: s["content"] for s in seeds}
        system_prompt = self.env_spec.system_prompt or ""
        sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)

        async def attempt_answer(question: str, context: str | None) -> str:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                user_content = (
                    f"Reference material:\n{context[:6000]}\n\n---\n\n{question}"
                )
            else:
                user_content = question
            messages.append({"role": "user", "content": user_content})
            async with sem:
                resp = await filter_client.chat.completions.create(
                    model=filter_model,
                    messages=messages,
                    temperature=0.0,
                )
            return (resp.choices[0].message.content or "").strip()

        async def judge(question: str, golden_answer: str, response: str) -> float:
            prompt = FILTER_JUDGE_PROMPT.format(
                question=question,
                golden_answer=golden_answer,
                response=response,
            )
            async with sem:
                resp = await filter_client.chat.completions.create(
                    model=filter_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
            text = (resp.choices[0].message.content or "").strip()
            parsed = _parse_json_object(text)
            if parsed:
                return float(parsed.get("score", 0.0))
            return 0.0

        if config.filter_mode == "standard":
            scored = await asyncio.gather(
                *[self._filter_standard(s, attempt_answer, judge) for s in samples]
            )
            return [
                s for s in scored
                if (s.score_with_context or 0.0) >= config.filter_threshold
            ]

        scored = await asyncio.gather(
            *[
                self._filter_icl(s, seed_map, attempt_answer, judge)
                for s in samples
            ]
        )
        return [
            s
            for s in scored
            if (s.score_with_context or 0.0) >= config.filter_threshold
            and (s.score_without_context or 0.0) <= config.filter_ceiling
        ]

    @staticmethod
    async def _filter_standard(
        sample: SynthSample,
        attempt_answer: Any,
        judge: Any,
    ) -> SynthSample:
        """Standard: verify a frontier model can solve the task (no context)."""
        resp = await attempt_answer(sample.question, None)
        score = await judge(sample.question, sample.answer, resp)
        sample.score_with_context = score
        return sample

    @staticmethod
    async def _filter_icl(
        sample: SynthSample,
        seed_map: dict[str, str],
        attempt_answer: Any,
        judge: Any,
    ) -> SynthSample:
        """ICL-calibrated: model should succeed WITH context, fail WITHOUT."""
        seed_content = seed_map.get(sample.seed_id)

        resp_with = await attempt_answer(sample.question, seed_content)
        score_with = await judge(sample.question, sample.answer, resp_with)
        sample.score_with_context = score_with

        resp_without = await attempt_answer(sample.question, None)
        score_without = await judge(sample.question, sample.answer, resp_without)
        sample.score_without_context = score_without

        return sample

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(
        raw: list[SynthSample],
        filtered: list[SynthSample],
        config: SynthConfig,
    ) -> dict[str, Any]:
        total_gen = len(raw)
        total_filt = len(filtered)

        per_subtopic: dict[str, dict[str, int]] = {}
        for s in raw:
            per_subtopic.setdefault(s.subtopic, {"generated": 0, "filtered": 0})
            per_subtopic[s.subtopic]["generated"] += 1
        for s in filtered:
            per_subtopic.setdefault(s.subtopic, {"generated": 0, "filtered": 0})
            per_subtopic[s.subtopic]["filtered"] += 1

        return {
            "total_generated": total_gen,
            "total_filtered": total_filt,
            "pass_rate": total_filt / total_gen if total_gen > 0 else 0.0,
            "per_subtopic": per_subtopic,
            "config": {
                "generator_model": config.generator_model,
                "filter_model": config.filter_model,
                "samples_per_subtopic": config.samples_per_subtopic,
                "subtopic_branches": config.subtopic_branches,
                "filter_mode": config.filter_mode,
                "filter_threshold": config.filter_threshold,
                "filter_ceiling": config.filter_ceiling,
            },
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _normalize_dataset_seeds(
    ds: Dataset, offset: int = 0
) -> list[dict[str, Any]]:
    """Convert an HF Dataset into a list of seed dicts."""
    content_col = None
    for candidate in ("content", "text", "question", "prompt"):
        if candidate in ds.column_names:
            content_col = candidate
            break
    if content_col is None:
        for col in ds.column_names:
            if ds.features[col].dtype == "string":
                content_col = col
                break
    if content_col is None:
        raise ValueError(
            f"Cannot find a text column in dataset. Columns: {ds.column_names}"
        )

    result: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        content = row[content_col]
        if isinstance(content, list):
            content = json.dumps(content)
        result.append({"id": str(offset + i), "content": str(content)})
    return result


def _parse_json_array(text: str, fallback_count: int = 3) -> list[str]:
    """Extract a JSON array of strings from LLM output, with fallback heuristics."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass

    return [f"subtopic_{i}" for i in range(fallback_count)]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None
