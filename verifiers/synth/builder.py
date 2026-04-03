from __future__ import annotations

import asyncio
import glob
import inspect
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from verifiers.clients import resolve_client
from verifiers.clients.client import Client
from verifiers.types import ClientConfig, Messages, Response, SystemMessage, UserMessage
from verifiers.utils.message_utils import maybe_normalize_messages
from verifiers.utils.config_utils import ensure_keys

from verifiers.synth.prompts import (
    FILTER_JUDGE_PROMPT,
    GENERATE_PROMPT,
    PLAN_PROMPT,
    _NO_SOURCE_DIRECTIVE,
    render_env_spec,
)
from verifiers.synth.models import EnvSpec, SynthConfig, SynthPlan, SynthSample
from verifiers.synth.result import BuildResult

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


def _make_client(model_str: str) -> tuple[Client, str]:
    """Create a verifiers client from a provider/model string."""
    provider, model = _parse_model(model_str)
    cfg = PROVIDER_CONFIGS[provider]
    client_type = (
        "anthropic_messages" if provider == "anthropic" else "openai_chat_completions"
    )
    client = resolve_client(
        ClientConfig(
            client_type=client_type,
            api_key_var=cfg["key"],
            api_base_url=cfg["url"],
        )
    )
    return client, model


def _jsonify_row(row: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy of a dataset row for JSON serialization."""
    return dict(row)


def _infer_task_answer_fields(columns: list[str]) -> tuple[str, str]:
    """Pick default task / answer column names from dataset columns."""
    if "question" in columns:
        task = "question"
    elif "prompt" in columns:
        task = "prompt"
    else:
        task = columns[0]
    if "answer" in columns:
        ans = "answer"
    elif "temp_answer" in columns:
        ans = "temp_answer"
    elif len(columns) > 1:
        ans = next((c for c in columns if c != task), columns[0])
    else:
        ans = columns[0]
    return task, ans


def _fallback_schema() -> dict[str, Any]:
    """Fallback row schema when the environment does not expose a dataset."""
    return {
        "question": {"dtype": "string", "_type": "Value"},
        "answer": {"dtype": "string", "_type": "Value"},
    }


def _stringify_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str)
    return str(value)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return " ".join(chunks).strip()
    return ""


def _response_to_text(response: Response) -> str:
    return _message_content_to_text(response.message.content).strip()


def _is_message_like(value: Any) -> bool:
    return isinstance(value, dict) or (
        hasattr(value, "role") and hasattr(value, "content")
    )


def _aggregate_seed_context(
    seeds: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Collect all dataset rows and build reference JSON for prompts / ICL."""
    all_rows: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    for s in seeds:
        if s.get("kind") == "rows":
            all_rows.extend(s["rows"])
        elif s.get("content"):
            text_blocks.append(s["content"])
    if all_rows:
        ref = json.dumps(all_rows, default=str, indent=2)
        return all_rows, ref
    if text_blocks:
        combined = "\n\n---\n\n".join(text_blocks)
        return [], combined
    return [], _NO_SOURCE_DIRECTIVE


def _schema_keys(schema: dict[str, Any]) -> list[str]:
    """Column keys from a Features.to_dict()-style schema."""
    return list(schema.keys())


def _validate_row_against_schema(
    row: dict[str, Any], schema: dict[str, Any]
) -> tuple[bool, str | None]:
    """Validate row keys and nested shape against the environment dataset schema."""
    if set(row.keys()) != set(_schema_keys(schema)):
        return False, "keys"

    try:
        from datasets import Features

        Features.from_dict(schema).encode_example(row)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _messages_for_task(
    row: dict[str, Any],
    task_field: str,
    answer_field: str,
    system_prompt: str,
) -> tuple[Messages, str]:
    """Build model messages and a human-readable task string from a dataset row."""
    value = row.get(task_field)
    if (
        task_field == "prompt"
        and isinstance(value, list)
        and all(_is_message_like(item) for item in value)
    ):
        messages = maybe_normalize_messages(
            cast(Messages, value), field_name=f"row[{task_field!r}]"
        )
        if system_prompt and (
            not messages or not isinstance(messages[0], SystemMessage)
        ):
            messages = [SystemMessage(content=system_prompt), *messages]
        return messages, _stringify_value(value)

    if task_field in row:
        task_text = _stringify_value(row[task_field])
    else:
        task_payload = {k: v for k, v in row.items() if k != answer_field}
        task_text = _stringify_value(task_payload)

    messages: Messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(UserMessage(content=task_text))
    return messages, task_text


def _with_reference_material(messages: Messages, context: str | None) -> Messages:
    """Inject bounded seed context into the messages without flattening prompt rows."""
    if not context:
        return messages

    ref_message = UserMessage(content=f"Reference material:\n{context[:6000]}")
    if messages and isinstance(messages[0], SystemMessage):
        return [messages[0], ref_message, *messages[1:]]
    return [ref_message, *messages]


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

        dataset_schema: dict[str, Any] = {}
        try:
            ds = env.get_dataset(n=1)
            if ds is not None and len(ds) > 0:
                dataset_schema = ds.features.to_dict()
        except Exception:
            logger.debug("Could not extract dataset schema from environment dataset")

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
            parser_info=parser_info,
            few_shot=env.few_shot,
        )

    async def build(
        self,
        seeds: list[str] | Dataset | None = None,
        max_seed_examples: int = 3,
        samples_per_subtopic: int = 5,
        max_subtopics: int | None = None,
        filter_threshold: float = 0.8,
        filter_ceiling: float | None = None,
        coverage_quality: float = 0.8,
        subtopics: list[str] | None = None,
    ) -> BuildResult:
        config = SynthConfig(
            generator_model=self.generator_model,
            filter_model=self.filter_model,
            max_seed_examples=max_seed_examples,
            samples_per_subtopic=samples_per_subtopic,
            max_subtopics=max_subtopics,
            filter_threshold=filter_threshold,
            filter_ceiling=filter_ceiling,
            coverage_quality=coverage_quality,
        )
        self._validate_keys(config)
        normalized = self._resolve_seeds(seeds, config.max_seed_examples)
        logger.info("Resolved %d seed bundle(s)", len(normalized))

        schema = self.env_spec.dataset_schema or _fallback_schema()
        task_field, answer_field = _infer_task_answer_fields(_schema_keys(schema))

        if subtopics is not None:
            _, ref = _aggregate_seed_context(normalized)
            plan = SynthPlan(
                subtopics=list(subtopics),
                total_target=len(subtopics) * config.samples_per_subtopic,
                generation_guidance="Regenerate rows per subtopic; match the environment schema.",
                task_field=task_field,
                answer_field=answer_field,
                reference_material=ref if ref != _NO_SOURCE_DIRECTIVE else None,
            )
        else:
            plan = await self._plan(normalized, config)
        logger.info(
            "Plan: %d subtopics, %d target samples, schema keys=%s",
            len(plan.subtopics),
            plan.total_target,
            _schema_keys(schema),
        )

        raw = await self._generate(plan, config)
        logger.info("Generated %d raw samples", len(raw))

        filtered = await self._filter(raw, plan, config)
        logger.info("Filtered to %d samples", len(filtered))

        stats = self._compute_stats(raw, filtered, config, plan)
        return BuildResult(raw_samples=raw, filtered_samples=filtered, stats=stats)

    @staticmethod
    def _validate_keys(config: SynthConfig) -> None:
        """Fail fast if required API keys are missing."""
        needed: set[str] = set()
        for model_str in (config.generator_model, config.filter_model):
            provider, _ = _parse_model(model_str)
            needed.add(PROVIDER_CONFIGS[provider]["key"])
        ensure_keys(sorted(needed))

    # ------------------------------------------------------------------
    # Seed resolution & normalization
    # ------------------------------------------------------------------

    def _resolve_seeds(
        self, seeds: list[str] | Any | None, max_examples: int = 3
    ) -> list[dict[str, Any]]:
        """Resolve seeds: explicit seeds, bounded env sample, or seedless."""
        if seeds is not None:
            return self._normalize_seeds(seeds, max_examples)

        try:
            ds = self.env.get_dataset(n=max_examples)
            if ds is not None and len(ds) > 0:
                logger.info("Using bounded environment seed sample (%d rows)", len(ds))
                return _normalize_dataset_seeds(ds, max_examples)
        except Exception:
            logger.debug("Could not load dataset from environment")

        logger.info("No seeds available; running in seedless mode")
        return [{"id": "0", "kind": "text", "content": ""}]

    @staticmethod
    def _normalize_seeds(
        seeds: list[str] | Any, max_examples: int = 3
    ) -> list[dict[str, Any]]:
        """Unify bounded paths, raw strings, and HF Datasets into seed bundles."""
        try:
            from datasets import Dataset

            if isinstance(seeds, Dataset):
                return _normalize_dataset_seeds(seeds, max_examples)
        except ImportError:
            pass

        if not isinstance(seeds, list):
            raise TypeError(f"seeds must be list[str] or Dataset, got {type(seeds)}")

        result: list[dict[str, Any]] = []
        idx = 0
        for item in seeds:
            if idx >= max_examples:
                break
            if not isinstance(item, str):
                try:
                    from datasets import Dataset

                    if isinstance(item, Dataset):
                        result.append(
                            {
                                "id": str(idx),
                                "kind": "rows",
                                "rows": _dataset_rows_list(item, max_examples),
                            }
                        )
                        idx += 1
                        continue
                except ImportError:
                    pass
                raise TypeError(f"Unexpected seed type: {type(item)}")

            expanded = glob.glob(item)
            if expanded:
                for path_str in sorted(expanded):
                    if idx >= max_examples:
                        break
                    p = Path(path_str)
                    if p.is_file():
                        content = p.read_text(encoding="utf-8", errors="replace")
                        result.append(
                            {"id": str(idx), "kind": "text", "content": content}
                        )
                        idx += 1
            else:
                result.append({"id": str(idx), "kind": "text", "content": item})
                idx += 1

        return result

    # ------------------------------------------------------------------
    # Stage 1: Planning (single orchestrator call)
    # ------------------------------------------------------------------

    async def _plan(
        self, seeds: list[dict[str, Any]], config: SynthConfig
    ) -> SynthPlan:
        planning_rows, examples_json = _aggregate_seed_context(seeds)
        spec_text = render_env_spec(asdict(self.env_spec))

        max_line = ""
        if config.max_subtopics is not None:
            max_line = f"\nReturn at most {config.max_subtopics} subtopics."

        gen_client, gen_model = _make_client(config.generator_model)
        prompt = PLAN_PROMPT.format(
            env_spec=spec_text,
            examples_json=examples_json,
            samples_per_subtopic=config.samples_per_subtopic,
            max_subtopics_line=max_line,
        )
        try:
            resp = await gen_client.get_response(
                prompt=[UserMessage(content=prompt)],
                model=gen_model,
                sampling_args={"temperature": 0.7},
            )
        finally:
            await gen_client.close()
        text = _response_to_text(resp)
        parsed = _parse_json_object(text)

        subtopics: list[str]
        gen_guidance: str | None = None
        if parsed and isinstance(parsed.get("subtopics"), list):
            subtopics = [str(x) for x in parsed["subtopics"]]
            if parsed.get("generation_guidance"):
                gen_guidance = str(parsed["generation_guidance"])
        else:
            subtopics = _parse_json_array(text)
            gen_guidance = None

        schema = self.env_spec.dataset_schema or _fallback_schema()
        task_field, answer_field = _infer_task_answer_fields(_schema_keys(schema))

        if config.max_subtopics is not None and len(subtopics) > config.max_subtopics:
            subtopics = subtopics[: config.max_subtopics]

        total = len(subtopics) * config.samples_per_subtopic
        if planning_rows or (examples_json and examples_json != _NO_SOURCE_DIRECTIVE):
            ref: str | None = examples_json
        else:
            ref = None

        return SynthPlan(
            subtopics=subtopics,
            total_target=total,
            generation_guidance=gen_guidance,
            task_field=task_field,
            answer_field=answer_field,
            reference_material=ref,
        )

    # ------------------------------------------------------------------
    # Stage 2: Generation (fan-out per subtopic)
    # ------------------------------------------------------------------

    async def _generate(
        self, plan: SynthPlan, config: SynthConfig
    ) -> list[SynthSample]:
        gen_client, gen_model = _make_client(config.generator_model)
        spec_text = render_env_spec(asdict(self.env_spec))
        sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)
        guidance = (
            plan.generation_guidance or "Match the example rows and schema exactly."
        )
        schema = self.env_spec.dataset_schema or _fallback_schema()
        schema_json = json.dumps(schema, indent=2)
        source = plan.reference_material or _NO_SOURCE_DIRECTIVE

        async def gen_one(subtopic: str) -> list[SynthSample]:
            samples: list[SynthSample] = []
            for _ in range(config.samples_per_subtopic):
                prompt = GENERATE_PROMPT.format(
                    env_spec=spec_text,
                    source_section=source,
                    subtopic=subtopic,
                    generation_guidance=guidance,
                    schema_json=schema_json,
                )
                async with sem:
                    resp = await gen_client.get_response(
                        prompt=[UserMessage(content=prompt)],
                        model=gen_model,
                        sampling_args={"temperature": 0.9},
                    )
                text = _response_to_text(resp)
                parsed = _parse_json_object(text)
                if not parsed:
                    logger.warning("Failed to parse sample from LLM response")
                    continue
                valid, reason = _validate_row_against_schema(parsed, schema)
                if not valid:
                    logger.warning("Generated row failed schema validation: %s", reason)
                    continue
                samples.append(SynthSample(row=dict(parsed), subtopic=subtopic))
            return samples

        try:
            results = await asyncio.gather(*[gen_one(st) for st in plan.subtopics])
            return [s for batch in results for s in batch]
        finally:
            await gen_client.close()

    # ------------------------------------------------------------------
    # Stage 3: Filtering (single path, two thresholds)
    # ------------------------------------------------------------------

    async def _filter(
        self,
        samples: list[SynthSample],
        plan: SynthPlan,
        config: SynthConfig,
    ) -> list[SynthSample]:
        if not samples:
            return []

        filter_client, filter_model = _make_client(config.filter_model)
        system_prompt = self.env_spec.system_prompt or ""
        sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)
        check_novelty = config.filter_ceiling is not None
        ref = plan.reference_material
        has_ref = bool(ref) and ref != _NO_SOURCE_DIRECTIVE

        async def attempt_answer(messages: Messages, context: str | None) -> str:
            messages = _with_reference_material(messages, context)
            async with sem:
                resp = await filter_client.get_response(
                    prompt=messages,
                    model=filter_model,
                    sampling_args={"temperature": 0.0},
                )
            return _response_to_text(resp)

        async def judge(task_text: str, golden: str, response: str) -> float:
            prompt = FILTER_JUDGE_PROMPT.format(
                task_text=task_text,
                golden_answer=golden,
                response=response,
            )
            judge_messages: Messages = [UserMessage(content=prompt)]
            async with sem:
                resp = await filter_client.get_response(
                    prompt=judge_messages,
                    model=filter_model,
                    sampling_args={"temperature": 0.0},
                )
            text = _response_to_text(resp)
            parsed = _parse_json_object(text)
            if parsed:
                return float(parsed.get("score", 0.0))
            return 0.0

        async def score_sample(sample: SynthSample) -> SynthSample:
            row = sample.row
            messages, task_text = _messages_for_task(
                row, plan.task_field, plan.answer_field, system_prompt
            )
            golden = _stringify_value(row.get(plan.answer_field, ""))

            ctx = ref if has_ref else None
            resp_with = await attempt_answer(messages, ctx)
            sample.score_with_context = await judge(task_text, golden, resp_with)

            if check_novelty and has_ref:
                resp_without = await attempt_answer(messages, None)
                sample.score_without_context = await judge(
                    task_text, golden, resp_without
                )
            return sample

        try:
            scored = await asyncio.gather(*[score_sample(s) for s in samples])
        finally:
            await filter_client.close()

        kept: list[SynthSample] = []
        for s in scored:
            if (s.score_with_context or 0.0) < config.filter_threshold:
                continue
            if (
                check_novelty
                and s.score_without_context is not None
                and s.score_without_context > config.filter_ceiling  # type: ignore[operator]
            ):
                continue
            kept.append(s)
        return kept

    # ------------------------------------------------------------------
    # Statistics & coverage
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        raw: list[SynthSample],
        filtered: list[SynthSample],
        config: SynthConfig,
        plan: SynthPlan,
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

        coverage: dict[str, dict[str, Any]] = {}
        by_subtopic: dict[str, list[SynthSample]] = {}
        for s in raw:
            by_subtopic.setdefault(s.subtopic, []).append(s)
        for name, samples in by_subtopic.items():
            passed = sum(
                1
                for s in samples
                if (s.score_with_context or 0.0) >= config.filter_threshold
            )
            total = len(samples)
            rate = passed / total if total > 0 else 0.0
            coverage[name] = {"total": total, "passed": passed, "rate": rate}
            if rate < config.coverage_quality:
                logger.warning(
                    "Coverage failure: subtopic %r has %.0f%% learnability "
                    "(threshold %.0f%%)",
                    name,
                    rate * 100,
                    config.coverage_quality * 100,
                )

        return {
            "total_generated": total_gen,
            "total_filtered": total_filt,
            "pass_rate": total_filt / total_gen if total_gen > 0 else 0.0,
            "per_subtopic": per_subtopic,
            "coverage": coverage,
            "config": {
                "generator_model": config.generator_model,
                "filter_model": config.filter_model,
                "samples_per_subtopic": config.samples_per_subtopic,
                "max_subtopics": config.max_subtopics,
                "max_seed_examples": config.max_seed_examples,
                "filter_threshold": config.filter_threshold,
                "filter_ceiling": config.filter_ceiling,
                "coverage_quality": config.coverage_quality,
                "task_field": plan.task_field,
                "answer_field": plan.answer_field,
            },
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dataset_rows_list(ds: Dataset, limit: int) -> list[dict[str, Any]]:
    """Bounded rows as plain dicts (every column)."""
    return [_jsonify_row(dict(ds[i])) for i in range(min(limit, len(ds)))]


def _normalize_dataset_seeds(
    ds: Dataset, limit: int, offset: int = 0
) -> list[dict[str, Any]]:
    """Single seed bundle containing a bounded row sample with all columns."""
    rows = _dataset_rows_list(ds, limit)
    return [{"id": str(offset), "kind": "rows", "rows": rows}]


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
