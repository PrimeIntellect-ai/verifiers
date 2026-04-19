"""YAML-driven prompt system for debate environments.

Loads prompt packs from YAML, compiles Jinja templates, resolves field
declarations into FieldSpec objects.  Single-pass sandboxed Jinja rendering.
"""

from __future__ import annotations

import functools
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
import jinja2.nodes
import jinja2.sandbox
import yaml

from .fields import (
    EnumScoring,
    FieldSpec,
    _resolve_fields,
    _TYPE_MAP,
    resolve_scoring,
    validate_type_scoring,
)
from .parsing import generate_format_instructions

if TYPE_CHECKING:
    pass

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_ROLE_NAMES = {"judge", "debater_a", "debater_b"}
_TAG_NAME_RE = re.compile(r"^\w+$")
_UTILITY_BLOCK_NAMES = ("_matcher", "_grader")

_THINK_VISIBILITY_LEVELS = frozenset(
    {"disabled", "private", "visible_to_judge", "open"}
)

_THINK_DESCRIPTIONS: dict[str, str] = {
    "private": "Your reasoning is private -- other participants will NOT see it.",
    "visible_to_judge": "The judge will see your reasoning but your opponent will NOT.",
    "open": "All reasoning is visible to all participants.",
}


# ---------------------------------------------------------------------------
# Jinja environment (module-level singleton)
# ---------------------------------------------------------------------------

_jinja_env = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)


# ---------------------------------------------------------------------------
# JudgeTemplate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeTemplate:
    """Raw user-prompt template for an eval-time LLM judge call.

    ``user`` is a Python ``.format()`` template with ``{question}``,
    ``{answer}``, and ``{response}`` placeholders consumed by
    ``JudgeRubric.judge``. ``positive`` / ``negative`` are the canonical
    verdict tokens (uppercased, punct-stripped first word) the rubric
    compares against the model's response.
    """

    user: str
    positive: str
    negative: str


# ---------------------------------------------------------------------------
# DebatePrompts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebatePrompts:
    """Frozen, LRU-cached prompt pack loaded from YAML.

    Pack invariants are enforced at construction (``__post_init__``):

    1. Every ``JudgeTemplate.user`` is a non-empty Python ``str.format`` string.
       An empty template would render a 0-byte prompt at score time.
    2. Verdict tokens (``positive``/``negative``) on each judge MUST NOT
       collide with any answer-field ``EnumScoring`` value. Collision would
       let transcript greps misattribute judge output to a debater commit.

    These checks live here (not in the factory) because they are intrinsic
    to the pack itself — every constructor of ``DebatePrompts`` (factory,
    test, custom loader) gets the guarantees for free.
    """

    system: dict[str, jinja2.Template]
    user: dict[str, dict[str, jinja2.Template]]
    question: dict[str, jinja2.Template]
    fields: dict[str, dict[str, dict[str, FieldSpec]]]
    think_visibility: dict[str, str]
    think_tag: str
    prefill: dict[str, dict[str, jinja2.Template]]
    opponent_wrap: dict[str, jinja2.Template] | None
    judges: dict[str, JudgeTemplate]
    source_ref: str

    def __post_init__(self) -> None:
        for kind, template in self.judges.items():
            if not template.user.strip():
                raise ValueError(
                    f"Judge template {kind!r} in pack {self.source_ref!r} "
                    "has an empty 'user' template. JudgeRubric.judge would "
                    "render a 0-byte prompt at score time. Production YAML "
                    "packs populate this from the _grader/_matcher 'user' "
                    "block at pack-load time."
                )
            pos = template.positive
            neg = template.negative
            for role_fields in self.fields.values():
                for phase_fields in role_fields.values():
                    answer_spec = phase_fields.get("answer")
                    if answer_spec is None:
                        continue
                    if not isinstance(answer_spec.scoring, EnumScoring):
                        continue
                    for enum_val in answer_spec.scoring.values:
                        upper = enum_val.upper()
                        if upper == pos or upper == neg:
                            raise ValueError(
                                f"Judge template {kind!r} in pack "
                                f"{self.source_ref!r}: verdict token "
                                f"{pos!r}/{neg!r} collides with answer enum "
                                f"value {enum_val!r}. Pick distinct verdict "
                                "tokens so transcript greps can't "
                                "misattribute judge output to a debater "
                                "commit."
                            )

    # -- Rendering ----------------------------------------------------------

    def render_system(self, role: str, ctx: dict[str, Any]) -> str:
        tmpl = self.system.get(role)
        if tmpl is None:
            raise KeyError(
                f"No system template for role={role} in {self.source_ref}. "
                f"Available roles: {sorted(self.system)}"
            )
        return tmpl.render(ctx)

    def render_question(self, role: str, ctx: dict[str, Any]) -> str | None:
        tmpl = self.question.get(role)
        if tmpl is None:
            return None
        result = tmpl.render(ctx)
        return result if result.strip() else None

    def render_instruction(
        self, role: str, phase: str, ctx: dict[str, Any]
    ) -> str | None:
        """Render user template + think instruction + field format instructions."""
        parts: list[str] = []

        # Phase instruction (explicit phase key wins, else fallback to 'default')
        role_block = self.user.get(role, {})
        tmpl = role_block.get(phase) or role_block.get("default")
        if tmpl is not None:
            rendered = tmpl.render(ctx).strip()
            if rendered:
                parts.append(rendered)

        # Think instruction
        think_instr = self._think_instruction(role)
        if think_instr is not None:
            parts.append(think_instr)

        # Field format instructions
        field_instr = self._field_instructions(role, phase)
        if field_instr is not None:
            parts.append(field_instr)

        return "\n\n".join(parts) if parts else None

    def render_prefill(self, role: str, phase: str, ctx: dict[str, Any]) -> str | None:
        role_block = self.prefill.get(role, {})
        tmpl = role_block.get(phase) or role_block.get("default")
        if tmpl is None:
            return None
        result = tmpl.render(ctx).strip()
        return result if result else None

    def wrap_opponent(
        self,
        phase: str,
        content: str,
        *,
        member_id: str,
        viewer_id: str,
    ) -> str:
        """Wrap opponent utterance text with speaker attribution.

        Template context: ``text``, ``phase``, ``member_id``,
        ``viewer_id``. A pack's ``opponent_wrap`` template SHOULD
        reference ``member_id`` so judges and peer debaters can attribute
        each block to the correct speaker. Without attribution, the judge
        has to infer speaker identity from transcript order — a latent
        ambiguity that breaks whenever argument lengths are asymmetric
        or the transcript gets reordered.

        ``viewer_id`` is threaded through so packs can switch framing on
        who is *reading* the transcript. When the viewer is a judge and
        the pack declares a ``judge`` template, that template is used;
        otherwise the ``debater`` template (or the first declared one).
        Packs that don't reference ``viewer_id`` in their Jinja body are
        unaffected.

        When no ``opponent_wrap`` template is defined, falls back to a
        minimal ``[member_id] content`` prefix. Bare passthrough would
        leave the attribution gap unfixed for packs that omit a wrap
        template.
        """
        if self.opponent_wrap is None:
            return f"[{member_id}] {content}"
        if viewer_id == "judge" and "judge" in self.opponent_wrap:
            key = "judge"
        elif "debater" in self.opponent_wrap:
            key = "debater"
        else:
            key = next(iter(self.opponent_wrap))
        return self.opponent_wrap[key].render(
            text=content,
            phase=phase,
            member_id=member_id,
            viewer_id=viewer_id,
        )

    def get_field_specs(self, role: str, phase: str) -> dict[str, FieldSpec] | None:
        trigger_map = self.fields.get(role)
        if trigger_map is None:
            return None
        specs = trigger_map.get(phase)
        return dict(specs) if specs else None

    # -- Private helpers ----------------------------------------------------

    def _think_instruction(self, role: str) -> str | None:
        vis = self.think_visibility.get(role, "disabled")
        if vis == "disabled":
            return None
        desc = _THINK_DESCRIPTIONS.get(vis, "")
        return f"Use <{self.think_tag}>...</{self.think_tag}> tags for your reasoning. {desc}"

    def _field_instructions(self, role: str, trigger: str) -> str | None:
        role_fields = self.fields.get(role)
        if not role_fields:
            return None
        trigger_fields = role_fields.get(trigger)
        if not trigger_fields:
            return None
        return generate_format_instructions(trigger_fields)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_context(
    *,
    task_prompt: str,
    viewer_id: str,
    phase: str,
    round_index: int,
    num_rounds: int,
    answer: str = "",
) -> dict[str, Any]:
    """Build Jinja template context for rendering."""
    return {
        "task_prompt": task_prompt,
        "viewer_id": viewer_id,
        "phase": phase,
        "round_index": round_index,
        "num_rounds": num_rounds,
        "is_first_round": round_index == 0,
        "is_last_round": round_index == num_rounds - 1,
        "answer": answer,
        "has_assigned_answer": bool(answer),
    }


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def resolve_prompts(ref: str) -> DebatePrompts:
    """Load and compile a prompt pack from YAML.

    If *ref* contains no '/' or '.', treat as built-in name (looked up in
    ``prompts/`` subdir). Otherwise treat as file path.
    """
    ref = ref.strip()
    if "/" not in ref and "." not in ref:
        path = _PROMPTS_DIR / f"{ref}.yaml"
    else:
        path = Path(ref)

    raw = path.read_text()
    d = yaml.safe_load(raw)

    _validate(d)

    system = _compile_flat_templates(d.get("system", {}))
    user = _compile_templates(d.get("user", {}))
    question = _compile_flat_templates(d.get("question", {}))
    think_visibility, think_tag = _normalize_think(d.get("think", {}))
    prefill = _compile_templates(d.get("prefill", {}))
    fields = _parse_fields(d.get("fields", {}))
    judges = _compile_judge_blocks(d)

    raw_ow = d.get("opponent_wrap")
    opponent_wrap: dict[str, jinja2.Template] | None = None
    if raw_ow is not None:
        opponent_wrap = {k: _jinja_env.from_string(v) for k, v in raw_ow.items()}

    return DebatePrompts(
        system=system,
        user=user,
        question=question,
        fields=fields,
        think_visibility=think_visibility,
        think_tag=think_tag,
        prefill=prefill,
        opponent_wrap=opponent_wrap,
        judges=judges,
        source_ref=str(path),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# Per-turn template variables that MUST NOT appear in system or question
# blocks. Those blocks render once per slot and sit in the prompt prefix —
# if they vary with round/phase/slot, the monotonic-extension invariant is
# silently broken and the stitcher can no longer reuse the prior prefix.
# User/instruction blocks are allowed to reference these (they live at the
# tail, not the prefix).
_PER_TURN_VARS = frozenset(
    {
        "round_index",
        "num_rounds",
        "phase",
        "is_first_round",
        "is_last_round",
        "slot_id",
        "member_id",
    }
)


def _reject_per_turn_vars(template: str, *, where: str) -> None:
    """AST-scan a Jinja template for per-turn variable references.

    Walks the parsed AST and rejects ANY reference — expression tag
    (``{{ round_index }}``), statement tag (``{% if is_last_round %}``),
    attribute access (``{{ ctx.round_index }}``), index access
    (``{{ data[round_index] }}``), set directive (``{% set r =
    round_index %}``) — to any name in ``_PER_TURN_VARS``.

    The hard correctness boundary is the runtime monotonic-prefix test;
    this scan is a load-time author-facing guard that catches the
    common mistake early with a diagnostic message.
    """
    try:
        ast = _jinja_env.parse(template)
    except jinja2.TemplateSyntaxError:
        # Syntax errors surface later when render is called; not our
        # concern here.
        return
    for node in ast.find_all(jinja2.nodes.Name):
        if node.name in _PER_TURN_VARS:
            raise ValueError(
                f"{where}: template references per-turn variable "
                f"{node.name!r}. System and question blocks render once "
                "per slot and must be turn-invariant so the prompt "
                "prefix is stable across slots. Move per-turn content "
                "to the 'user' block."
            )


def _validate(d: dict) -> None:
    if d.get("version") != 2:
        raise ValueError(f"Unsupported prompt version: {d.get('version')} (expected 2)")

    _validate_judge_blocks(d)

    for section in ("system", "user"):
        block = d.get(section, {})
        for role in block:
            if role not in _ROLE_NAMES:
                raise ValueError(f"Unknown role '{role}' in {section}")

    for role, val in d.get("system", {}).items():
        if not isinstance(val, str):
            raise ValueError(
                f"system.{role}: expected a string, got {type(val).__name__}. "
                "System prompts are role-only; phase-varying content belongs in the user block."
            )
        if not val.strip():
            raise ValueError(f"system.{role}: empty template")
        _reject_per_turn_vars(val, where=f"system.{role}")

    question_block = d.get("question", {})
    for required_role in ("debater_a", "debater_b"):
        if required_role not in question_block:
            raise ValueError(
                f"question section missing required role '{required_role}'"
            )
    for role, val in question_block.items():
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in question")
        if isinstance(val, str):
            _reject_per_turn_vars(val, where=f"question.{role}")

    # Validate think config
    think_block = d.get("think", {})
    for role, val in think_block.items():
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in think")
        if val is True:
            raise ValueError(
                f"think.{role}: bare `true` is not allowed -- "
                "specify a visibility string (private, visible_to_judge, open) or false"
            )
        if isinstance(val, str):
            if val not in _THINK_VISIBILITY_LEVELS:
                raise ValueError(f"think.{role}: unknown visibility '{val}'")
        elif val is not False and not isinstance(val, dict):
            raise ValueError(
                f"think.{role}: expected false, str, or dict, got {type(val).__name__}"
            )

    # Validate field tag names and scoring compatibility
    for role, triggers in d.get("fields", {}).items():
        if role not in _ROLE_NAMES:
            raise ValueError(f"Unknown role '{role}' in fields")
        for trigger, field_defs in triggers.items():
            for tag_name, props in field_defs.items():
                if not _TAG_NAME_RE.match(tag_name):
                    raise ValueError(f"Invalid field tag name: '{tag_name}'")
                if isinstance(props, dict) and "scoring" in props:
                    type_str = props.get("type", "str")
                    if type_str in _TYPE_MAP:
                        scoring = resolve_scoring(props["scoring"])
                        if scoring is not None:
                            validate_type_scoring(
                                tag_name, _TYPE_MAP[type_str], scoring
                            )

    # Validate opponent_wrap
    ow = d.get("opponent_wrap")
    if ow is not None:
        if not isinstance(ow, dict):
            raise ValueError(
                f"opponent_wrap: expected mapping, got {type(ow).__name__}"
            )
        valid_keys = {"debater", "judge"}
        extra = set(ow) - valid_keys
        if extra:
            raise ValueError(
                f"opponent_wrap: unknown keys {sorted(extra)} (expected 'debater' and/or 'judge')"
            )
        for key, val in ow.items():
            if not isinstance(val, str):
                raise ValueError(
                    f"opponent_wrap.{key}: expected str, got {type(val).__name__}"
                )


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------


def _compile_templates(block: dict) -> dict[str, dict[str, jinja2.Template]]:
    result: dict[str, dict[str, jinja2.Template]] = {}
    for role, phases in block.items():
        result[role] = {}
        for phase_name, template_str in phases.items():
            result[role][phase_name] = _jinja_env.from_string(template_str)
    return result


def _compile_flat_templates(block: dict[str, str]) -> dict[str, jinja2.Template]:
    return {role: _jinja_env.from_string(tmpl_str) for role, tmpl_str in block.items()}


def _normalize_think(block: dict) -> tuple[dict[str, str], str]:
    """Parse think block into (role → visibility, tag).

    Returns a flat dict mapping role to visibility string,
    plus the think tag (uniform across roles, default "thinking").
    """
    visibility: dict[str, str] = {}
    tag = "thinking"

    for role, val in block.items():
        if isinstance(val, dict):
            tag = val.get("tag", tag)
            vis = val.get("visibility", "disabled")
        elif val is False:
            vis = "disabled"
        elif isinstance(val, str):
            vis = val
        else:
            raise ValueError(
                f"think.{role}: expected false, str, or dict, got {type(val).__name__}"
            )
        if vis not in _THINK_VISIBILITY_LEVELS:
            raise ValueError(f"think.{role}: unknown visibility '{vis}'")
        visibility[role] = vis

    return visibility, tag


def _parse_fields(block: dict) -> dict[str, dict[str, dict[str, FieldSpec]]]:
    result: dict[str, dict[str, dict[str, FieldSpec]]] = {}
    for role, triggers in block.items():
        result[role] = {}
        for trigger, field_defs in triggers.items():
            result[role][trigger] = _resolve_fields(field_defs)
    return result


def _normalize_verdict_token(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    token = stripped.split()[0].rstrip(string.punctuation)
    return token.upper() if token else None


def _validate_judge_blocks(d: dict) -> None:
    for block_name in _UTILITY_BLOCK_NAMES:
        block = d.get(block_name)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(
                f"{block_name}: expected mapping, got {type(block).__name__}"
            )
        required_keys = {"system", "user", "positive", "negative"}
        missing = required_keys - set(block)
        extra = set(block) - required_keys
        if missing:
            raise ValueError(f"{block_name}: missing required keys {sorted(missing)}")
        if extra:
            raise ValueError(f"{block_name}: unknown keys {sorted(extra)}")
        for key in ("system", "user", "positive", "negative"):
            if not isinstance(block[key], str):
                raise ValueError(f"{block_name}.{key}: expected str")
        positive = _normalize_verdict_token(block["positive"])
        negative = _normalize_verdict_token(block["negative"])
        if positive is None:
            raise ValueError(f"{block_name}.positive: expected non-empty verdict token")
        if negative is None:
            raise ValueError(f"{block_name}.negative: expected non-empty verdict token")
        if positive == negative:
            raise ValueError(
                f"{block_name}: positive/negative must normalize to distinct verdicts"
            )


def _compile_judge_blocks(d: dict) -> dict[str, JudgeTemplate]:
    result: dict[str, JudgeTemplate] = {}
    for block_name, short_name in (("_matcher", "matcher"), ("_grader", "grader")):
        block = d.get(block_name)
        if block is None:
            continue
        positive = _normalize_verdict_token(block["positive"])
        negative = _normalize_verdict_token(block["negative"])
        if positive is None or negative is None:
            raise ValueError(
                f"{block_name}: verdict tokens must be non-empty "
                f"(positive={block['positive']!r}, negative={block['negative']!r})"
            )
        result[short_name] = JudgeTemplate(
            user=block["user"],
            positive=positive,
            negative=negative,
        )
    return result
