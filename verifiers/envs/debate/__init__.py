"""Debate environment: prompts, fields, parsing, think-tag handling."""

from .fields import FieldSpec
from .parsing import extract_fields
from .prompts import DebatePrompts, JudgeTemplate, build_context, resolve_prompts

__all__ = [
    "DebatePrompts",
    "FieldSpec",
    "JudgeTemplate",
    "build_context",
    "extract_fields",
    "resolve_prompts",
]
