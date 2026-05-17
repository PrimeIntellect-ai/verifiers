"""Deterministic grading helpers for emulator benchmark outputs."""

from .audio import audio_crc
from .framebuffer import framebuffer_sha256, frame_sequence_sha256
from .perf import fps, stable_within
from .trace import trace_crc

__all__ = [
    "audio_crc",
    "fps",
    "frame_sequence_sha256",
    "framebuffer_sha256",
    "stable_within",
    "trace_crc",
]
