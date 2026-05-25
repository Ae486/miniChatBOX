"""Centralized prompt accessors for SetupAgent."""

from __future__ import annotations

from .compact import COMPACT_SYSTEM_PROMPT
from .main import (
    RUNTIME_OVERLAY_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    render_setup_agent_system_prompt,
    render_specialist_preamble,
)
from .stages import stage_objective_text
from .tools import capability_guidance_text, tool_description_text


def runtime_overlay_instruction() -> str:
    return RUNTIME_OVERLAY_PROMPT


def compact_prompt_system_prompt() -> str:
    return COMPACT_SYSTEM_PROMPT


__all__ = [
    "COMPACT_SYSTEM_PROMPT",
    "RUNTIME_OVERLAY_PROMPT",
    "SYSTEM_PROMPT_TEMPLATE",
    "capability_guidance_text",
    "compact_prompt_system_prompt",
    "render_setup_agent_system_prompt",
    "render_specialist_preamble",
    "runtime_overlay_instruction",
    "stage_objective_text",
    "tool_description_text",
]
