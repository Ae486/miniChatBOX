"""Default policy builders for the common context engineering kernel."""

from __future__ import annotations

from rp.context_engineering.contracts import (
    ContextBudgetPolicy,
    ContextFallbackPolicy,
    ContextPlacementPolicy,
    ContextValidationPolicy,
)

DEFAULT_CONTEXT_ESTIMATE_CHARS_PER_TOKEN = 4
DEFAULT_FALLBACK_SUMMARY_LINE_LIMIT = 6
DEFAULT_SECTION_TITLE = "Context"

DEFAULT_ORDERED_SLOTS = [
    "stable_prefix",
    "runtime_overlay",
    "compact_artifact",
    "recent_raw",
    "tool_outcomes",
    "metadata_only",
]

DEFAULT_SLOT_BY_SOURCE_FAMILY = {
    "system_instruction": "stable_prefix",
    "runtime_state": "runtime_overlay",
    "compact_artifact": "compact_artifact",
    "user_turn": "recent_raw",
    "assistant_turn": "recent_raw",
    "tool_outcome": "tool_outcomes",
}


def default_budget_policy(
    *,
    context_window_tokens: int | None = None,
    response_reserve_tokens: int = 1024,
    operation_budget_tokens: int | None = None,
    recent_window_items: int | None = None,
    source_family_token_caps: dict[str, int] | None = None,
    source_family_item_caps: dict[str, int] | None = None,
) -> ContextBudgetPolicy:
    """Build a provider-neutral budget policy."""

    return ContextBudgetPolicy(
        context_window_tokens=context_window_tokens,
        response_reserve_tokens=response_reserve_tokens,
        operation_budget_tokens=operation_budget_tokens,
        recent_window_items=recent_window_items,
        source_family_token_caps=dict(source_family_token_caps or {}),
        source_family_item_caps=dict(source_family_item_caps or {}),
    )


def default_placement_policy(
    *,
    ordered_slots: list[str] | None = None,
    slot_by_source_family: dict[str, str] | None = None,
) -> ContextPlacementPolicy:
    """Build the default slot order shared by initial adapters."""

    slots = list(ordered_slots or DEFAULT_ORDERED_SLOTS)
    slot_map = dict(DEFAULT_SLOT_BY_SOURCE_FAMILY)
    slot_map.update(slot_by_source_family or {})
    return ContextPlacementPolicy(
        ordered_slots=slots,
        slot_by_source_family=slot_map,
        stable_prefix_slots=["stable_prefix"],
    )


def default_validation_policy(
    *,
    schema_id: str | None = None,
    allowed_recovery_ref_prefixes: list[str] | None = None,
    allowed_source_refs: list[str] | None = None,
    forbidden_payload_fields: list[str] | None = None,
    max_list_lengths: dict[str, int] | None = None,
    max_string_lengths: dict[str, int] | None = None,
    reject_unknown_fields: bool = True,
    metadata: dict[str, object] | None = None,
) -> ContextValidationPolicy:
    """Build a validation policy without runtime-specific semantics."""

    return ContextValidationPolicy(
        schema_id=schema_id,
        allowed_recovery_ref_prefixes=list(allowed_recovery_ref_prefixes or []),
        allowed_source_refs=list(allowed_source_refs or []),
        forbidden_payload_fields=list(forbidden_payload_fields or []),
        max_list_lengths=dict(max_list_lengths or {}),
        max_string_lengths=dict(max_string_lengths or {}),
        reject_unknown_fields=reject_unknown_fields,
        metadata=dict(metadata or {}),
    )


def default_fallback_policy(
    *,
    fallback_summary_line_limit: int = DEFAULT_FALLBACK_SUMMARY_LINE_LIMIT,
    user_visible_error_code: str | None = None,
) -> ContextFallbackPolicy:
    """Build the single fallback policy for compact operations."""

    return ContextFallbackPolicy(
        fallback_summary_line_limit=fallback_summary_line_limit,
        user_visible_error_code=user_visible_error_code,
    )
