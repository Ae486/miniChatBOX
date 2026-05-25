"""Read-manifest and trace helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rp.context_engineering.contracts import (
    ContextBudgetDecision,
    ContextManifestDecision,
    ContextManifestItem,
    ContextOperationKind,
    ContextReadManifest,
    ContextSourceItem,
    ContextTrace,
)
from rp.context_engineering.estimation import (
    estimate_source_item_tokens,
    estimate_source_items_tokens,
)


def build_manifest_item(
    item: ContextSourceItem,
    *,
    decision: ContextManifestDecision,
    reason: str,
    slot: str | None = None,
    estimated_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContextManifestItem:
    """Build a content-free manifest row for one source item."""

    return ContextManifestItem(
        source_item_id=item.source_item_id,
        source_family=item.source_family,
        source_scope=item.source_scope,
        source_ref=item.source_ref,
        visibility=item.visibility,
        decision=decision,
        reason=reason,
        slot=slot,
        estimated_tokens=(
            estimate_source_item_tokens(item)
            if estimated_tokens is None
            else estimated_tokens
        ),
        metadata=dict(metadata or {}),
    )


def empty_read_manifest() -> ContextReadManifest:
    """Return an empty read manifest."""

    return ContextReadManifest()


def build_trace(
    *,
    operation_id: str,
    operation_kind: ContextOperationKind,
    runtime_family: str,
    source_items: Sequence[ContextSourceItem],
    selected_items: Sequence[ContextSourceItem],
    read_manifest: ContextReadManifest,
    budget_decisions: Sequence[ContextBudgetDecision] = (),
    summary_action: str | None = None,
    fallback_reason: str | None = None,
    provider_usage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContextTrace:
    """Build a transient trace that excludes raw hidden source content."""

    source_counts = Counter(item.source_family for item in source_items)
    selected_counts = Counter(item.source_family for item in selected_items)
    return ContextTrace(
        operation_id=operation_id,
        operation_kind=operation_kind,
        runtime_family=runtime_family,
        estimate_method="approx_chars_div_4",
        input_source_count=len(source_items),
        selected_source_count=len(selected_items),
        omitted_source_count=len(read_manifest.omitted),
        hidden_source_count=len(read_manifest.hidden),
        forbidden_source_count=len(read_manifest.forbidden),
        metadata_only_source_count=len(read_manifest.metadata_only),
        estimated_input_tokens=estimate_source_items_tokens(source_items),
        selected_tokens=estimate_source_items_tokens(selected_items),
        budget_decisions=list(budget_decisions),
        source_counts_by_family=dict(source_counts),
        selected_counts_by_family=dict(selected_counts),
        summary_action=summary_action,
        fallback_reason=fallback_reason,
        provider_usage=dict(provider_usage or {}),
        metadata=dict(metadata or {}),
    )


def merge_trace_metadata(
    trace: ContextTrace,
    metadata: Mapping[str, Any],
) -> ContextTrace:
    """Return a copy of trace with merged metadata."""

    merged = dict(trace.metadata)
    merged.update(metadata)
    return trace.model_copy(update={"metadata": merged})
