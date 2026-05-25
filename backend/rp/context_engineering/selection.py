"""Deterministic source selection and read-manifest construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import cast

from rp.context_engineering.contracts import (
    ContextBudgetDecision,
    ContextOperationRequest,
    ContextReadManifest,
    ContextSelectionResult,
    ContextSourceItem,
    ContextStability,
)
from rp.context_engineering.estimation import estimate_source_item_tokens
from rp.context_engineering.policies import DEFAULT_SECTION_TITLE
from rp.context_engineering.serialization import build_section_for_items
from rp.context_engineering.tracing import build_manifest_item, build_trace

_CONVERSATION_FAMILIES = {"user_turn", "assistant_turn"}


def select_context_sections(
    request: ContextOperationRequest,
) -> ContextSelectionResult:
    """Select model-visible context using request-local policies only."""

    manifest = ContextReadManifest()
    budget_decisions: list[ContextBudgetDecision] = []
    visible_candidates: list[ContextSourceItem] = []
    omitted_ids: set[str] = set()

    for item in request.source_items:
        slot = _slot_for_item(request, item)
        if item.visibility == "forbidden":
            manifest.forbidden.append(
                build_manifest_item(
                    item,
                    decision="forbidden",
                    reason="forbidden_by_policy",
                    slot=slot,
                )
            )
            continue
        if item.visibility == "hidden":
            manifest.hidden.append(
                build_manifest_item(
                    item,
                    decision="hidden",
                    reason="hidden_by_policy",
                    slot=slot,
                )
            )
            continue
        if item.visibility == "metadata_only":
            manifest.metadata_only.append(
                build_manifest_item(
                    item,
                    decision="metadata_only",
                    reason="metadata_only",
                    slot=slot,
                )
            )
            continue
        if _is_empty_model_content(item):
            manifest.omitted.append(
                build_manifest_item(
                    item,
                    decision="omitted",
                    reason="empty_content",
                    slot=slot,
                )
            )
            omitted_ids.add(item.source_item_id)
            continue
        visible_candidates.append(item)

    sorted_candidates = sorted(
        visible_candidates,
        key=lambda item: _sort_key(request, item),
    )
    capped_candidates = _apply_family_caps(
        request=request,
        candidates=sorted_candidates,
        manifest=manifest,
        budget_decisions=budget_decisions,
        omitted_ids=omitted_ids,
    )
    recent_ids = _recent_raw_item_ids(request, capped_candidates)

    selected_items: list[ContextSourceItem] = []
    selected_ids: set[str] = set()
    compactable_dropped_items: list[ContextSourceItem] = []
    selected_token_count = 0

    for item in capped_candidates:
        if item.source_item_id in selected_ids or item.source_item_id in omitted_ids:
            continue
        slot = _slot_for_item(request, item)
        if (
            _is_raw_conversation_item(item)
            and recent_ids
            and item.source_item_id not in recent_ids
        ):
            manifest.omitted.append(
                build_manifest_item(
                    item,
                    decision="omitted",
                    reason="replaced_by_compact_artifact",
                    slot=slot,
                )
            )
            budget_decisions.append(
                ContextBudgetDecision(
                    decision="omit",
                    reason="replaced_by_compact_artifact",
                    source_item_id=item.source_item_id,
                    slot=slot,
                    estimated_tokens=estimate_source_item_tokens(item),
                )
            )
            compactable_dropped_items.append(item)
            omitted_ids.add(item.source_item_id)
            continue

        estimate = estimate_source_item_tokens(item)
        budget = request.budget_policy.operation_budget_tokens
        protected = item.source_item_id in recent_ids
        if (
            budget is not None
            and not protected
            and selected_token_count + estimate > int(budget)
        ):
            manifest.omitted.append(
                build_manifest_item(
                    item,
                    decision="omitted",
                    reason="operation_budget_exceeded",
                    slot=slot,
                    estimated_tokens=estimate,
                )
            )
            budget_decisions.append(
                ContextBudgetDecision(
                    decision="omit",
                    reason="operation_budget_exceeded",
                    source_item_id=item.source_item_id,
                    slot=slot,
                    estimated_tokens=estimate,
                )
            )
            omitted_ids.add(item.source_item_id)
            continue
        _select_item(
            request=request,
            item=item,
            selected_items=selected_items,
            selected_ids=selected_ids,
            manifest=manifest,
            reason="selected",
        )
        selected_token_count += estimate

    recent_raw_items = [
        item
        for item in selected_items
        if item.source_item_id in recent_ids and _is_raw_conversation_item(item)
    ]
    sections = _build_sections(request, selected_items)
    trace = build_trace(
        operation_id=request.operation_id,
        operation_kind=request.operation_kind,
        runtime_family=request.runtime_family,
        source_items=request.source_items,
        selected_items=selected_items,
        read_manifest=manifest,
        budget_decisions=budget_decisions,
        provider_usage=dict((request.metadata or {}).get("provider_usage") or {}),
        metadata={
            **request.metadata,
            "recent_raw_count": len(recent_raw_items),
            "compactable_dropped_count": len(compactable_dropped_items),
        },
    )
    return ContextSelectionResult(
        selected_items=selected_items,
        recent_raw_items=recent_raw_items,
        compactable_dropped_items=compactable_dropped_items,
        sections=sections,
        read_manifest=manifest,
        trace=trace,
    )


def _apply_family_caps(
    *,
    request: ContextOperationRequest,
    candidates: Sequence[ContextSourceItem],
    manifest: ContextReadManifest,
    budget_decisions: list[ContextBudgetDecision],
    omitted_ids: set[str],
) -> list[ContextSourceItem]:
    item_counts: dict[str, int] = defaultdict(int)
    token_counts: dict[str, int] = defaultdict(int)
    kept: list[ContextSourceItem] = []
    for item in candidates:
        family = str(item.source_family)
        slot = _slot_for_item(request, item)
        estimate = estimate_source_item_tokens(item)
        item_cap = request.budget_policy.source_family_item_caps.get(family)
        if item_cap is not None and item_counts[family] >= int(item_cap):
            manifest.omitted.append(
                build_manifest_item(
                    item,
                    decision="omitted",
                    reason="family_item_cap",
                    slot=slot,
                    estimated_tokens=estimate,
                )
            )
            budget_decisions.append(
                ContextBudgetDecision(
                    decision="omit",
                    reason="family_item_cap",
                    source_item_id=item.source_item_id,
                    slot=slot,
                    estimated_tokens=estimate,
                )
            )
            omitted_ids.add(item.source_item_id)
            continue
        token_cap = request.budget_policy.source_family_token_caps.get(family)
        if token_cap is not None and token_counts[family] + estimate > int(token_cap):
            manifest.omitted.append(
                build_manifest_item(
                    item,
                    decision="omitted",
                    reason="family_token_cap",
                    slot=slot,
                    estimated_tokens=estimate,
                )
            )
            budget_decisions.append(
                ContextBudgetDecision(
                    decision="omit",
                    reason="family_token_cap",
                    source_item_id=item.source_item_id,
                    slot=slot,
                    estimated_tokens=estimate,
                )
            )
            omitted_ids.add(item.source_item_id)
            continue
        item_counts[family] += 1
        token_counts[family] += estimate
        kept.append(item)
    return kept


def _recent_raw_item_ids(
    request: ContextOperationRequest,
    candidates: Sequence[ContextSourceItem],
) -> set[str]:
    raw_items = [item for item in candidates if _is_raw_conversation_item(item)]
    if not raw_items:
        return set()
    recent_by_sequence = sorted(
        raw_items,
        key=lambda item: (
            item.source_scope or "",
            item.sequence_index if item.sequence_index is not None else -1,
            item.created_at.isoformat() if item.created_at is not None else "",
            item.source_item_id,
        ),
        reverse=True,
    )
    item_limit = request.budget_policy.recent_window_items
    if item_limit is None:
        return {item.source_item_id for item in recent_by_sequence}
    return {
        item.source_item_id for item in recent_by_sequence[: max(int(item_limit), 0)]
    }


def _select_item(
    *,
    request: ContextOperationRequest,
    item: ContextSourceItem,
    selected_items: list[ContextSourceItem],
    selected_ids: set[str],
    manifest: ContextReadManifest,
    reason: str,
) -> None:
    if item.source_item_id in selected_ids:
        return
    selected_items.append(item)
    selected_ids.add(item.source_item_id)
    manifest.selected.append(
        build_manifest_item(
            item,
            decision="selected",
            reason=reason,
            slot=_slot_for_item(request, item),
        )
    )


def _build_sections(
    request: ContextOperationRequest,
    selected_items: Sequence[ContextSourceItem],
) -> list:
    by_slot: dict[str, list[ContextSourceItem]] = defaultdict(list)
    for item in selected_items:
        by_slot[_slot_for_item(request, item)].append(item)
    sections = []
    for slot in request.placement_policy.ordered_slots:
        items = by_slot.get(slot)
        if not items:
            continue
        stability = cast(
            ContextStability,
            "stable_prefix"
            if slot in request.placement_policy.stable_prefix_slots
            else "volatile",
        )
        sections.append(
            build_section_for_items(
                section_id=f"{request.operation_id}:{slot}",
                slot=slot,
                title=slot.replace("_", " ").title() or DEFAULT_SECTION_TITLE,
                items=items,
                stability=stability,
            )
        )
    return sections


def _slot_for_item(request: ContextOperationRequest, item: ContextSourceItem) -> str:
    if item.visibility == "metadata_only":
        return "metadata_only"
    return request.placement_policy.slot_by_source_family.get(
        str(item.source_family),
        "metadata_only",
    )


def _sort_key(request: ContextOperationRequest, item: ContextSourceItem) -> tuple:
    slot = _slot_for_item(request, item)
    try:
        slot_order = request.placement_policy.ordered_slots.index(slot)
    except ValueError:
        slot_order = len(request.placement_policy.ordered_slots)
    return (
        slot_order,
        item.source_scope or "",
        item.sequence_index if item.sequence_index is not None else 10**9,
        item.created_at.isoformat() if item.created_at is not None else "",
        item.source_item_id,
    )


def _is_raw_conversation_item(item: ContextSourceItem) -> bool:
    return item.source_family in _CONVERSATION_FAMILIES


def _is_empty_model_content(item: ContextSourceItem) -> bool:
    return not (item.text and str(item.text).strip()) and not item.payload
