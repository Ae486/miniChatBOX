"""Selection tests for the common context engineering kernel."""

from __future__ import annotations

from rp.context_engineering.contracts import ContextOperationRequest, ContextSourceItem
from rp.context_engineering.policies import (
    default_budget_policy,
    default_fallback_policy,
    default_placement_policy,
    default_validation_policy,
)
from rp.context_engineering.selection import select_context_sections


def _item(
    item_id: str,
    *,
    family: str = "user_turn",
    text: str = "text",
    sequence_index: int = 0,
    visibility: str = "model_visible",
    estimated_tokens: int = 1,
) -> ContextSourceItem:
    return ContextSourceItem(
        source_item_id=item_id,
        source_family=family,  # type: ignore[arg-type]
        source_scope="scope",
        sequence_index=sequence_index,
        visibility=visibility,  # type: ignore[arg-type]
        text=text,
        estimated_tokens=estimated_tokens,
    )


def _request(
    items: list[ContextSourceItem],
    *,
    recent_window_items: int | None = None,
    operation_budget_tokens: int | None = None,
    family_item_caps: dict[str, int] | None = None,
    family_token_caps: dict[str, int] | None = None,
) -> ContextOperationRequest:
    return ContextOperationRequest(
        operation_id="op-selection",
        operation_kind="compact",
        runtime_family="test",
        source_items=items,
        budget_policy=default_budget_policy(
            recent_window_items=recent_window_items,
            operation_budget_tokens=operation_budget_tokens,
            source_family_item_caps=family_item_caps,
            source_family_token_caps=family_token_caps,
        ),
        placement_policy=default_placement_policy(),
        validation_policy=default_validation_policy(),
        fallback_policy=default_fallback_policy(),
    )


def test_forbidden_hidden_and_metadata_only_items_are_excluded_and_reported():
    result = select_context_sections(
        _request(
            [
                _item("visible"),
                _item("hidden", visibility="hidden", text="secret hidden text"),
                _item("forbidden", visibility="forbidden"),
                _item("metadata", visibility="metadata_only"),
            ]
        )
    )

    assert [item.source_item_id for item in result.selected_items] == ["visible"]
    assert result.read_manifest.hidden[0].source_item_id == "hidden"
    assert result.read_manifest.forbidden[0].source_item_id == "forbidden"
    assert result.read_manifest.metadata_only[0].source_item_id == "metadata"
    assert "secret hidden text" not in result.read_manifest.hidden[0].model_dump_json()
    assert all("metadata" not in section.source_item_ids for section in result.sections)


def test_family_caps_omit_deterministically():
    item_cap_result = select_context_sections(
        _request(
            [
                _item("tool-1", family="tool_outcome", sequence_index=0),
                _item("tool-2", family="tool_outcome", sequence_index=1),
                _item("tool-3", family="tool_outcome", sequence_index=2),
            ],
            family_item_caps={"tool_outcome": 2},
        )
    )
    token_cap_result = select_context_sections(
        _request(
            [
                _item("tool-1", family="tool_outcome", sequence_index=0),
                _item("tool-2", family="tool_outcome", sequence_index=1),
            ],
            family_token_caps={"tool_outcome": 1},
        )
    )

    item_cap_reasons = {
        item.source_item_id: item.reason
        for item in item_cap_result.read_manifest.omitted
    }
    token_cap_reasons = {
        item.source_item_id: item.reason
        for item in token_cap_result.read_manifest.omitted
    }
    assert item_cap_reasons["tool-3"] == "family_item_cap"
    assert token_cap_reasons["tool-2"] == "family_token_cap"


def test_operation_budget_omits_when_budget_is_exceeded():
    result = select_context_sections(
        _request(
            [
                _item(
                    "state-1",
                    family="runtime_state",
                    sequence_index=0,
                    estimated_tokens=2,
                ),
                _item(
                    "state-2",
                    family="runtime_state",
                    sequence_index=1,
                    estimated_tokens=2,
                ),
            ],
            operation_budget_tokens=2,
        )
    )

    assert [item.source_item_id for item in result.selected_items] == ["state-1"]
    assert result.read_manifest.omitted[0].reason == "operation_budget_exceeded"


def test_recent_raw_window_keeps_latest_conversation_items_and_drops_only_raw():
    items = [
        _item(f"turn-{index}", sequence_index=index, family="user_turn")
        for index in range(5)
    ]
    items.extend(
        [
            _item("runtime", family="runtime_state", sequence_index=10),
            _item("tool", family="tool_outcome", sequence_index=11),
            _item("hidden", visibility="hidden", sequence_index=12),
        ]
    )

    result = select_context_sections(_request(items, recent_window_items=2))

    assert [item.source_item_id for item in result.recent_raw_items] == [
        "turn-3",
        "turn-4",
    ]
    assert [item.source_item_id for item in result.compactable_dropped_items] == [
        "turn-0",
        "turn-1",
        "turn-2",
    ]
    assert all(
        item.source_family in {"user_turn", "assistant_turn"}
        for item in result.compactable_dropped_items
    )


def test_section_order_follows_placement_policy():
    result = select_context_sections(
        _request(
            [
                _item("turn", sequence_index=1),
                _item("runtime", family="runtime_state", sequence_index=0),
                _item("tool", family="tool_outcome", sequence_index=2),
            ]
        )
    )

    assert [section.slot for section in result.sections] == [
        "runtime_overlay",
        "recent_raw",
        "tool_outcomes",
    ]
