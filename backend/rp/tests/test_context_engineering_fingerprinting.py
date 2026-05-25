"""Fingerprinting tests for context engineering source items."""

from __future__ import annotations

from datetime import datetime, timezone

from rp.context_engineering.contracts import (
    ContextArtifact,
    ContextSourceItem,
    ContextValidationReport,
)
from rp.context_engineering.fingerprinting import (
    fingerprint_source_items,
    is_valid_prefix_artifact,
)


def _item(
    item_id: str,
    *,
    text: str = "hello",
    scope: str = "scope",
    sequence_index: int = 0,
    payload: dict[str, object] | None = None,
) -> ContextSourceItem:
    return ContextSourceItem(
        source_item_id=item_id,
        source_family="user_turn",
        source_scope=scope,
        sequence_index=sequence_index,
        source_ref=f"ref:{item_id}",
        recovery_refs=["draft:one"],
        text=text,
        payload=payload or {},
    )


def test_fingerprint_is_stable_across_payload_key_ordering():
    left = _item("item-1", payload={"b": 2, "a": 1})
    right = _item("item-1", payload={"a": 1, "b": 2})

    assert fingerprint_source_items([left]) == fingerprint_source_items([right])


def test_changing_text_changes_fingerprint():
    assert fingerprint_source_items([_item("item-1", text="a")]) != (
        fingerprint_source_items([_item("item-1", text="b")])
    )


def test_changing_scope_or_sequence_changes_fingerprint():
    base = fingerprint_source_items(
        [_item("item-1", scope="scope-a", sequence_index=1)]
    )

    assert base != fingerprint_source_items(
        [_item("item-1", scope="scope-b", sequence_index=1)]
    )
    assert base != fingerprint_source_items(
        [_item("item-1", scope="scope-a", sequence_index=2)]
    )


def test_created_at_does_not_change_fingerprint():
    left = _item("item-1")
    right = left.model_copy(update={"created_at": datetime.now(timezone.utc)})

    assert fingerprint_source_items([left]) == fingerprint_source_items([right])


def test_previous_artifact_payload_never_contributes_to_source_fingerprint():
    dropped = [_item("item-1"), _item("item-2", sequence_index=1)]
    prefix_fp = fingerprint_source_items(dropped[:1])
    artifact = ContextArtifact(
        artifact_id="artifact-1",
        artifact_kind="compact_summary",
        schema_id="test.v1",
        schema_version="1",
        source_fingerprint=prefix_fp,
        source_item_count=1,
        payload={"summary_lines": ["this payload is ignored by source hash"]},
        created_by="adapter",
        validation_report=ContextValidationReport(valid=True),
    )

    assert is_valid_prefix_artifact(
        previous_artifact=artifact,
        dropped_items=dropped,
    )
