"""Contract tests for the common context engineering kernel."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rp.context_engineering.contracts import (
    ContextArtifact,
    ContextBudgetPolicy,
    ContextManifestItem,
    ContextOperationRequest,
    ContextReadManifest,
    ContextSourceItem,
    ContextValidationReport,
)
from rp.context_engineering.policies import (
    default_budget_policy,
    default_fallback_policy,
    default_placement_policy,
    default_validation_policy,
)


def _item(item_id: str = "item-1") -> ContextSourceItem:
    return ContextSourceItem(
        source_item_id=item_id,
        source_family="user_turn",
        text="hello",
    )


def test_minimal_context_operation_request_validates():
    request = ContextOperationRequest(
        operation_id="op-1",
        operation_kind="compact",
        runtime_family="test",
        source_items=[_item()],
        budget_policy=default_budget_policy(recent_window_items=1),
        placement_policy=default_placement_policy(),
        validation_policy=default_validation_policy(),
        fallback_policy=default_fallback_policy(),
    )

    assert request.operation_id == "op-1"
    assert request.source_items[0].source_family == "user_turn"


def test_unknown_fields_are_rejected_for_policy_models():
    with pytest.raises(ValidationError):
        ContextBudgetPolicy(extra_field=True)  # type: ignore[call-arg]


def test_source_item_preserves_scope_ordering_and_refs():
    item = ContextSourceItem(
        source_item_id="turn-1",
        source_family="assistant_turn",
        source_scope="setup_stage:foundation",
        sequence_index=3,
        source_ref="setup:foundation:history:3",
        recovery_refs=["draft:foundation"],
        text="assistant reply",
    )

    assert item.source_scope == "setup_stage:foundation"
    assert item.sequence_index == 3
    assert item.source_ref == "setup:foundation:history:3"
    assert item.recovery_refs == ["draft:foundation"]


def test_read_manifest_separates_decision_buckets():
    base = {
        "source_family": "user_turn",
        "visibility": "model_visible",
        "source_item_id": "item",
        "reason": "test",
    }
    manifest = ContextReadManifest(
        selected=[
            ContextManifestItem(
                **base,
                decision="selected",
            )
        ],
        omitted=[
            ContextManifestItem(
                **{**base, "source_item_id": "omitted"},
                decision="omitted",
            )
        ],
        hidden=[
            ContextManifestItem(
                **{**base, "source_item_id": "hidden", "visibility": "hidden"},
                decision="hidden",
            )
        ],
        forbidden=[
            ContextManifestItem(
                **{
                    **base,
                    "source_item_id": "forbidden",
                    "visibility": "forbidden",
                },
                decision="forbidden",
            )
        ],
        metadata_only=[
            ContextManifestItem(
                **{
                    **base,
                    "source_item_id": "metadata",
                    "visibility": "metadata_only",
                },
                decision="metadata_only",
            )
        ],
    )

    assert len(manifest.selected) == 1
    assert len(manifest.omitted) == 1
    assert len(manifest.hidden) == 1
    assert len(manifest.forbidden) == 1
    assert len(manifest.metadata_only) == 1


def test_previous_artifact_is_request_state_not_source_item():
    artifact = ContextArtifact(
        artifact_id="artifact-1",
        artifact_kind="compact_summary",
        schema_id="test.v1",
        schema_version="1",
        source_fingerprint="fingerprint",
        source_item_count=1,
        payload={"summary_lines": ["old"]},
        created_by="adapter",
        validation_report=ContextValidationReport(valid=True),
    )
    request = ContextOperationRequest(
        operation_id="op-1",
        operation_kind="compact",
        runtime_family="test",
        source_items=[_item("source-1")],
        budget_policy=default_budget_policy(),
        placement_policy=default_placement_policy(),
        validation_policy=default_validation_policy(),
        fallback_policy=default_fallback_policy(),
        previous_artifact=artifact,
    )

    assert request.previous_artifact is artifact
    assert [item.source_item_id for item in request.source_items] == ["source-1"]
